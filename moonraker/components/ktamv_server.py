# Camera Alignment Component for Moonraker
#
# Copyright (C) 2025 Creatbot
#
# This file may be distributed under the terms of the GNU GPLv3 license.
from __future__ import annotations
import os
import logging
import cv2
import asyncio
import time
import io
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
from ..utils import json_wrapper as jsonw
from ..common import RequestType, KlippyState, APITransport
import time
import requests
from requests.exceptions import InvalidURL, ConnectionError
from dataclasses import dataclass
# Annotation imports
from typing import (
    TYPE_CHECKING,
    List,
    Any,
    Dict,
    Tuple,
    Optional
)

if TYPE_CHECKING:
    from ..confighelper import ConfigHelper
    from ..common import WebRequest
    from .klippy_apis import KlippyAPI

IMG_W, IMG_H = 640, 480
SAVE_ROOT_DIR = "/tmp/nozzle_detection_results"

@dataclass
class NozzleAlgo:
    pre_idx: int
    detector: cv2.SimpleBlobDetector
    color: tuple
    aid: int

@dataclass
class AlgorithmConfig:
    idx: int
    mode: str
    color: Tuple[int, int, int]
    algo_id: int

@dataclass
class BlobParams:
    min_area: int
    max_area: int
    min_circularity: float
    min_convexity: float
    filter_by_area: bool
    filter_by_circularity: bool
    filter_by_convexity: bool

    def to_opencv_params(self, scale: float = 1.0) -> cv2.SimpleBlobDetector_Params:
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = self.filter_by_area
        params.filterByCircularity = self.filter_by_circularity
        params.filterByConvexity = self.filter_by_convexity

        params.minArea = int(self.min_area * (scale ** 2))
        params.maxArea = int(self.max_area * (scale ** 2))
        params.minCircularity = self.min_circularity
        params.minConvexity = self.min_convexity

        return params

class AlgorithmSelector:
    def __init__(self, configs: List[AlgorithmConfig]):
        self.configs = configs
        self.current_algorithm: Optional[int] = None
        self.performance_stats: Dict[int, Dict[str, int]] = {}
        self.reset_stats()

    def reset_stats(self):
        for cfg in self.configs:
            self.performance_stats[cfg.algo_id] = {
                'success': 0,
                'attempts': 0,
                'recent_success': 0
            }

    def update_performance(self, algo_id: int, success: bool):
        if algo_id not in self.performance_stats:
            return

        stats = self.performance_stats[algo_id]
        stats['attempts'] += 1
        if success:
            stats['success'] += 1
            stats['recent_success'] += 1

    def get_best_algorithm(self, exclude_current: bool = False) -> Optional[int]:
        if not self.performance_stats:
            return None

        valid_algorithms = [algo_id for algo_id in self.performance_stats
                          if not exclude_current or algo_id != self.current_algorithm]

        if not valid_algorithms:
            return None

        return max(valid_algorithms,
                  key=lambda x: self.performance_stats[x].get('recent_success', 0))
class CameraAglin(APITransport):
    def __init__(self, config: ConfigHelper) -> None:
        super().__init__()
        self.server = config.get_server()
        self.camera_url = config.get("nozzle_cam_url", "http://127.0.0.1/webrtc/api/frame.jpeg?src=Alignment_RAW")
        self.save_image = config.getboolean('save_image', True)
        self.cv_timeout = config.getfloat('cv_timeout', 20)
        self.min_matches = config.getint('min_matches', 3)
        self.frame_width = config.getint('frame_width', IMG_W)
        self.frame_height = config.getint('frame_height', IMG_H)
        self.max_frame_rate = config.getint('max_frame_rate', 15)

        self.processed_frame = None
        self.standby_image = None
        self.transform_matrix = None
        self.preview_running = False
        self.update_static_image = True
        self.printer_state = "unknown"

        self.request_results = {}
        self.max_stored_results = 100
        self.last_frame_time = 0

        self.server.register_event_handler("server:klippy_ready", self._handle_klippy_ready)
        self.server.register_event_handler("server:klippy_disconnect", self._handle_klippy_disconnect)
        self.server.register_event_handler("server:klippy_shutdown", self._handle_klippy_shutdown)

        self.camera_handler = CameraStreamHandler(self.camera_url)
        self.detection_manager = Ktamv_Detection_Manager(self.camera_url, self.save_image)

        self.server.register_remote_method(
            "get_nozzle_position", self.get_nozzle_position)
        self.server.register_remote_method(
            "calculate_camera_to_space_matrix", self.calculate_camera_to_space_matrix)
        self.server.register_endpoint("/server/ktamv/preview", RequestType.GET, self._handle_get_preview)
        self.server.register_endpoint("/server/ktamv/status", RequestType.GET, self._handle_get_status)
        self.server.register_endpoint("/server/ktamv/overlay_stream", RequestType.GET, self._handle_overlay_stream)
        self.server.register_endpoint("/server/ktamv/simple_overlay", RequestType.GET, self._handle_simple_overlay)

    @property
    def klippy_apis(self) -> KlippyAPI:
        return self.server.lookup_component("klippy_apis")

    async def _handle_klippy_ready(self) -> None:
        previous_state = self.printer_state
        self.printer_state = "ready"
        logging.info(f"Klippy state changed from {previous_state} to ready")

    async def _handle_klippy_disconnect(self) -> None:
        previous_state = self.printer_state
        self.printer_state = "disconnected"
        logging.warning(f"Klippy disconnected from previous state: {previous_state}")

    async def _handle_klippy_shutdown(self) -> None:
        try:
            self.printer_state = "shutdown"
            self.request_results.clear()
            self.camera_handler.close_stream()
        except Exception as e:
            logging.error(f"Error during shutdown: {str(e)}")

    async def get_nozzle_position(self) -> None:
        if hasattr(self, '_nozzle_detection_task') and not self._nozzle_detection_task.done():
            logging.warning("Nozzle detection already in progress, ignoring new request")
        async def _notify_result(result_dict: dict):
            try:
                await self.klippy_apis.ktamv_result(result_dict)
            except Exception as e:
                logging.warning(f"Failed to notify Klipper of result: {e}")

        async def detection_task():
            start_time = time.time()
            try:
                eventloop = self.server.get_event_loop()
                position = await eventloop.run_in_thread(
                    self.detection_manager.recursively_find_nozzle_position,
                    self._put_frame,
                    self.min_matches,
                    self.cv_timeout
                )
                runtime = time.time() - start_time
                if position is not None:
                    result = {
                        "function": "get_nozzle_position",
                        "status": "success",
                        "position": position,
                        "runtime": round(runtime, 3),
                        "message": "Nozzle position detected successfully!"
                    }
                    logging.info(f"Nozzle detected at {position} in {runtime:.3f}s")
                else:
                    result = {
                        "function": "get_nozzle_position",
                        "status": "error",
                        "runtime": round(runtime, 3),
                        "message": "Failed to detect nozzle position!"
                    }
                    logging.warning("Nozzle detection returned None")
                await _notify_result(result)

            except asyncio.CancelledError:
                logging.info("Nozzle detection task was cancelled")
                raise
            except Exception as e:
                runtime = time.time() - start_time
                error_msg = f"Internal error during detection: {str(e)}"
                logging.exception(error_msg)
                result = {
                    "function": "get_nozzle_position",
                    "status": "error",
                    "runtime": round(runtime, 3),
                    "message": error_msg
                }
                await _notify_result(result)
        self._nozzle_detection_task = asyncio.create_task(detection_task())
        logging.info("Nozzle detection task started")

    async def calculate_camera_to_space_matrix(self, calibration_points: List[Tuple[List[float], List[float]]]) -> None:
        try:
            logging.info("Calculating camera to space matrix")
            if not calibration_points:
                self.klippy_apis.ktamv_result({
                    "function": "calculate_camera_to_space_matrix",
                    "status": "error",
                    "message": "Calibration Points not provided"
                })
                return
            n = len(calibration_points)
            real_coords = np.empty((n, 2))
            pixel_coords = np.empty((n, 2))

            for i, (r, p) in enumerate(calibration_points):
                real_coords[i] = r
                pixel_coords[i] = p

            x, y = pixel_coords[:, 0], pixel_coords[:, 1]
            A = np.vstack([x**2, y**2, x * y, x, y, np.ones(n)]).T
            transform = np.linalg.lstsq(A, real_coords, rcond=None)

            self.transform_matrix = transform[0].T

            await self.klippy_apis.ktamv_result({
                "function": "calculate_camera_to_space_matrix",
                "status": "success",
                "matrix": self.transform_matrix.tolist(),
                "message": "Matrix calculated successfully"
            })
        except Exception as e:
            error_msg = f"Error calculating camera to space matrix: {str(e)}"
            self.klippy_apis.ktamv_result({
                    "function": "calculate_camera_to_space_matrix",
                    "status": "error",
                    "message": error_msg
                })
            logging.error(error_msg)

    def _draw_on_frame(self, frame: Image.Image, data: Dict[str, Any] = None) -> Image.Image:
        try:
            draw = ImageDraw.Draw(frame)
            try:
                font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='Arial')), 14)
            except:
                font = ImageFont.load_default()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            draw.text((10, 10), f"Time: {timestamp}", fill=(0, 255, 0), font=font)
            status_text = "Running" if self.preview_running else "Idle"
            draw.text((10, 30), f"Status: {status_text}", fill=(0, 255, 0), font=font)
            if data:
                y_offset = 50
                for key, value in data.items():
                    if y_offset < self.frame_height - 20:
                        text = f"{key}: {value}"
                        draw.text((10, y_offset), text, fill=(0, 255, 255), font=font)
                        y_offset += 20
            return frame
        except Exception as e:
            logging.error(f"Error drawing on frame: {e}")
            return frame

    def _frame_to_base64(self, frame: Image.Image) -> str:
        try:
            buffered = io.BytesIO()
            frame.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            logging.error(f"Error converting frame to base64: {e}")
            raise

    async def _get_printer_position(self) -> Dict[str, Any]:
        logging.debug("Querying printer position")
        if self.printer_state != "ready":
            logging.warning("Cannot get printer position: Klippy not ready")
            return {"status": "error", "message": "Klippy not ready"}
        try:
            try:
                result = await self.klippy_apis.query_objects({"toolhead": None})
            except Exception:
                error_msg = "Invalid response format from Klippy"
                logging.error(error_msg)
                return {"status": "error", "message": error_msg}
            if "toolhead" in result and "position" in result["toolhead"]:
                position = result["toolhead"]["position"]
                logging.debug(f"Retrieved printer position: X{position[0]:.2f}, Y{position[1]:.2f}, Z{position[2]:.2f}")
                return {
                    "status": "success",
                    "position": {
                        "x": position[0],
                        "y": position[1],
                        "z": position[2]
                    }
                }
            else:
                error_msg = 'Position data not available in Klippy response'
                logging.error(error_msg)
                return {"status": "error", "message": error_msg}
        except Exception as e:
            error_msg = f"Failed to get printer position: {str(e)}"
            logging.error(error_msg)
            return {"status": "error", "message": error_msg}

    def _put_frame(self, frame):
        try:
            current_time = time.time()
            if current_time - self.last_frame_time < 1.0 / self.max_frame_rate:
                return
            self.last_frame_time = current_time
            if isinstance(frame, np.ndarray):
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.processed_frame = Image.fromarray(rgb_frame)
            else:
                self.processed_frame = Image.fromarray(frame)
            self.update_static_image = True
        except Exception as e:
            logging.error(f"Frame update error: {e}")

    async def _handle_get_preview(
        self, web_request: WebRequest
    ) -> Dict[str, Any]:
        try:
            show_data = web_request.get_boolean('show_data', True)
            frame_bytes = None

            def put_frame(frame):
                nonlocal frame_bytes
                try:
                    if isinstance(frame, np.ndarray):
                        if frame.size == 0:
                            raise ValueError("Empty frame array")
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_frame = Image.fromarray(rgb_frame)
                    else:
                        pil_frame = Image.fromarray(frame)

                    if show_data:
                        latest_data = None
                        if self.request_results:
                            latest_key = max(self.request_results.keys())
                            latest_result = self.request_results[latest_key]
                            if latest_result.data:
                                try:
                                    latest_data = jsonw.loads(latest_result.data)
                                except Exception:
                                    pass
                        pil_frame = self._draw_on_frame(pil_frame, latest_data)

                    img_io = io.BytesIO()
                    pil_frame.save(img_io, format="JPEG", quality=85)
                    img_io.seek(0)
                    frame_bytes = img_io.read()
                except Exception as e:
                    logging.exception(f"Error processing frame in put_frame: {e}")
                    frame_bytes = None
            try:
                self.detection_manager.get_preview_frame(put_frame)
            except Exception as e:
                logging.warning(f"Failed to get camera frame: {e}, using standby image")
                if self.standby_image is not None:
                    try:
                        standby_array = np.array(self.standby_image.copy())
                        put_frame(standby_array)
                    except Exception as ex:
                        logging.error(f"Failed to use standby image: {ex}")
                        frame_bytes = None
            if frame_bytes:
                return {
                    "status": "success",
                    "image": base64.b64encode(frame_bytes).decode('utf-8'),
                    "format": "jpeg"
                }
            else:
                return {
                    "status": "error",
                    "message": "Could not get preview frame"
                }
        except Exception as e:
            logging.exception(f"Unexpected error in _handle_get_preview: {e}")
            return {
                "status": "error",
                "message": f"Internal error: {str(e)}"
            }

    async def _handle_get_status(self, web_request: WebRequest) -> Dict[str, Any]:
        try:
            status = {
                'is_running': getattr(self, 'preview_running', False),
                'klippy_status': self.printer_state,
                'camera_status': getattr(self, 'camera_url', None),
                'request_stats': {
                    'stored_results': len(self.request_results),
                    'max_stored_results': self.max_stored_results
                },
                'timestamp': time.time()
            }
            logging.debug("Status information requested")
            return status
        except Exception as e:
            error_msg = f"Failed to retrieve status information: {str(e)}"
            self._record_error(error_msg)
            return {'status': 'error', 'message': error_msg}

    async def _handle_overlay_stream(
        self, web_request: WebRequest
    ) -> None:
        try:
            web_request.set_header('Content-Type', 'multipart/x-mixed-replace; boundary=--ktamv_overlay_boundary')
            web_request.set_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            web_request.set_header('Pragma', 'no-cache')
            web_request.set_header('Expires', '0')

            connection = web_request.connection
            eventloop = self.server.get_event_loop()

            while connection.is_alive():
                frame_bytes = None

                def put_frame(frame):
                    nonlocal frame_bytes
                    try:
                        if isinstance(frame, np.ndarray):
                            if frame.size == 0:
                                raise ValueError("Empty frame array")
                            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                            frame_bytes = buffer.tobytes()
                        else:
                            raise ValueError("Unexpected frame type")
                    except Exception as e:
                        logging.exception(f"Error processing frame for overlay stream: {e}")
                        frame_bytes = None

                await eventloop.run_in_thread(
                    self.detection_manager.get_preview_frame, put_frame
                )

                if frame_bytes:
                    connection.write(b"--ktamv_overlay_boundary\r\n")
                    connection.write(b"Content-Type: image/jpeg\r\n")
                    connection.write(f"Content-Length: {len(frame_bytes)}\r\n".encode('utf-8'))
                    connection.write(b"X-Timestamp: " + str(time.time()).encode('utf-8') + b"\r\n")
                    connection.write(b"\r\n")
                    connection.write(frame_bytes)
                    connection.write(b"\r\n")
                    await connection.drain()

                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logging.info("Overlay stream request cancelled")
        except Exception as e:
            logging.exception(f"Overlay stream error: {e}")

        web_request.close()
        logging.info("Overlay stream ended")

    async def _handle_simple_overlay(
        self, web_request: WebRequest
    ) -> bytes:
        try:
            logging.info(f"Simple overlay frame requested from: {web_request.client_address}")
            frame_bytes = None

            def put_frame(frame):
                nonlocal frame_bytes
                try:
                    if isinstance(frame, np.ndarray):
                        if frame.size == 0:
                            raise ValueError("Empty frame array")
                        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                        frame_bytes = buffer.tobytes()
                        logging.debug(f"Encoded simple overlay frame, size: {len(frame_bytes)} bytes")
                    else:
                        raise ValueError("Unexpected frame type")
                except Exception as e:
                    logging.exception(f"Error processing frame for simple overlay: {e}")
                    frame_bytes = None
            eventloop = self.server.get_event_loop()
            await eventloop.run_in_thread(
                self.detection_manager.get_preview_frame, put_frame
            )

            if frame_bytes:
                web_request.set_header('Content-Type', 'image/jpeg')
                web_request.set_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                web_request.set_header('Pragma', 'no-cache')
                web_request.set_header('Expires', '0')
                web_request.set_header('Content-Length', str(len(frame_bytes)))
                web_request.set_header('X-Timestamp', str(time.time()))
                return frame_bytes
            else:
                logging.error("Failed to generate simple overlay frame")
                web_request.set_header('Content-Type', 'text/plain')
                return b"Failed to generate overlay frame"

        except Exception as e:
            logging.exception(f"Simple overlay error: {e}")
            web_request.set_header('Content-Type', 'text/plain')
            return f"Overlay generation failed: {str(e)}".encode('utf-8')

class ImagePreprocessor:
    @staticmethod
    def preprocess_method_0(img: np.ndarray, scale: float = 1.0) -> np.ndarray:
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)

        blur_ksize = ImagePreprocessor._get_odd_blur_size(5, scale)
        y = cv2.GaussianBlur(y, (blur_ksize, blur_ksize), 3 * scale)

        block_size = ImagePreprocessor._get_valid_block_size(25, scale)
        y = cv2.adaptiveThreshold(
            y, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, 2
        )
        return cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def preprocess_method_1(img: np.ndarray, scale: float = 1.0) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

        blur_ksize = ImagePreprocessor._get_odd_blur_size(5, scale)
        thresh = cv2.GaussianBlur(thresh, (blur_ksize, blur_ksize), 3 * scale)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def preprocess_method_3(img: np.ndarray, scale: float = 1.0) -> np.ndarray:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_abs = np.uint8(np.absolute(laplacian))

        canny_low = int(100 * scale)
        canny_high = int(200 * scale)
        canny = cv2.Canny(gray, canny_low, canny_high)

        combined = cv2.bitwise_or(laplacian_abs, canny)
        return cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _get_odd_blur_size(base_size: int, scale: float) -> int:
        blur_size = int(base_size * scale)
        blur_size = max(3, blur_size if blur_size % 2 == 1 else blur_size + 1)
        return blur_size

    @staticmethod
    def _get_valid_block_size(base_size: int, scale: float) -> int:
        block_size = int(base_size * scale)
        block_size = max(3, block_size if block_size % 2 == 1 else block_size + 1)
        return block_size
class CameraStreamHandler:
    def __init__(self, camera_url):
        self.camera_url = camera_url
        self.session = requests.Session()

    def get_single_frame(self):
        try:
            response = self.session.get(self.camera_url, stream=True, timeout=5)
            response.raise_for_status()
            image_data = io.BytesIO(response.content)
            pil_image = Image.open(image_data)
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return opencv_image
        except Exception as e:
            logging.error(f"Error getting frame: {str(e)}")
            return None

    def can_read_stream(self):
        try:
            with self.session.get(self.camera_url) as _:
                return True
        except InvalidURL as _:
            logging.error(f"Could not read nozzle camera address, got InvalidURL error {self.camera_url}")
            raise ValueError(f"Invalid camera URL: {self.camera_url}")
        except ConnectionError as _:
            logging.error(f"Failed to establish connection with nozzle camera {self.camera_url}")
            raise ConnectionError(f"Cannot connect to camera: {self.camera_url}")
        except Exception as e:
            logging.error(f"Nozzle camera request failed {str(e)}")
            raise Exception(f"Camera request failed: {str(e)}")

    def close_stream(self):
        if self.session is not None:
            self.session.close()
            self.session = None

class Ktamv_Detection_Manager:
    CFG = [
        AlgorithmConfig(0, 'standard', (0, 0, 255), 1),
        AlgorithmConfig(1, 'standard', (0, 255, 0), 2),
        AlgorithmConfig(3, 'standard', (255, 0, 255), 3),
        AlgorithmConfig(0, 'relaxed', (255, 0, 0), 4),
        AlgorithmConfig(1, 'relaxed', (39, 127, 255), 5),
        AlgorithmConfig(3, 'relaxed', (0, 255, 255), 6),
    ]

    BASE_WIDTH = 640
    BASE_HEIGHT = 480
    DEFAULT_MIN_MATCHES = 5
    DEFAULT_TIMEOUT = 10.0
    STATS_RESET_INTERVAL = 100

    def __init__(self, camera_url: str, save_image: bool = False, *args, **kwargs):
        self.__io = CameraStreamHandler(camera_url=camera_url)
        self.save_image = save_image

        self._blob_params = self._initialize_blob_params()
        self._current_algorithm: Optional[int] = None
        self._last_keypoint_size: Optional[float] = None

        self._stats = self._initialize_statistics()

        self._detector_cache: Dict[Tuple[str, float], Any] = {}
        self._image_center_cache: Optional[Tuple[int, int]] = None

        if self.save_image:
            self._initialize_save_directories()

    def _initialize_save_directories(self):
        if not os.path.exists(SAVE_ROOT_DIR):
            os.makedirs(SAVE_ROOT_DIR, exist_ok=True)
            logging.info(f"Created image save root directory: {SAVE_ROOT_DIR}")

    def _initialize_blob_params(self) -> Dict[str, BlobParams]:
        return {
            'standard': BlobParams(
                min_area=180,
                max_area=1000,
                min_circularity=0.7,
                min_convexity=0.8,
                filter_by_area=True,
                filter_by_circularity=True,
                filter_by_convexity=False
            ),
            'relaxed': BlobParams(
                min_area=180,
                max_area=1500,
                min_circularity=0.6,
                min_convexity=0.6,
                filter_by_area=True,
                filter_by_circularity=True,
                filter_by_convexity=False
            )
        }

    def _initialize_statistics(self) -> Dict[str, Any]:
        return {
            'success_counts': {cfg.algo_id: 0 for cfg in self.CFG},
            'frame_count': 0,
            'fail_count': 0,
            'total_attempts': 0
        }

    def _get_detector(self, mode: str, scale: float) -> cv2.SimpleBlobDetector:
        cache_key = (mode, round(scale, 2))

        if cache_key not in self._detector_cache:
            params = self._blob_params[mode].to_opencv_params(scale)
            self._detector_cache[cache_key] = cv2.SimpleBlobDetector_create(params)

        return self._detector_cache[cache_key]

    def _preprocess_image(self, img: np.ndarray, method_idx: int, scale: float) -> np.ndarray:
        preprocessor_map = {
            0: ImagePreprocessor.preprocess_method_0,
            1: ImagePreprocessor.preprocess_method_1,
            3: ImagePreprocessor.preprocess_method_3,
        }

        if method_idx in preprocessor_map:
            return preprocessor_map[method_idx](img, scale)

        return img

    def _get_image_center(self, img: np.ndarray) -> Tuple[int, int]:
        rows, cols = img.shape[:2]
        return (cols // 2, rows // 2)

    def _calculate_scale(self, img_shape: Tuple[int, int, int]) -> float:
        rows, cols = img_shape[:2]
        return max(cols / self.BASE_WIDTH, rows / self.BASE_HEIGHT)

    def _get_config_by_id(self, algo_id: int) -> Optional[AlgorithmConfig]:
        for config in self.CFG:
            if config.algo_id == algo_id:
                return config
        return None

    def _update_detection_stats(self, algo_id: Optional[int], success: bool):
        if success and algo_id is not None:
            self._stats['success_counts'][algo_id] += 1

        self._stats['frame_count'] += 1
        if self._stats['frame_count'] >= self.STATS_RESET_INTERVAL:
            self._stats = self._initialize_statistics()

    def _detect_with_algorithm(self, img: np.ndarray, config: AlgorithmConfig, scale: float) -> Optional[Tuple[Tuple[int, int], float, np.ndarray]]:
        detector = self._get_detector(config.mode, scale)
        processed_img = self._preprocess_image(img, config.idx, scale)

        keypoints = detector.detect(processed_img)
        if not keypoints:
            return None

        image_center = self._get_image_center(img)
        closest_kp = min(keypoints,
                        key=lambda kp: np.linalg.norm(np.array(kp.pt) - np.array(image_center)))

        center = (int(closest_kp.pt[0]), int(closest_kp.pt[1]))
        size = closest_kp.size

        return center, size, processed_img

    def _draw_detection_result(self, img: np.ndarray, center: Optional[Tuple[int, int]],
                             size: Optional[float], color: Optional[Tuple[int, int, int]],
                             processed_img: Optional[np.ndarray] = None) -> np.ndarray:
        rows, cols = img.shape[:2]
        cx, cy = self._get_image_center(img)

        img = cv2.line(img, (cx, 0), (cx, rows), (0, 0, 0), 2)
        img = cv2.line(img, (0, cy), (cols, cy), (0, 0, 0), 2)
        img = cv2.line(img, (cx, 0), (cx, rows), (255, 255, 255), 1)
        img = cv2.line(img, (0, cy), (cols, cy), (255, 255, 255), 1)

        cross_size = max(3, int(min(rows, cols) * 0.01))

        if center is not None and size is not None:
            radius = int(np.around(size / 2))
            circle_frame = cv2.circle(
                img=img,
                center=center,
                radius=radius,
                color=(0, 255, 0),
                thickness=-1,
                lineType=cv2.LINE_AA
            )
            img = cv2.addWeighted(circle_frame, 0.4, img, 0.6, 0)
            img = cv2.circle(
                img=img,
                center=center,
                radius=radius,
                color=(0, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA
            )
            x, y = center
            img = cv2.line(img, (x - cross_size, y), (x + cross_size, y), (255, 255, 255), 2)
            img = cv2.line(img, (x, y - cross_size), (x, y + cross_size), (255, 255, 255), 2)
        else:
            default_radius = int(min(rows, cols) * 0.035)
            img = cv2.circle(
                img=img,
                center=(cx, cy),
                radius=default_radius,
                color=(0, 0, 0),
                thickness=3,
                lineType=cv2.LINE_AA
            )
            img = cv2.circle(
                img=img,
                center=(cx, cy),
                radius=default_radius + 1,
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA
            )

        return img


    def _create_call_dir(self) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        call_dir = os.path.join(SAVE_ROOT_DIR, f"frame_{timestamp}")
        os.makedirs(call_dir, exist_ok=True)
        logging.info(f"Created image save folder for this call: {call_dir}")
        return call_dir

    def _save_drawn_image(self, call_dir, drawn_img, frame_idx):
        img_filename = f"frame_{frame_idx}_{time.strftime('%Y%m%d_%H%M%S_%f', time.localtime())[:-3]}.jpg"
        img_save_path = os.path.join(call_dir, img_filename)
        try:
            rgb_img = cv2.cvtColor(drawn_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            pil_img.save(img_save_path, format='JPEG', quality=90)
        except Exception as e:
            logging.error(f"Failed to save image as JPEG: {str(e)}")
            try:
                img_filename = img_filename.replace('.jpg', '.png')
                img_save_path = os.path.join(call_dir, img_filename)
                pil_img.save(img_save_path, format='PNG')
            except Exception as fallback_e:
                logging.error(f"Both JPEG and PNG save failed: {str(fallback_e)}")

    def recursively_find_nozzle_position(self, put_frame_func, min_matches, timeout):
        if self.save_image:
            current_call_dir = self._create_call_dir()
            current_frame_idx = 0

        start, counter, last = time.time(), {}, None
        while time.time() - start < timeout:
            frame = self.__io.get_single_frame()
            if frame is None:
                logging.error("Failed to get frame from camera")
                continue
            pos, vis = self.nozzleDetection(frame)
            if vis is not None:
                if self.save_image:
                    self._save_drawn_image(current_call_dir, vis, current_frame_idx)
                    current_frame_idx += 1
                put_frame_func(vis)
            if pos is None:
                continue
            key = (int(pos[0]), int(pos[1]))
            counter[key] = counter.get(key, 0) + 1
            if counter[key] >= min_matches:
                break
            last = pos
            time.sleep(0.3)
        return last

    def get_preview_frame(self, put_frame_func):
        _, vis = self.nozzleDetection(self.__io.get_single_frame())
        if vis is not None:
            put_frame_func(vis)

    def nozzleDetection(self, img):
        return self.nozzle_detection(img)

    def nozzle_detection(self, img: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[np.ndarray]]:
        if img is None:
            return None, None

        scale = self._calculate_scale(img.shape)
        image_center = self._get_image_center(img)

        if self._current_algorithm is not None:
            config = self._get_config_by_id(self._current_algorithm)
            if config:
                result = self._detect_with_algorithm(img, config, scale)
                if result:
                    center, size, processed_img = result
                    self._update_detection_stats(config.algo_id, success=True)
                    return center, self._draw_detection_result(img.copy(), center, size, config.color, processed_img)

        for config in self.CFG:
            result = self._detect_with_algorithm(img, config, scale)
            if result:
                center, size, processed_img = result
                self._current_algorithm = config.algo_id
                self._update_detection_stats(config.algo_id, success=True)
                return center, self._draw_detection_result(img.copy(), center, size, config.color, processed_img)

        self._update_detection_stats(None, success=False)
        return None, self._draw_detection_result(img.copy(), None, None, None, None)

def load_component(config: ConfigHelper) -> CameraAglin:
    return CameraAglin(config)
