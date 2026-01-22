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
    Tuple
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
    uv = [None, None]
    __algorithm = None
    CFG = [
        (0, 'standard', (0, 0, 255), 1),
        (1, 'standard', (0, 255, 0), 2),
        (2, 'standard', (39, 255, 127), 3),
        (3, 'standard', (255, 0, 255), 4),
        (0, 'relaxed',  (255, 0, 0), 5),
        (1, 'relaxed',  (39, 127, 255), 6),
        (2, 'relaxed',  (39, 255, 127), 7),
        (3, 'relaxed',  (0, 255, 255), 8),
    ]
    def __init__(self, camera_url, save_image=False, *a, **kw):
        self.__io = CameraStreamHandler(camera_url=camera_url)
        self._base_params = self._setup_base_params()
        self._algos = self._build_algorithms()
        self._success = [0]*(len(self.CFG)+1)
        self._fail_cnt=0
        self._last_size=None
        self._frame_cnt=0
        self.save_image = save_image
        if self.save_image:
            self._creat_save_root()

    def _creat_save_root(self):
        if not os.path.exists(SAVE_ROOT_DIR):
            os.makedirs(SAVE_ROOT_DIR, exist_ok=True)
            logging.info(f"Created image save root directory: {SAVE_ROOT_DIR}")

    def _create_call_dir(self):
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
        if img is None:
            return None, None
        center = self._detect_blob(img)
        vis = self._draw(img.copy(), center)
        return center, vis

    def _setup_base_params(self):
        return {
            'standard': {
                'minArea': 200, 'maxArea': 1000,
                'minCircularity': 0.8,
                'minConvexity': 0.8,
                'filterByArea': True,
                'filterByCircularity': True,
                'filterByConvexity': False,
                'filterByInertia': True,
                'minInertiaRatio': 0.7
            },
            'relaxed': {
                'minArea': 150, 'maxArea': 1500,
                'minCircularity': 0.6,
                'minConvexity': 0.7,
                'filterByArea': True,
                'filterByCircularity': True,
                'filterByConvexity': False,
                'filterByInertia': True,
                'minInertiaRatio': 0.5
            },
        }

    def _build_algorithms(self):
        def make(pkey):
            p = cv2.SimpleBlobDetector_Params()
            src = self._base_params[pkey]
            for k, v in src.items():
                setattr(p, k, v)
            return cv2.SimpleBlobDetector_create(p)
        return [NozzleAlgo(pre, make(key), color, aid)
                for pre, key, color, aid in self.CFG]

    def _preprocess(self, img, idx):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if idx == 0:
            y = cv2.GaussianBlur(gray, (5, 5), 3)
            return cv2.adaptiveThreshold(y, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 25, 2)
        if idx == 1:
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
            return cv2.GaussianBlur(th, (5, 5), 3)
        if idx == 2:
            return cv2.medianBlur(gray, 3)
        return gray

    def _detect_blob(self, img):
        if self.__algorithm:
            algo = self._algos[self.__algorithm-1]
            pt = self._try_algo(img, algo)
            if pt:
                self._success[algo.aid] += 1
                self._fail_cnt = 0
                return pt
            self._fail_cnt += 1
            if self._fail_cnt >= 3:
                self.__algorithm = None
        for algo in sorted(self._algos, key=lambda x: self._success[x.aid], reverse=True):
            pt = self._try_algo(img, algo)
            if pt:
                self.__algorithm = algo.aid
                self._fail_cnt = 0
                return pt
        return None

    def _try_algo(self, img, algo):
        kps = algo.detector.detect(self._preprocess(img, algo.pre_idx))
        if not kps:
            return None
        kp = min(kps, key=lambda k: np.linalg.norm(np.array(k.pt) - np.array([IMG_W//2, IMG_H//2])))
        x, y, s = kp.pt[0], kp.pt[1], kp.size
        if not (20 <= x <= IMG_W-20 and 20 <= y <= IMG_H-20):
            return None
        if self._last_size and not (0.5 <= s / self._last_size <= 2):
            return None
        self._last_size = s
        return int(round(x)), int(round(y))

    def _draw(self, img, center):
        cx, cy = IMG_W//2, IMG_H//2
        if center:
            cv2.circle(img, center, int(self._last_size//2), (0, 255, 0), -1)
            cv2.line(img, (center[0]-5, center[1]), (center[0]+5, center[1]), (255, 255, 255), 2)
            cv2.line(img, (center[0], center[1]-5), (center[0], center[1]+5), (255, 255, 255), 2)
        else:
            r = 17
            cv2.circle(img, (cx, cy), r, (0, 0, 0), 3)
            cv2.circle(img, (cx, cy), r+1, (0, 0, 255), 1)
        cv2.line(img, (cx, 0), (cx, IMG_H), (0, 0, 0), 2)
        cv2.line(img, (0, cy), (IMG_W, cy), (0, 0, 0), 2)
        cv2.line(img, (cx, 0), (cx, IMG_H), (255, 255, 255), 1)
        cv2.line(img, (0, cy), (IMG_W, cy), (255, 255, 255), 1)
        return img

def load_component(config: ConfigHelper) -> CameraAglin:
    return CameraAglin(config)
