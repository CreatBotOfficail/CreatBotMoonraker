from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict, Any, List
from tornado.httpclient import AsyncHTTPClient, HTTPRequest, HTTPError
from urllib.parse import quote

if TYPE_CHECKING:
    from ..confighelper import ConfigHelper

API_WEBRTC_URL = "http://localhost:1984/api/webrtc"


class WebRTCBridge:
    def __init__(self, config: ConfigHelper):
        default_cameras = config.getlist("camera_name", ["Camera"])
        self.default_cameras = []
        for camera in default_cameras:
            self.default_cameras.extend(
                [cam.strip() for cam in camera.split(",") if cam.strip()]
            )
        logging.info(f"WebRTC Bridge initialized with cameras: {self.default_cameras}")

    def _parse_cameras(self, cameras) -> List[str]:
        if isinstance(cameras, str):
            return [cam.strip() for cam in cameras.split(",") if cam.strip()]
        elif isinstance(cameras, list):
            result = []
            for camera in cameras:
                if isinstance(camera, str):
                    result.extend(
                        [cam.strip() for cam in camera.split(",") if cam.strip()]
                    )
                else:
                    result.append(str(camera).strip())
            return result
        else:
            return self.default_cameras

    def _build_url(self, cameras: List[str]) -> str:
        params = "&".join(f"src={quote(cam)}" for cam in cameras if cam)
        return f"{API_WEBRTC_URL}?{params}"

    async def handle_sdp(self, data: Dict[str, Any], msgUUID: str) -> Dict[str, Any]:
        try:
            sdp = data.get("sdp", "")
            if not sdp:
                return {"type": "error", "message": "Missing SDP in offer"}
            cameras = self._parse_cameras(data.get("cameras"))
            logging.info(f"Received SDP offer for cameras: {cameras}")
            if not cameras:
                return {"type": "error", "message": "No cameras specified"}

            url = self._build_url(cameras)
            http_client = AsyncHTTPClient()
            try:
                request = HTTPRequest(
                    url=url,
                    method="POST",
                    body=sdp,
                    headers={
                        "Content-Type": "application/sdp",
                        "Accept": "application/sdp",
                        "X-MQTT-User": msgUUID,
                    },
                    request_timeout=10,
                )
                logging.debug(f"Sending SDP offer to: {url}")
                response = await http_client.fetch(request)

                if response.code in (200, 201):
                    logging.info(f"Received SDP answer for cameras: {cameras}")
                    return {"type": "answer", "sdp": response.body.decode("utf-8")}
                else:
                    error_msg = response.body.decode("utf-8")
                    logging.error(f"go2rtc API error {response.code}: {error_msg}")
                    return {"type": "error", "message": error_msg}

            except HTTPError as e:
                logging.error(f"HTTP error: {e}")
                return {"type": "error", "message": str(e)}
            finally:
                http_client.close()

        except Exception as e:
            logging.error(f"SDP handling error: {e}")
            return {"type": "error", "message": str(e)}


def load_component(config: ConfigHelper) -> WebRTCBridge:
    return WebRTCBridge(config)
