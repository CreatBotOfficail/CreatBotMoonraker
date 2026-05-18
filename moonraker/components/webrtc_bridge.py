from __future__ import annotations
import asyncio
import logging
import time
import tornado.websocket
from typing import TYPE_CHECKING, Dict, Any, List, Optional
from tornado.httpclient import AsyncHTTPClient, HTTPRequest, HTTPError
from urllib.parse import quote, urlencode
from ..utils import json_wrapper as jsonw

if TYPE_CHECKING:
    from ..confighelper import ConfigHelper
    from tornado.websocket import WebSocketClientConnection
    from .mqtt import MQTTClient

API_WEBRTC_URL = "http://localhost:1984/api/webrtc"
API_WS_URL = "ws://localhost:1984/api/ws"


class TrickleSession:
    def __init__(self,
                 ws: Optional[WebSocketClientConnection],
                 msg_uuid: str,
                 cameras: List[str]) -> None:
        self.ws = ws
        self.msg_uuid = msg_uuid
        self.cameras = cameras
        self.reader_task: Optional[asyncio.Task] = None
        self.created_time: float = time.time()
        self.last_activity: float = time.time()
        self.pending_candidates: List[str] = []


class WebRTCBridge:
    SESSION_TIMEOUT = 300  # 5 minutes
    CLEANUP_INTERVAL = 60  # 1 minute

    def __init__(self, config: ConfigHelper):
        self.server = config.get_server()
        self.mqtt: Optional[MQTTClient] = None
        default_cameras = config.getlist("camera_name", ["Camera"])
        self.default_cameras = []
        for camera in default_cameras:
            self.default_cameras.extend(
                [cam.strip() for cam in camera.split(",") if cam.strip()]
            )
        self.ws_api_url = config.get("ws_api_url", API_WS_URL)
        self.ws_timeout = config.getfloat("ws_timeout", 10.0, above=0.0)
        self.session_timeout = config.getfloat("session_timeout", self.SESSION_TIMEOUT, above=0.0)
        self.sessions: Dict[str, TrickleSession] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        logging.info(f"WebRTC Bridge initialized with cameras: {self.default_cameras}")

    def _parse_cameras(self, cameras) -> List[str]:
        if isinstance(cameras, str):
            result = [cam.strip() for cam in cameras.split(",") if cam.strip()]
            return result if result else self.default_cameras
        elif isinstance(cameras, list):
            result = []
            for camera in cameras:
                if isinstance(camera, str):
                    result.extend(
                        [cam.strip() for cam in camera.split(",") if cam.strip()]
                    )
                else:
                    result.append(str(camera).strip())
            return result if result else self.default_cameras
        else:
            return self.default_cameras

    def _build_url(self, cameras: List[str]) -> str:
        params = "&".join(f"src={quote(cam)}" for cam in cameras if cam)
        return f"{API_WEBRTC_URL}?{params}"

    def _build_ws_url(self, cameras: List[str]) -> str:
        params = [("src", cam) for cam in cameras if cam]
        query = urlencode(params)
        if "?" in self.ws_api_url:
            joiner = "" if self.ws_api_url.endswith(("?", "&")) else "&"
        else:
            joiner = "?"
        return f"{self.ws_api_url}{joiner}{query}" if query else self.ws_api_url

    def _get_topic_for_uuid(self, msg_uuid: str) -> str:
        req_topic = self.mqtt.api_request_topic
        if "+" in req_topic:
            return req_topic.replace("+", msg_uuid, 1)
        return req_topic

    def _parse_ice_candidate(self, pkt_value: Any) -> Dict[str, Any]:
        data: Dict[str, Any] = {"type": "answer_candidate", "sdp": ""}
        if isinstance(pkt_value, dict):
            data["sdp"] = pkt_value.get("candidate", "")
        else:
            data["sdp"] = pkt_value
        return data

    async def _cleanup_expired_sessions(self) -> None:
        while True:
            await asyncio.sleep(self.CLEANUP_INTERVAL)
            current_time = time.time()
            expired = [
                uuid for uuid, session in self.sessions.items()
                if current_time - session.last_activity > self.session_timeout
            ]
            for uuid in expired:
                await self._close_session(uuid)
                logging.info("Expired session closed: %s", uuid)

    async def component_init(self) -> None:
        """Initialize component and start cleanup task."""
        self.mqtt = self.server.lookup_component("mqtt")
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())

    async def close(self) -> None:
        """Stop cleanup task and close all sessions."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        # Close all active sessions
        for msg_uuid in list(self.sessions.keys()):
            await self._close_session(msg_uuid)

    async def _publish_sdp_to_app(self, msg_uuid: str, data: Dict[str, Any]) -> None:
        payload = {
            "ver": 3.1,
            "cmd": "SDP",
            "imei": self.mqtt.client_id,
            "uuid": msg_uuid,
            "data": data
        }
        topic = self._get_topic_for_uuid(msg_uuid)
        await self.mqtt.publish_topic(topic, payload, self.mqtt.api_qos)

    async def _close_session(self, msg_uuid: str) -> None:
        session = self.sessions.pop(msg_uuid, None)
        if session is None:
            return
        if session.reader_task is not None:
            current = asyncio.current_task()
            if session.reader_task is not current:
                session.reader_task.cancel()
                try:
                    await session.reader_task
                except asyncio.CancelledError:
                    pass
        if session.ws is not None:
            try:
                session.ws.close()
            except Exception:
                pass

    async def _ws_reader(self, session: TrickleSession) -> None:
        msg_uuid = session.msg_uuid
        try:
            while True:
                message = await session.ws.read_message()
                if message is None:
                    break
                session.last_activity = time.time()
                if not isinstance(message, str):
                    continue
                try:
                    packet = jsonw.loads(message)
                except jsonw.JSONDecodeError:
                    logging.debug(f"Invalid go2rtc WebSocket payload: {message}")
                    continue
                pkt_type = packet.get("type")
                pkt_value = packet.get("value", "")
                if pkt_type == "webrtc/candidate":
                    data = self._parse_ice_candidate(pkt_value)
                    await self._publish_sdp_to_app(msg_uuid, data)
                elif pkt_type == "webrtc":
                    logging.debug("Unexpected late webrtc answer from go2rtc")
                else:
                    logging.debug(f"Ignoring unsupported go2rtc ws message: {packet}")
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.exception("go2rtc WebSocket reader failed")
        finally:
            await self._close_session(msg_uuid)

    async def _handle_http_offer(self, data: Dict[str, Any], msgUUID: str) -> Dict[str, Any]:
        await self._close_session(msgUUID)

        sdp = data.get("sdp", "")
        if not sdp:
            return {"type": "error", "message": "Missing SDP in offer"}
        cameras = self._parse_cameras(data.get("cameras"))
        logging.info(f"Received SDP offer for cameras: {cameras}")

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
            error_msg = response.body.decode("utf-8")
            logging.error(f"go2rtc API error {response.code}: {error_msg}")
            return {"type": "error", "message": error_msg}
        except HTTPError as e:
            logging.error(f"HTTP error: {e}")
            return {"type": "error", "message": str(e)}
        finally:
            http_client.close()

    async def _handle_trickle_offer(self, data: Dict[str, Any], msgUUID: str) -> Dict[str, Any]:
        sdp = data.get("sdp", "")
        if not sdp:
            return {"type": "error", "message": "Missing SDP in offer_trickle"}
        cameras = self._parse_cameras(data.get("cameras"))
        ice_servers = data.get("ice_servers")

        await self._close_session(msgUUID)
        pending_session = TrickleSession(None, msgUUID, cameras)
        self.sessions[msgUUID] = pending_session

        ws_url = self._build_ws_url(cameras)
        ws: Optional[WebSocketClientConnection] = None
        try:
            logging.debug(f"Connecting go2rtc websocket: {ws_url}")
            ws = await tornado.websocket.websocket_connect(
                ws_url, connect_timeout=self.ws_timeout
            )
            offer_value: Dict[str, Any] = {
                "type": "offer",
                "sdp": sdp,
            }
            if ice_servers:
                offer_value["ice_servers"] = ice_servers
            offer_msg: Dict[str, Any] = {
                "type": "webrtc",
                "value": offer_value,
                "X-MQTT-User": msgUUID,
            }
            await ws.write_message(jsonw.dumps(offer_msg))

            for candidate in pending_session.pending_candidates:
                await ws.write_message(jsonw.dumps({
                    "type": "webrtc/candidate",
                    "value": candidate,
                }))
            pending_session.pending_candidates.clear()

            while True:
                message = await asyncio.wait_for(
                    ws.read_message(), timeout=self.ws_timeout
                )
                if message is None:
                    raise RuntimeError("go2rtc websocket closed before answer")
                if not isinstance(message, str):
                    continue
                packet = jsonw.loads(message)
                pkt_type = packet.get("type")
                pkt_value = packet.get("value", "")
                if pkt_type == "webrtc":
                    answer_sdp = pkt_value.get("sdp", "") if isinstance(pkt_value, dict) else pkt_value
                    pending_session.ws = ws
                    pending_session.reader_task = asyncio.create_task(self._ws_reader(pending_session))
                    return {"type": "answer_trickle", "sdp": answer_sdp}
                if pkt_type == "webrtc/candidate":
                    data_ice = self._parse_ice_candidate(pkt_value)
                    await self._publish_sdp_to_app(msgUUID, data_ice)
        except Exception as e:
            logging.error(f"WebSocket trickle mode failed: {e}")
            self.sessions.pop(msgUUID, None)
            if ws is not None:
                ws.close()
            return {"type": "error", "message": str(e)}

    async def _handle_trickle_ice(self, data: Dict[str, Any], msgUUID: str) -> Optional[Dict[str, Any]]:
        session = self.sessions.get(msgUUID)
        if session is None:
            return {"type": "error", "message": "No active trickle session"}
        session.last_activity = time.time()
        candidate = data.get("sdp", "")
        if session.ws is None:
            session.pending_candidates.append(candidate)
            return None
        payload = {"type": "webrtc/candidate", "value": candidate}
        try:
            await session.ws.write_message(jsonw.dumps(payload))
        except Exception as e:
            logging.error(f"Failed to forward ICE candidate to go2rtc: {e}")
            await self._close_session(msgUUID)
            return {"type": "error", "message": str(e)}
        return None

    async def handle_sdp(self, data: Dict[str, Any], msgUUID: str) -> Optional[Dict[str, Any]]:
        try:
            msg_type = data.get("type", "offer")
            if msg_type == "offer_trickle":
                return await self._handle_trickle_offer(data, msgUUID)
            if msg_type == "offer_candidate":
                return await self._handle_trickle_ice(data, msgUUID)
            if msg_type == "offer":
                return await self._handle_http_offer(data, msgUUID)
            return {"type": "error", "message": f"Unsupported SDP type: {msg_type}"}
        except Exception as e:
            logging.error(f"SDP handling error: {e}")
            return {"type": "error", "message": str(e)}


def load_component(config: ConfigHelper) -> WebRTCBridge:
    return WebRTCBridge(config)
