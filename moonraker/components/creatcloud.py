# Support for option secrets from database
#
# Copyright (C) 2025 CreatBot LYN <yinyueguodong@foxmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.

from __future__ import annotations
import logging
import paho.mqtt.client as paho_mqtt

from ..common import JsonRPC, RequestType, TransportType, WebRequest
from ..utils import json_wrapper as jsonw

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional
)

if TYPE_CHECKING:
    from ..confighelper import ConfigHelper
    from .mqtt import MQTTClient
    from .machine import Machine
    from .safe_options import SafeOptions
    from .authorization import Authorization
    from .webrtc_bridge import WebRTCBridge
    AuthComp = Optional[Authorization]


class CreatCloud:
    def __init__(self, config: ConfigHelper) -> None:
        self.server = config.get_server()
        self.mqtt: MQTTClient = self.server.load_component(config, 'mqtt')
        machine: Machine = self.server.lookup_component("machine")

        self.creatcloud_enabled: Optional[bool] = config.getboolean("enable", None)
        self.err_code = 0
        self.err_str = ""

        # Reinitialize the client
        if self.mqtt.client_id is None:
            client_id = self.mqtt.client_id = machine.get_machine_uuid()
            self.mqtt.client._client_id = client_id
        else:
            client_id = self.mqtt.client_id
            self.server.add_warning(
                "CreatCloud client_id conflict risk (using specified ID)")

        creatcloud_options = self._get_creatcloud_options()
        if creatcloud_options is None:
            self.mqtt.user_name = None
            self.mqtt.password = None
        else:
            self.mqtt.user_name = creatcloud_options["username"]
            self.mqtt.password = creatcloud_options["password"]

        self.creatcloud_topic_prefix = "CreatCloud/Klipper"
        self.mqtt.api_request_topic = f"{self.creatcloud_topic_prefix}/{client_id}/+/Action"
        self.mqtt.api_resp_topic = f"{self.creatcloud_topic_prefix}/{client_id}/000000/Action"
        self.mqtt.klipper_status_topic = f"{self.creatcloud_topic_prefix}/{client_id}/Status"
        self.mqtt.klipper_state_prefix = f"{self.creatcloud_topic_prefix}/{client_id}/State"
        self.mqtt.moonraker_status_topic = f"{self.creatcloud_topic_prefix}/{client_id}/Public"

        self.mqtt.subscribed_topics.clear()
        self.mqtt.subscribe_topic(self.mqtt.api_request_topic,
                                  self._process_creatcloud_request,
                                  self.mqtt.api_qos)

        # Rewrite reconnect
        self.old_do_reconnect = self.mqtt._do_reconnect
        self.mqtt._do_reconnect = self._creatcloud_reconnect

        # Register CreatCloud Interface
        ep_transports = TransportType.all() & ~TransportType.MQTT
        self.server.register_endpoint(
            "/server/creatcloud/enable", RequestType.POST, self._handle_creatcloud_enable,
            transports=ep_transports, auth_required=False
        )
        self.server.register_endpoint(
            "/server/creatcloud/user", RequestType.POST, self._handle_creatcloud_user,
            transports=ep_transports, auth_required=False
        )
        self.server.register_endpoint(
            "/server/creatcloud/info", RequestType.GET, self._handle_creatcloud_info,
            transports=ep_transports
        )

        # Register CreatCloud notification and event
        self.server.register_notification(
            "creatcloud:info_update", "creatcloud_info_update")
        self.server.register_event_handler(
            "machine:public_ip_changed", self._update_local_ip)
        self.server.register_event_handler(
            "mqtt:connected", self._update_connect_state)
        self.server.register_event_handler(
            "mqtt:disconnected", self._update_connect_state)
        self.server.register_event_handler(
            "mqtt:connect_error", self._update_connect_state)

    async def component_init(self) -> None:
        pass

    async def _creatcloud_reconnect(self, first: bool = False) -> None:
        if self._check_creatcloud_registerd():
            if self._check_creatcloud_enabled():
                logging.info("CreatCloud Enabled")
                await self.old_do_reconnect(first)
            else:
                logging.info("CreatCloud Disabled")
        else:
            logging.info("CreatCloud Unregistered")

    async def _process_creatcloud_request(self, payload: bytes, topic: str = None) -> None:
        try:
            request: Dict[str, Any] = jsonw.loads(payload)
            msgVer = request.get("ver")
            response = request.copy()
            if msgVer == 3:  # msg version is 3 or 3.0
                msgIMEI = request.get("imei")
                msgUUID = request.get("uuid")
                msgCmd = request.get("cmd")
                msgData = request.get("data")
                response["data"] = ""

                if msgIMEI == self.mqtt.client_id:
                    auth: AuthComp = self.server.lookup_component('authorization', None)
                    if auth is None or auth.check_mqtt(msgUUID) or msgCmd == 'PWD':
                        if msgCmd == 'PWD':
                            if auth is not None:
                                response['data'] = 'OK' if auth.validate_mqtt(
                                    msgUUID, msgData) else 'INCORRECT'
                            else:
                                response['data'] = 'IGNORE'
                        elif msgCmd == 'API':
                            rpc: JsonRPC = self.server.lookup_component("jsonrpc")
                            result = await rpc.dispatch(jsonw.dumps(msgData), self.mqtt)
                            response["data"] = jsonw.loads(result)
                        elif msgCmd == 'SDP':
                            webrtc_bridge: WebRTCBridge = self.server.lookup_component(
                                "webrtc_bridge", None)
                            if webrtc_bridge:
                                response["data"] = await webrtc_bridge.handle_sdp(msgData, topic)
                            else:
                                response["data"] = {
                                    "type": "error", "message": "WebRTC Bridge component not available"}
                        else:
                            response["data"] = f"error: Unknown MQTT message cmd: {msgCmd}"
                    else:
                        response['data'] = f"error: MQTT UserID [{msgUUID}] needs authentication"
                else:
                    response["data"] = f"error: MQTT client_id [{msgIMEI}] does not match"
            else:
                response["data"] = f"error: MQTT message version [{msgVer}] is not supported"
        except jsonw.JSONDecodeError:
            data = payload.decode()
            response = f"MQTT payload is not valid json: {data}"
            logging.exception(response)
        except Exception as e:
            response = None
            logging.exception(e)

        if response is not None and topic is not None:
            await self.mqtt.publish_topic(topic, response, self.mqtt.api_qos)

    def _get_creatcloud_options(self) -> Optional[Dict[str, Any]]:
        safeOptions: SafeOptions = self.server.lookup_component("safe_options")
        return safeOptions.get("CreatCloud")

    async def _update_creatcloud_options(self, key: str, value: Any) -> None:
        safeOptions: SafeOptions = self.server.lookup_component("safe_options")
        if not safeOptions.has_section("CreatCloud"):
            # the database needs to be initialized
            await safeOptions.init_section("CreatCloud", {
                "actived": (False, False),
                "username": (True, ""),
                "password": (True, ""),
            })
        await safeOptions.update_option("CreatCloud", key, value)

    def _check_creatcloud_enabled(self) -> bool:
        if self.creatcloud_enabled is not None:
            return self.creatcloud_enabled

        creatcloud_options = self._get_creatcloud_options()
        return False if creatcloud_options is None else creatcloud_options["actived"]

    def _check_creatcloud_registerd(self) -> bool:
        creatcloud_options = self._get_creatcloud_options()
        return creatcloud_options["username"] is not None \
            and creatcloud_options["password"] is not None

    async def _handle_creatcloud_enable(self, web_request: WebRequest) -> Dict[str, Any]:
        active = web_request.get_boolean("active")
        curActive = self._check_creatcloud_enabled()

        # update databse safeOptions
        creatcloud_options = self._get_creatcloud_options()
        if creatcloud_options is None or creatcloud_options["actived"] is not active:
            await self._update_creatcloud_options("actived", active)

        # set the config value temporarily
        if self.creatcloud_enabled is not None:
            self.creatcloud_enabled = active

        if active is not curActive:
            # switch the MQTT connection state.
            logging.info(f"CreatCloud Change to {'Enable' if active else 'Disable'}")
            if active and not self.mqtt.is_connected() and self.mqtt.connect_task is None:
                self.mqtt.connect_task = self.mqtt.eventloop.create_task(
                    self.mqtt._do_reconnect(first=True)
                )
            else:
                await self.mqtt.close()

            # send notification
            self.server.get_event_loop().delay_callback(
                0.01, self.server.send_event,
                "creatcloud:info_update",
                {"actived": active}
            )

        return {"result": "success", "actived": active}

    async def _handle_creatcloud_user(self, web_request: WebRequest) -> Dict[str, Any]:
        username = web_request.get_str("username")
        password = web_request.get_str("password")

        # update databse safeOptions
        await self._update_creatcloud_options("username", username)
        await self._update_creatcloud_options("password", password)
        creatcloud_options = self._get_creatcloud_options()
        new_username = creatcloud_options["username"]
        new_password = creatcloud_options["password"]

        if self.mqtt.user_name != new_username \
                or self.mqtt.password != new_password:
            # update username and password
            self.mqtt.user_name = new_username
            self.mqtt.password = new_password

            # MQTT reconnect with new credential
            logging.info("CreatCloud Restart [Username/Password Change]")
            await self.mqtt.close()
            self.mqtt.client.username_pw_set(self.mqtt.user_name, self.mqtt.password)
            self.connect_task = self.mqtt.eventloop.create_task(
                self.mqtt._do_reconnect(first=True)
            )

            # send notification
            self.server.get_event_loop().delay_callback(
                0.01, self.server.send_event,
                "creatcloud:info_update",
                {"registerd": self._check_creatcloud_registerd()}
            )

        return {"result": "success", "actived": self._check_creatcloud_enabled()}

    async def _handle_creatcloud_info(self, web_request: WebRequest) -> Dict[str, Any]:
        machine: Machine = self.server.lookup_component("machine")
        creatcloud_options = self._get_creatcloud_options()
        if creatcloud_options is None:
            await self._update_creatcloud_options("actived", False)
            creatcloud_options = self._get_creatcloud_options()

        return {
            "local_ip": machine.public_ip or "unknown",
            "hostname": self.server.get_host_info()["hostname"],
            "machine_id": self.mqtt.client_id,
            "pubkey": creatcloud_options["__pubkey__"],
            "actived": creatcloud_options["actived"],
            "registerd": self._check_creatcloud_registerd(),
            "online": self.mqtt.is_connected(),
            "conn_code": self.err_code,
            "conn_err": self.err_str
        }

    def _update_local_ip(self, ip: str) -> None:
        self.server.send_event("creatcloud:info_update", {
            "local_ip": ip
        })

    def _update_connect_state(self, code: int = 0, info: str = "") -> None:
        self.err_code, self.err_str = code, info
        self.server.send_event("creatcloud:info_update", {
            "online": self.mqtt.is_connected(),
            "conn_code": self.err_code,
            "conn_err": self.err_str
        })


def load_component(config: ConfigHelper) -> CreatCloud:
    return CreatCloud(config)
