# Support for Linux Input Devices (evdev)
#
# Copyright (C) 2025
#
# This file may be distributed under the terms of the GNU GPLv3 license.
from __future__ import annotations
import asyncio
import logging

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Union
)

try:
    import evdev
except ImportError:
    evdev = None

if TYPE_CHECKING:
    from ..confighelper import ConfigHelper
    from .application import InternalTransport as ITransport


class InputDeviceManager:
    def __init__(self, config: ConfigHelper) -> None:
        self.server = config.get_server()
        self.devices: Dict[str, LinuxInputDevice] = {}
        prefix_sections = config.get_prefix_sections("input_device")
        logging.info(f"Loading Input Devices: {prefix_sections}")

        if not prefix_sections:
            return

        if evdev is None:
            self.server.add_warning(
                "The 'evdev' package is required to use input devices. "
                "Please install it via pip: pip install evdev"
            )
            return

        for section in prefix_sections:
            cfg = config[section]
            try:
                dev = LinuxInputDevice(cfg)
            except Exception as e:
                msg = f"Failed to load input device [{cfg.get_name()}]\n{e}"
                self.server.add_warning(msg, exc_info=e)
                continue
            self.devices[dev.name] = dev
        self.server.register_notification("input_device:event")

    def component_init(self) -> None:
        for dev in self.devices.values():
            dev.initialize()


class LinuxInputDevice:
    def __init__(self, config: ConfigHelper) -> None:
        self.server = config.get_server()
        self.eventloop = self.server.get_event_loop()
        self.name = config.get_name().split()[-1]
        self.itransport: ITransport = self.server.lookup_component(
            "internal_transport")
        self.mutex = asyncio.Lock()

        self.device: Optional[evdev.InputDevice] = None
        device_path = config.get("device_path", None)
        device_name = config.get("device_name", None)
        if device_path is None and device_name is None:
            raise config.error(
                f"[{config.get_name()}]: Must specify either 'device_path' or 'device_name'"
            )

        self.device = self._find_device(device_path, device_name)
        if self.device is None:
            raise config.error(
                f"[{config.get_name()}]: Unable to find input device"
            )

        self.key_code: Union[int, str] = config.get("key_code", "KEY_POWER")
        if isinstance(self.key_code, str):
            if self.key_code.startswith("KEY_"):
                code = evdev.ecodes.ecodes.get(self.key_code)
                if code is None:
                    raise config.error(f"Unknown key code: {self.key_code}")
                self.key_code = code
            else:
                try:
                    self.key_code = int(self.key_code)
                except ValueError:
                    raise config.error(f"Invalid key code: {self.key_code}")

        self.grab = config.getboolean("grab", True)
        self.min_event_time = config.getfloat(
            "minimum_event_time", 0, minval=0.0)

        self.press_template = config.gettemplate(
            "on_press", None, is_async=True)
        self.release_template = config.gettemplate(
            "on_release", None, is_async=True)
        if (
            self.press_template is None and
            self.release_template is None
        ):
            raise config.error(
                f"[{config.get_name()}]: No template option configured"
            )

        self.notification_sent: bool = False
        self.user_data: Dict[str, Any] = {}
        self.context: Dict[str, Any] = {
            'call_method': self.itransport.call_method,
            'send_notification': self._send_notification,
            'event': {
                'elapsed_time': 0.,
                'received_time': 0.,
                'render_time': 0.,
                'pressed': False,
            },
            'user_data': self.user_data
        }
        self.read_task: Optional[asyncio.Task] = None
        self.last_press_time: float = 0.

    def _find_device(self, path: Optional[str], name: Optional[str]) -> Optional[evdev.InputDevice]:
        try:
            if path:
                return evdev.InputDevice(path)

            for dev_path in evdev.list_devices():
                dev = None
                try:
                    dev = evdev.InputDevice(dev_path)
                    if dev.name == name:
                        return dev
                except Exception:
                    pass

                if dev is not None:
                    dev.close()
        except Exception:
            logging.exception(f"Error finding input device: {path or name}")
        return None

    def initialize(self) -> None:
        if self.device:
            if self.grab:
                try:
                    self.device.grab()
                except Exception:
                    logging.exception(
                        f"Button {self.name}: Unable to grab input device")
            self.read_task = self.eventloop.create_task(self._read_loop())

    def get_status(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': "input_device",
            'event': self.context['event'],
        }

    def _send_notification(self, result: Any = None) -> None:
        if self.notification_sent:
            return
        self.notification_sent = True
        data = self.get_status()
        data['aux'] = result
        self.server.send_event("input_device:event", data)

    async def _read_loop(self) -> None:
        if self.device is None:
            return
        try:
            async for event in self.device.async_read_loop():
                if event.type == evdev.ecodes.EV_KEY and event.code == self.key_code:
                    await self._handle_event(event)
        except Exception:
            logging.exception(f"Button {self.name}: Error in read loop")

    async def _handle_event(self, event) -> None:
        if event.value == 2:
            return

        pressed = bool(event.value)
        if hasattr(event, "timestamp"):
            eventtime = event.timestamp()
        else:
            eventtime = event.sec + event.usec / 1000000.0

        if pressed:
            self.last_press_time = eventtime
            elapsed_time = 0.
        else:
            elapsed_time = eventtime - self.last_press_time

        if not pressed and elapsed_time < self.min_event_time:
            return

        template = self.press_template if pressed else self.release_template
        if template is None:
            return

        async with self.mutex:
            self.notification_sent = False
            event_info: Dict[str, Any] = {
                'elapsed_time': elapsed_time,
                'received_time': eventtime,
                'render_time': self.eventloop.get_loop_time(),
                'pressed': pressed
            }
            self.context['event'] = event_info
            try:
                await template.render_async(self.context)
            except Exception:
                action = "on_press" if pressed else "on_release"
                logging.exception(
                    f"Button {self.name}: '{action}' template error")


def load_component(config: ConfigHelper) -> InputDeviceManager:
    return InputDeviceManager(config)
