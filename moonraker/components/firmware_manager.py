import os
import logging
import asyncio
from .flash_tool import FlashTool
from ..confighelper import ConfigHelper
from ..common import KlippyState
from .klippy_apis import KlippyAPI

from typing import (
    TYPE_CHECKING,
    Any,
    Union,
    Dict,
)

if TYPE_CHECKING:
    from .machine import Machine

HOME = os.path.expanduser("~")
KLIPPER_DIR = os.path.join(HOME, "klipper/firmware")
MAIN_DEV = "/dev/ttyACM0"
SYSTEM_PYTHON = "python3"

class FirmwareUpdateNotifier:
    def __init__(self, server) -> None:
        self.server = server
        self.full_complete = False
        self.firmware = ""
    def _set_current_firmware(self, firmware: str) -> None:
        self.firmware = firmware
    def _set_full_complete(self, complete: bool) -> None:
        self.full_complete = complete
        if complete:
            self.notify_update_response(f"All firmware update finished!", is_complete=complete)
        elif not complete:
            self.notify_update_response(f"Firmware update failed!", is_complete=True)
    def notify_update_response(
            self, resp: Union[str, bytes], is_complete: bool = False
        ) -> None:
            resp = resp.strip()
            if isinstance(resp, bytes):
                resp = resp.decode()
            done = is_complete
            done &= self.full_complete
            notification = {
                'message': resp,
                'application': self.firmware,
                'complete': done}
            self.server.send_event(
                "update_manager:update_response", notification)

class FirmwareUpdate:
    def __init__(self, config: ConfigHelper) -> None:
        self.server = config.get_server()
        self.mcu_info: Dict[str, Dict[str, Any]] = {}
        self._service_info: Dict[str, Any] = {}
        self.min_version: str = ""
        self.klipper_version: str = ""
        self.current_progress = 0
        self.updating = False
        self.need_check_update = True
        self.notifier = None
        self.server.register_event_handler(
            "server:klippy_started", self._on_klippy_startup)

    @property
    def klippy_apis(self) -> KlippyAPI:
        return self.server.lookup_component("klippy_apis")

    def is_updating(self) -> bool:
        return self.updating

    async def _on_klippy_startup(self, state: KlippyState) -> None:
        if not self.need_check_update or not self._check_firmware_exists():
            return
        self.notifier = FirmwareUpdateNotifier(self.server)
        try:
            await self.build_mcu_info()
            table = []
            for device, info in self.mcu_info.items():
                table.append(
                    f"{device:<12} "
                    f"{info['canbus_uuid']:<14} "
                    f"{info.get('mcu_version', 'Unknown'):<9} "
                    f"{info['mcu_type']:<14} "
                    f"{'Yes' if info['need_update'] else 'No':<12}"
                )
            header = f"{'Device':<12} {'CANbus UUID':<14} {'Version':<9} {'MCU Type':<14} {'Update':<7}"
            separator = "-" * len(header)
            logging.info("MCU Information:\n" + "\n".join([header, separator] + table))
        except Exception as e:
            logging.exception(f"An error occurred during the building mcu info process: {e}")
        for mcu_data in self.mcu_info.values():
            if mcu_data.get('need_update', False):
                await self.start_update()
                break

    async def _do_klipper_action(self, action: str) -> None:
        try:
            machine: Machine = self.server.lookup_component("machine")
            await machine.do_service_action(action, "klipper")
            logging.info(f"Klipper service {action}.")
        except self.server.error:
            pass

    def _check_firmware_exists(self, firmware_path: str = None) -> bool:
        if not os.path.exists(KLIPPER_DIR):
            return False
        if firmware_path:
            return os.path.exists(firmware_path)
        return True

    async def start_update(self):
        try:
            self.updating = True
            uinfo: Dict[str, Any] = {}
            uinfo['busy'] = True
            uinfo['application'] = "firmware"
            self.server.send_event("update_manager:update_refreshed", uinfo)
            await self._do_klipper_action("stop")
            await self.upgrade_needed_tool_mcus()
            await self.upgrade_mcu()
            self.notifier._set_full_complete(True)
            await self._do_klipper_action("start")
            self.updating = False
        except Exception as e:
            self.notifier._set_full_complete(False)
            logging.exception(f"An error occurred during the update process: {e}")
        self.updating = False

    async def build_mcu_info(self) -> None:
        printer_info: Dict[str, Any] = {}
        cfg_status: Dict[str, Any] = {}
        try:
            printer_info = await self.klippy_apis.get_klippy_info()
            cfg_status = await self.klippy_apis.query_objects({'configfile': None})
        except self.server.error:
            logging.exception("PanelDue initialization request failed")
        config = cfg_status.get('configfile', {}).get('config', {})
        self.klipper_version = printer_info.get("software_version", "").split('-')[0]
        try:
            self._build_basic_mcu_info(config)
            await self._update_mcu_versions()
            self._check_mcu_update_needed()

        except Exception as e:
            logging.exception(f"An error occurred while building MCU info: {e}")

    def _build_basic_mcu_info(self, config: Dict[str, Any]) -> None:
        for mcu, value in config.items():
            if mcu.startswith("mcu") and "canbus_uuid" in value:
                self.mcu_info[mcu] = {"canbus_uuid": value["canbus_uuid"]}

    async def _update_mcu_versions(self) -> None:
        for mcu in self.mcu_info:
            try:
                response = await self.klippy_apis.query_objects({mcu: None})
                mcu_data = response.get(mcu, {})
                if mcu_data:
                    mcu_version: str = mcu_data.get('mcu_version', '')
                    mcu_constants: str = mcu_data.get('mcu_constants', '')
                    mcu_type: str = mcu_constants.get('MCU', '')
                    if mcu == "mcu":
                        self.min_version = mcu_data.get('min_firmware_version', "")
                    if mcu_version and mcu_type:
                        self.need_check_update = False
                        short_version: str = mcu_version.split('-')[0]
                        if short_version.lower().startswith('v'):
                            short_version = short_version[1:]
                        self.mcu_info[mcu]['mcu_version'] = short_version
                        self.mcu_info[mcu]['mcu_type'] = mcu_type
            except Exception as e:
                logging.error(f"Error querying {mcu}: {e}")

    def _compare_versions(self, version1, version2):
        if not version1 or not version2:
            return 0
        v1_parts = [int(part) for part in version1.split('.') if part]
        v2_parts = [int(part) for part in version2.split('.') if part]

        max_length = max(len(v1_parts), len(v2_parts))
        v1_parts.extend([0] * (max_length - len(v1_parts)))
        v2_parts.extend([0] * (max_length - len(v2_parts)))

        for i in range(max_length):
            if v1_parts[i] < v2_parts[i]:
                return -1
            elif v1_parts[i] > v2_parts[i]:
                return 1
        return 0

    def _check_mcu_update_needed(self) -> None:
        if self.klipper_version.lower().startswith('v'):
            self.klipper_version = self.klipper_version[1:]
        logging.info(f"min version: {self.min_version}")
        logging.info(f"klipper version: {self.klipper_version}")
        for mcu in self.mcu_info:
            mcu_version = self.mcu_info[mcu].get('mcu_version', "")
            if not mcu_version:
                self.mcu_info[mcu]['need_update'] = False
                continue

            mcu_vs_min = self._compare_versions(mcu_version, self.min_version)
            mcu_vs_klipper = self._compare_versions(mcu_version, self.klipper_version)

            if mcu_vs_min < 0 or mcu_vs_klipper > 0:
                self.mcu_info[mcu]['need_update'] = True
            else:
                self.mcu_info[mcu]['need_update'] = False

    def _get_firmware_path(self, mcu_name, mcu_type):
        STANDARD_MCU_FIRMWARE = {
            "mcu L_tool": "F072_L.bin",
            "mcu R_tool": "F072_R.bin",
            "mcu detect": "detect.bin",
            "mcu motion": "motion.bin",
            "mcu AC": "AC.bin",
        }
        if mcu_name in STANDARD_MCU_FIRMWARE:
            firmware_file = STANDARD_MCU_FIRMWARE[mcu_name]
            return os.path.join(KLIPPER_DIR, firmware_file)
        if mcu_name == "mcu tool":
            TOOL_MCU_FIRMWARE = {
                "stm32f072xb": "F072_L.bin",
                "stm32g431xx": "G431_L.bin"
            }
            if mcu_type in TOOL_MCU_FIRMWARE:
                firmware_file = TOOL_MCU_FIRMWARE[mcu_type]
                return os.path.join(KLIPPER_DIR, firmware_file)
            else:
                raise ValueError(f"Unsupported MCU type '{mcu_type}' for 'mcu tool'")

    async def upgrade_needed_tool_mcus(self):
        for mcu_name, mcu_data in self.mcu_info.items():
            if mcu_data.get('need_update', False) and mcu_name is not "mcu":
                firmware_path = self._get_firmware_path(mcu_name, mcu_data.get('mcu_type', ''))
                if firmware_path and self._check_firmware_exists(firmware_path):
                    self.notifier._set_current_firmware(mcu_name)
                    try:
                        flash_tool = FlashTool(
                            self.notifier,
                            uuid = mcu_data['canbus_uuid'],
                            firmware = firmware_path
                        )
                        exit_code = await flash_tool.run()
                        await asyncio.sleep(0.5)
                        if exit_code == 0:
                            logging.info(f"Upgrade operation for {mcu_name} succeeded!")
                        else:
                            logging.error(f"Upgrade operation for {mcu_name} failed, exit code: {exit_code}")
                    except Exception as e:
                        logging.exception(f"An exception occurred during the upgrade process for {mcu_name}: {e}")
                else:
                    logging.warning(f"No firmware specified for {mcu_name}, skipping upgrade.")

    async def upgrade_mcu(self):
        MCU_FIRMWARE_MAP = {
                "stm32f446xx": "F446.bin",
                "stm32g0b1xx": "G0B1.bin"
            }
        for mcu_name, mcu_data in self.mcu_info.items():
            if mcu_name == "mcu" and mcu_data.get('need_update', False):
                firmware_name = MCU_FIRMWARE_MAP[mcu_data.get('mcu_type', '')]
                firmware_path = os.path.join(KLIPPER_DIR, firmware_name)
                if firmware_name and self._check_firmware_exists(firmware_path):
                    self.notifier._set_current_firmware(mcu_name)
                    try:
                        flash_tool = FlashTool(
                            self.notifier,
                            uuid=mcu_data['canbus_uuid'],
                            request_bootloader=True
                        )
                        exit_code = await flash_tool.run()
                        if exit_code != 0:
                            logging.error(f"Failed to request bootloader for {mcu_name}, exit code: {exit_code}")
                            continue
                        await asyncio.sleep(0.5)
                        flash_tool = FlashTool(
                            self.notifier,
                            device=MAIN_DEV,
                            firmware=os.path.join(KLIPPER_DIR, firmware_file)
                        )
                        exit_code = await flash_tool.run()
                        if exit_code == 0:
                            logging.info(f"Upgrade operation for {mcu_name} succeeded, exit code: {exit_code}")
                        else:
                            logging.error(f"Upgrade operation for {mcu_name} failed, exit code: {exit_code}")
                    except Exception as e:
                        logging.exception(f"An exception occurred during the upgrade process for {mcu_name}: {e}")
                else:
                    logging.warning(f"No firmware specified for {mcu_name}, skipping upgrade.")

def load_component(config: ConfigHelper) -> FirmwareUpdate:
    return FirmwareUpdate(config)
