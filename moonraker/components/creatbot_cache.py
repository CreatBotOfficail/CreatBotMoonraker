# Cache maintenance API for the CreatBot platform
#
# 打印机上有几处目录会随使用无限增长，占满后会直接导致上传或延时摄影失败：
#   upload_tmp        Moonraker 上传暂存目录（= 进程临时目录，见 [server] temp_dir）
#                     断电/断网导致上传中断时，暂存文件不会被上游清理，
#                     会一直占空间（/tmp 场景下甚至占的是内存）
#   timelapse_frames  延时摄影原始帧目录（timelapse 组件注册的 timelapse_frames）
#   gcode_thumbs      G-Code 缩略图缓存（<gcodes>/.thumbs，可自动重建）
#
# 对外提供查询与清理接口，并在 Klipper 就绪时清理过期的上传暂存文件。
#
# 接口：
#   GET  /server/cache/info
#        -> {"targets": {name: {path, exists, size, files}}, "total_size": N}
#   POST /server/cache/cleanup
#        body: {"targets": "upload_tmp,timelapse_frames"}   # 逗号分隔
#              {"all": true}                                # = upload_tmp + timelapse_frames
#              {"min_age": 300}                             # 只删早于 N 秒的文件
#        -> {"freed": {...}, "removed": {...}, "skipped": {...},
#            "errors": {...}, "total_freed": N}
#
# Copyright (C) 2026 CreatBot
# This file may be distributed under the terms of the GNU GPLv3 license.

from __future__ import annotations
import logging
import os
import pathlib
import tempfile
import time

from ..common import RequestType

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple
)

if TYPE_CHECKING:
    from ..confighelper import ConfigHelper
    from .file_manager.file_manager import FileManager

UPLOAD_TMP = "upload_tmp"
TIMELAPSE_FRAMES = "timelapse_frames"
GCODE_THUMBS = "gcode_thumbs"
ALL_TARGETS = (UPLOAD_TMP, TIMELAPSE_FRAMES, GCODE_THUMBS)
# all=true 时处理的目标；thumb 会让前端丢缩略图，必须显式指定
DEFAULT_TARGETS = (UPLOAD_TMP, TIMELAPSE_FRAMES)
# 上传暂存文件的最小保留时间：正在上传的文件 mtime 一直在更新，
# 只有断电/断网遗留的旧文件才会被清掉
DEFAULT_MIN_AGE = 300
THUMB_MIN_AGE = 60


class CreatBotCache:
    def __init__(self, config: ConfigHelper) -> None:
        self.server = config.get_server()
        self.min_upload_age: int = config.getint(
            "min_upload_age", DEFAULT_MIN_AGE, minval=0)
        self.clean_on_startup: bool = config.getboolean(
            "clean_uploads_on_startup", True)

        self.server.register_endpoint(
            "/server/cache/info", RequestType.GET,
            self._handle_cache_info
        )
        self.server.register_endpoint(
            "/server/cache/cleanup", RequestType.POST,
            self._handle_cache_cleanup
        )
        if self.clean_on_startup:
            self.server.register_event_handler(
                "server:klippy_ready", self._handle_startup)

    # ---------------------------------------------------------------- helpers
    def _get_file_manager(self) -> Optional[FileManager]:
        return self.server.lookup_component("file_manager", None)

    def _get_targets(self) -> Dict[str, Tuple[str, str]]:
        """Return {target: (directory, file pattern)} for the available areas."""
        targets: Dict[str, Tuple[str, str]] = {}
        # 上传暂存目录就是进程临时目录（由 [server] temp_dir 统一配置）
        targets[UPLOAD_TMP] = (
            tempfile.gettempdir(), "moonraker.upload-*.mru")
        fm = self._get_file_manager()
        if fm is None:
            return targets
        timelapse_dir = fm.get_directory("timelapse_frames")
        if timelapse_dir:
            targets[TIMELAPSE_FRAMES] = (timelapse_dir, "*")
        gcodes_dir = fm.get_directory("gcodes")
        if gcodes_dir:
            targets[GCODE_THUMBS] = (
                os.path.join(gcodes_dir, ".thumbs"), "*")
        return targets

    @staticmethod
    def _measure(path: str, pattern: str) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "path": path, "exists": False, "size": 0, "files": 0}
        try:
            target = pathlib.Path(path)
            if not target.is_dir():
                return info
            info["exists"] = True
            for item in target.glob(pattern):
                try:
                    if item.is_file():
                        info["size"] += item.stat().st_size
                        info["files"] += 1
                except OSError:
                    continue
        except OSError:
            pass
        return info

    @staticmethod
    def _clean(path: str, pattern: str,
               min_age: int) -> Tuple[int, int, List[str]]:
        """Delete matching files older than min_age.

        Returns (freed_bytes, removed_files, errors).
        """
        freed = 0
        removed = 0
        errors: List[str] = []
        now = time.time()
        try:
            target = pathlib.Path(path)
            if not target.is_dir():
                return 0, 0, [f"not a directory: {path}"]
            items = list(target.glob(pattern))
        except OSError as e:
            return 0, 0, [f"{path}: {e}"]
        for item in items:
            try:
                if not item.is_file():
                    continue
                stat = item.stat()
                if min_age > 0 and (now - stat.st_mtime) < min_age:
                    continue
                size = stat.st_size
                item.unlink()
                freed += size
                removed += 1
            except OSError as e:
                errors.append(f"{item}: {e}")
        return freed, removed, errors

    async def _is_printing(self) -> bool:
        """Be conservative: on any failure assume a print is running."""
        apis = self.server.lookup_component("klippy_apis", None)
        if apis is None:
            return False
        try:
            result: Dict[str, Any] = await apis.query_objects(
                {"print_stats": None})
            state = str(result.get("print_stats", {}).get(
                "state", "")).lower()
            return state in ("printing", "paused")
        except Exception:
            return True

    # -------------------------------------------------------------- endpoints
    async def _handle_cache_info(self, web_request) -> Dict[str, Any]:
        targets = self._get_targets()
        info: Dict[str, Any] = {}
        total = 0
        for name, (path, pattern) in targets.items():
            data = self._measure(path, pattern)
            info[name] = data
            total += data["size"]
        return {"targets": info, "total_size": total}

    async def _handle_cache_cleanup(self, web_request) -> Dict[str, Any]:
        all_flag = web_request.get_boolean("all", False)
        raw_targets = web_request.get_str("targets", "")
        min_age = web_request.get_int("min_age", DEFAULT_MIN_AGE, minval=0)
        if raw_targets.strip():
            wanted = [t.strip() for t in raw_targets.split(",") if t.strip()]
        elif all_flag:
            wanted = list(DEFAULT_TARGETS)
        else:
            raise self.server.error(
                "no cache target specified, use 'targets' or 'all'", 400)

        unknown = [t for t in wanted if t not in ALL_TARGETS]
        if unknown:
            raise self.server.error(
                f"unknown cache target(s): {', '.join(unknown)}", 400)

        targets = self._get_targets()
        printing = await self._is_printing()
        loop = self.server.get_event_loop()
        result: Dict[str, Any] = {
            "freed": {}, "removed": {}, "skipped": {}, "errors": {},
            "total_freed": 0
        }
        for name in wanted:
            if name not in targets:
                result["skipped"][name] = "target not available"
                continue
            if name == TIMELAPSE_FRAMES and printing:
                result["skipped"][name] = "printer is printing"
                continue
            path, pattern = targets[name]
            age = min_age
            if name == GCODE_THUMBS:
                age = max(min_age, THUMB_MIN_AGE)
            freed, removed, errors = await loop.run_in_thread(
                self._clean, path, pattern, age)
            result["freed"][name] = freed
            result["removed"][name] = removed
            if errors:
                result["errors"][name] = errors[:5]
            result["total_freed"] += freed
        logging.info(
            f"[creatbot_cache]: removed "
            f"{sum(result['removed'].values())} cache file(s), "
            f"freed {result['total_freed']} byte(s)")
        return result

    async def _handle_startup(self, *args) -> None:
        """Drop upload temp files left behind by an aborted upload."""
        targets = self._get_targets()
        if UPLOAD_TMP not in targets:
            return
        path, pattern = targets[UPLOAD_TMP]
        loop = self.server.get_event_loop()
        try:
            freed, removed, errors = await loop.run_in_thread(
                self._clean, path, pattern, self.min_upload_age)
            if removed:
                logging.info(
                    f"[creatbot_cache]: removed {removed} stale upload temp "
                    f"file(s) from {path}, freed {freed} byte(s)")
            for err in errors[:5]:
                logging.info(f"[creatbot_cache]: {err}")
        except Exception:
            logging.exception("[creatbot_cache]: startup cleanup failed")


def load_component(config: ConfigHelper) -> CreatBotCache:
    return CreatBotCache(config)
