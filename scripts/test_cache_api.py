#!/usr/bin/env python3
"""Standalone self-test for the CreatBot temp-dir and cache maintenance code.

Runs without Moonraker's third-party dependencies: the cache component is
imported normally when that works, otherwise the relevant class is extracted
from the source tree, and Server._set_temp_dir is always extracted by AST.

Usage:
    python3 scripts/test_cache_api.py
"""
import ast
import asyncio
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import typing

REPO = pathlib.Path(__file__).resolve().parent.parent
FAILED = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:  # noqa: BLE001 - report everything
        FAILED.append(name)
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")


# --------------------------------------------------------------- loading
def _extract_method(path, cls_name, method_name):
    src = pathlib.Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == cls_name)
    method = next(n for n in cls.body
                  if isinstance(n, ast.FunctionDef) and n.name == method_name)
    mini = ast.ClassDef(name="Mini", bases=[], keywords=[], body=[method],
                        decorator_list=[])
    mod = ast.Module(body=[mini], type_ignores=[])
    ast.fix_missing_locations(mod)
    import logging
    ns = {"os": os, "pathlib": pathlib, "tempfile": tempfile,
          "logging": logging}
    exec(compile(mod, f"<{cls_name}.{method_name}>", "exec"), ns)
    return ns["Mini"]


def _extract_cache_module():
    path = REPO / "moonraker" / "components" / "creatbot_cache.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted_assigns = {
        "UPLOAD_TMP", "TIMELAPSE_FRAMES", "GCODE_THUMBS", "ALL_TARGETS",
        "DEFAULT_TARGETS", "DEFAULT_MIN_AGE", "THUMB_MIN_AGE"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "CreatBotCache":
            body.append(node)
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0],
                                                         ast.Name):
            if node.targets[0].id in wanted_assigns:
                body.append(node)
    import logging
    ns = dict(logging=logging, os=os, pathlib=pathlib, tempfile=tempfile,
              time=time, ConfigHelper=object, FileManager=object,
              TYPE_CHECKING=False, Any=typing.Any, Dict=typing.Dict,
              List=typing.List, Optional=typing.Optional, Tuple=typing.Tuple,
              RequestType=type("RT", (), {"GET": "GET", "POST": "POST"}))
    exec(compile(ast.Module(body=body, type_ignores=[]), "<cache>", "exec"),
         ns)
    return ns


# --------------------------------------------------------------- fakes
class ServerError(Exception):
    pass


class FakeLoop:
    def run_in_thread(self, fn, *args):
        async def wrapper():
            return fn(*args)
        return wrapper()


class FakeFileManager:
    def __init__(self, frames, gcodes):
        self._dirs = {"timelapse_frames": str(frames),
                      "gcodes": str(gcodes)}

    def get_directory(self, root):
        return self._dirs.get(root, "")


class FakeServer:
    def __init__(self, fm):
        self.fm = fm
        self.endpoints = []
        self.events = []
        self.warnings = []

    def register_endpoint(self, path, methods, callback):
        self.endpoints.append((path, methods, callback.__name__))

    def register_event_handler(self, name, callback):
        self.events.append((name, callback.__name__))

    def lookup_component(self, name, default=None):
        return self.fm if name == "file_manager" else default

    def error(self, message, status_code=400):
        return ServerError(f"{status_code}: {message}")

    def get_event_loop(self):
        return FakeLoop()


class FakeConfig:
    def __init__(self, server, options):
        self.server = server
        self.options = options

    def get_server(self):
        return self.server

    def getint(self, option, default=None, **kwargs):
        return int(self.options.get(option, default))

    def getboolean(self, option, default=None, **kwargs):
        return bool(self.options.get(option, default))


class FakeRequest:
    """Mirrors the real WebRequest signatures: (key, default)."""

    def __init__(self, data):
        self.data = data

    def get_boolean(self, key, default=None):
        return bool(self.data.get(key, default))

    def get_str(self, key, default=None):
        return str(self.data.get(key, default))

    def get_int(self, key, default=None):
        return int(self.data.get(key, default))


# --------------------------------------------------------------- tests
def main():
    print(f"repo: {REPO}")
    cache_ns = _extract_cache_module()
    Cache = cache_ns["CreatBotCache"]
    SetTemp = _extract_method(REPO / "moonraker" / "server.py", "Server",
                              "_set_temp_dir")

    class RecordingTemp(SetTemp):
        def __init__(self):
            self.warnings = []

        def add_warning(self, warning, warn_id=None, log=True,
                        exc_info=None):
            self.warnings.append((warn_id, warning))
            return str(warn_id)

    # ---- _set_temp_dir -------------------------------------------------
    original = tempfile.gettempdir()
    original_env = os.environ.get("TMPDIR")

    def empty_value_warns():
        inst = RecordingTemp()
        inst._set_temp_dir("")
        assert len(inst.warnings) == 1, inst.warnings
        assert inst.warnings[0][0] == "temp_dir", inst.warnings
        assert "upload" in inst.warnings[0][1]
        assert tempfile.gettempdir() == original

    def bad_path_warns():
        blocker = pathlib.Path(tempfile.mkdtemp()) / "file"
        blocker.write_text("x", encoding="utf-8")
        inst = RecordingTemp()
        inst._set_temp_dir(str(blocker / "sub"))
        assert len(inst.warnings) == 1, inst.warnings
        assert tempfile.gettempdir() == original

    def good_path_applies():
        target = pathlib.Path(tempfile.mkdtemp(prefix="tmpdir-test-"))
        inst = RecordingTemp()
        inst._set_temp_dir(str(target))
        assert not inst.warnings, inst.warnings
        assert pathlib.Path(tempfile.gettempdir()) == target.resolve()
        assert os.environ.get("TMPDIR") == str(target.resolve())
        fd, path = tempfile.mkstemp()
        os.close(fd)
        assert pathlib.Path(path).parent == target.resolve()
        out = subprocess.run(
            [sys.executable, "-c",
             "import tempfile; print(tempfile.gettempdir())"],
            capture_output=True, text=True)
        assert out.stdout.strip() == str(target.resolve()), out

    check("_set_temp_dir: empty value -> add_warning, dir unchanged",
          empty_value_warns)
    check("_set_temp_dir: bad path -> add_warning, dir unchanged",
          bad_path_warns)
    check("_set_temp_dir: good path applies to process and children",
          good_path_applies)

    # ---- cache component -----------------------------------------------
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="cache-test-"))
    frames = scratch / "timelapse_frames"
    thumbs = scratch / "gcodes" / ".thumbs"
    frames.mkdir()
    thumbs.mkdir(parents=True)
    server = FakeServer(FakeFileManager(frames, scratch / "gcodes"))
    component = Cache(FakeConfig(server, {"min_upload_age": 300,
                                          "clean_uploads_on_startup": True}))

    def endpoints_registered():
        paths = {(p, m) for p, m, _ in server.endpoints}
        assert ("/server/cache/info", "GET") in paths, paths
        assert ("/server/cache/cleanup", "POST") in paths, paths
        assert ("server:klippy_ready", "_handle_startup") in server.events

    def targets_are_correct():
        targets = component._get_targets()
        assert targets["upload_tmp"][0] == tempfile.gettempdir()
        assert targets["upload_tmp"][1] == "moonraker.upload-*.mru"
        assert targets["timelapse_frames"] == (str(frames), "frame*.jpg")
        assert targets["gcode_thumbs"] == (str(scratch / "gcodes"),
                                          "**/.thumbs/*")

    def cleanup_respects_age_and_pattern():
        keep = frames / "frame000001.jpg"
        keep.write_bytes(b"a" * 10)
        old = frames / "frame000002.jpg"
        old.write_bytes(b"b" * 20)
        os.utime(old, (time.time() - 900, time.time() - 900))
        mp4 = frames / "render.mp4"
        mp4.write_bytes(b"c" * 30)
        os.utime(mp4, (time.time() - 900, time.time() - 900))
        freed, removed, errors = Cache._clean(str(frames), "frame*.jpg", 300)
        assert (freed, removed) == (20, 1), (freed, removed, errors)
        assert keep.exists() and not old.exists() and mp4.exists()

    def thumbs_recursive():
        nested = scratch / "gcodes" / "sub1" / ".thumbs"
        nested.mkdir(parents=True)
        root_thumb = thumbs / "a-32x32.png"
        nested_thumb = nested / "b-32x32.png"
        keep = scratch / "gcodes" / "keep.gcode"
        for path in (root_thumb, nested_thumb, keep):
            path.write_bytes(b"x" * 10)
            os.utime(path, (time.time() - 900, time.time() - 900))
        freed, removed, errors = Cache._clean(
            str(scratch / "gcodes"), "**/.thumbs/*", 300)
        assert (freed, removed) == (20, 2), (freed, removed, errors)
        assert not root_thumb.exists() and not nested_thumb.exists()
        assert keep.exists()

    def cleanup_endpoint_behaviour():
        results = asyncio.run(
            component._handle_cache_cleanup(FakeRequest({"all": True})))
        assert set(results["removed"]) == {"upload_tmp", "timelapse_frames"}
        for payload in ({}, {"targets": "nope"}):
            try:
                asyncio.run(component._handle_cache_cleanup(
                    FakeRequest(payload)))
            except ServerError:
                pass
            else:
                raise AssertionError(f"expected error for {payload}")
        async def printing():
            return True
        component._is_printing = printing
        results = asyncio.run(
            component._handle_cache_cleanup(FakeRequest({"all": True})))
        assert results["skipped"].get("timelapse_frames") == \
            "printer is printing", results

    def info_endpoint_works():
        data = asyncio.run(component._handle_cache_info(FakeRequest({})))
        assert data["targets"]["upload_tmp"]["path"] == tempfile.gettempdir()
        assert set(data["targets"]) == {"upload_tmp", "timelapse_frames",
                                        "gcode_thumbs"}

    check("cache component: endpoints and event registered",
          endpoints_registered)
    check("cache component: target paths and patterns", targets_are_correct)
    check("cache component: cleanup honours min_age and frame pattern",
          cleanup_respects_age_and_pattern)
    check("cache component: thumbnails cleaned recursively (nested dirs)",
          thumbs_recursive)
    check("cache component: cleanup endpoints, errors and print guard",
          cleanup_endpoint_behaviour)
    check("cache component: info endpoint", info_endpoint_works)

    # restore process state
    tempfile.tempdir = original
    if original_env is None:
        os.environ.pop("TMPDIR", None)
    else:
        os.environ["TMPDIR"] = original_env

    print()
    if FAILED:
        print(f"FAILED: {len(FAILED)} check(s): {', '.join(FAILED)}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
