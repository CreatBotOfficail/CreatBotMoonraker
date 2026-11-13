# Direct ctypes binding to the vendor RKNN runtime (librknnrt.so).
#
# Copyright (C) 2025 Creatbot
#
# This file may be distributed under the terms of the GNU GPLv3 license.
#
# Replaces rknnlite for the nozzle model. Two things keep it fast:
#   - outputs are preallocated float32 buffers handed to
#     rknn_outputs_get(is_prealloc=1), so the runtime writes straight into
#     them instead of allocating and converting per call;
#   - the input array is passed by pointer, no intermediary copy.
# The final numpy copy is kept small and optional (see copy_outputs).

from __future__ import annotations

import ctypes
import logging
import threading
from ctypes import POINTER, c_char_p, c_float, c_int, c_uint32, c_uint64, c_void_p
from typing import List, Optional

import numpy as np

_LIB_CANDIDATES = ("librknnrt.so", "/usr/lib/librknnrt.so", "/lib/librknnrt.so")

# rknn_query_cmd
QUERY_IN_OUT_NUM = 0
QUERY_OUTPUT_ATTR = 2
QUERY_SDK_VERSION = 5

# rknn_tensor_type / rknn_tensor_format
TENSOR_UINT8 = 3
FORMAT_NHWC = 1

_lib = None
_lib_error: Optional[str] = None
_lib_lock = threading.Lock()


class RknnRuntimeError(RuntimeError):
    pass


class _TensorAttr(ctypes.Structure):
    _fields_ = [
        ("index", c_uint32),
        ("n_dims", c_uint32),
        ("dims", c_uint32 * 16),
        ("name", ctypes.c_char * 256),
        ("n_elems", c_uint32),
        ("size", c_uint32),
        ("fmt", c_int),
        ("type", c_int),
        ("qnt_type", c_int),
        ("fl", ctypes.c_int8),
        ("zp", ctypes.c_int32),
        ("scale", c_float),
        ("w_stride", c_uint32),
        ("size_with_stride", c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", c_uint32),
    ]


class _Input(ctypes.Structure):
    _fields_ = [
        ("index", c_uint32),
        ("buf", c_void_p),
        ("size", c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", c_int),
        ("fmt", c_int),
    ]


class _Output(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index", c_uint32),
        ("buf", c_void_p),
        ("size", c_uint32),
    ]


class _IOParam(ctypes.Structure):
    _fields_ = [("n_input", c_uint32), ("n_output", c_uint32)]


def _load() -> Optional[ctypes.CDLL]:
    global _lib, _lib_error
    with _lib_lock:
        if _lib is not None:
            return _lib
        for path in _LIB_CANDIDATES:
            try:
                lib = ctypes.CDLL(path)
            except OSError:
                continue
            try:
                lib.rknn_init.restype = c_int
                lib.rknn_init.argtypes = [POINTER(c_uint64), c_void_p,
                                          c_uint32, c_uint32, c_void_p]
                lib.rknn_query.restype = c_int
                lib.rknn_query.argtypes = [c_uint64, c_int, c_void_p, c_uint32]
                lib.rknn_inputs_set.restype = c_int
                lib.rknn_inputs_set.argtypes = [c_uint64, c_uint32, POINTER(_Input)]
                lib.rknn_run.restype = c_int
                lib.rknn_run.argtypes = [c_uint64, c_void_p]
                lib.rknn_outputs_get.restype = c_int
                lib.rknn_outputs_get.argtypes = [c_uint64, c_uint32,
                                                 POINTER(_Output), c_void_p]
                lib.rknn_destroy.restype = c_int
                lib.rknn_destroy.argtypes = [c_uint64]
            except AttributeError as exc:
                _lib_error = f"ABI mismatch in {path}: {exc}"
                logging.warning(_lib_error)
                continue
            _lib = lib
            logging.info("librknnrt loaded from %s", path)
            return _lib
        if _lib_error is None:
            _lib_error = "librknnrt.so not found"
        return None


def is_available() -> bool:
    return _load() is not None


def last_load_error() -> Optional[str]:
    _load()
    return _lib_error


class RknnRuntime:
    """One model = one context. Mirrors the subset of the rknnlite API we use."""

    def __init__(self, model_path: str, core_mask: int = 0,
                 copy_outputs: bool = True) -> None:
        lib = _load()
        if lib is None:
            raise RknnRuntimeError(f"librknnrt unavailable: {_lib_error}")
        self._lib = lib
        self.model_path = model_path
        self.copy_outputs = copy_outputs
        self._lock = threading.Lock()

        with open(model_path, "rb") as f:
            model = f.read()
        self._model_buf = ctypes.create_string_buffer(model, len(model))

        self._ctx = c_uint64(0)
        ret = lib.rknn_init(ctypes.byref(self._ctx),
                            ctypes.cast(self._model_buf, c_void_p),
                            len(model), 0, None)
        if ret != 0:
            raise RknnRuntimeError(f"rknn_init failed: {ret}")

        self.in_w = self.in_h = self.in_c = 0
        self._out_dims: List[tuple] = []
        self._out_bufs: List[np.ndarray] = []
        self._out_structs = None
        self.n_out = 0
        self._query_io()
        self._prepare_outputs()

    # ------------------------------------------------------------------
    def _query_io(self) -> None:
        io = _IOParam()
        if self._lib.rknn_query(self._ctx, QUERY_IN_OUT_NUM, ctypes.byref(io),
                                ctypes.sizeof(io)) != 0:
            raise RknnRuntimeError("rknn_query(IN_OUT_NUM) failed")
        if io.n_input != 1:
            raise RknnRuntimeError(f"expected 1 input, got {io.n_input}")
        self.n_out = io.n_output

        # input attr: the model is exported as NHWC
        attr = _TensorAttr()
        attr.index = 0
        if self._lib.rknn_query(self._ctx, 1, ctypes.byref(attr),
                                ctypes.sizeof(attr)) != 0:
            raise RknnRuntimeError("rknn_query(INPUT_ATTR) failed")
        self.in_h, self.in_w, self.in_c = attr.dims[1], attr.dims[2], attr.dims[3]

    def _prepare_outputs(self) -> None:
        self._out_structs = (_Output * self.n_out)()
        for i in range(self.n_out):
            attr = _TensorAttr()
            attr.index = i
            if self._lib.rknn_query(self._ctx, QUERY_OUTPUT_ATTR, ctypes.byref(attr),
                                    ctypes.sizeof(attr)) != 0:
                raise RknnRuntimeError(f"rknn_query(OUTPUT_ATTR[{i}]) failed")
            dims = tuple(int(attr.dims[d]) for d in range(attr.n_dims))
            elems = int(attr.n_elems)
            buf = np.empty(elems, dtype=np.float32)
            self._out_dims.append(dims)
            self._out_bufs.append(buf)
            self._out_structs[i].index = i
            self._out_structs[i].want_float = 1
            self._out_structs[i].is_prealloc = 1
            self._out_structs[i].buf = buf.ctypes.data_as(c_void_p)
            self._out_structs[i].size = elems * 4

    # ------------------------------------------------------------------
    def inference(self, inputs: List[np.ndarray]) -> Optional[List[np.ndarray]]:
        if not inputs:
            return None
        arr = inputs[0]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        inp = _Input()
        inp.index = 0
        inp.buf = arr.ctypes.data_as(c_void_p)
        inp.size = int(arr.size)
        inp.pass_through = 0
        inp.type = TENSOR_UINT8
        inp.fmt = FORMAT_NHWC

        with self._lock:
            ret = self._lib.rknn_inputs_set(self._ctx, 1, ctypes.byref(inp))
            if ret != 0:
                raise RknnRuntimeError(f"rknn_inputs_set failed: {ret}")
            ret = self._lib.rknn_run(self._ctx, None)
            if ret != 0:
                raise RknnRuntimeError(f"rknn_run failed: {ret}")
            ret = self._lib.rknn_outputs_get(self._ctx, self.n_out,
                                             self._out_structs, None)
            if ret != 0:
                raise RknnRuntimeError(f"rknn_outputs_get failed: {ret}")
            if self.copy_outputs:
                return [b.reshape(d).copy() for b, d in zip(self._out_bufs, self._out_dims)]
            return [b.reshape(d) for b, d in zip(self._out_bufs, self._out_dims)]

    # ------------------------------------------------------------------
    def get_sdk_version(self) -> str:
        buf = ctypes.create_string_buffer(512)
        ret = self._lib.rknn_query(self._ctx, QUERY_SDK_VERSION, buf, len(buf))
        if ret != 0:
            return "unknown"
        return buf.value.decode(errors="replace").split("\x00")[0] or "unknown"

    def release(self) -> None:
        if getattr(self, "_ctx", None) is not None:
            self._lib.rknn_destroy(self._ctx)
            self._ctx = None
            self._out_bufs = []

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass
