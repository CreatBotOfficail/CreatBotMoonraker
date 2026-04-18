# NPU Nozzle Detection Module for Moonraker
#
# Copyright (C) 2025 Creatbot
#
# This file may be distributed under the terms of the GNU GPLv3 license.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    from ...confighelper import ConfigHelper

MODEL_INPUT_SIZE: Tuple[int, int] = (640, 640)
OBJ_THRESH: float = 0.25
NMS_THRESH: float = 0.45
REG_MAX: int = 16

_strides = np.array([8, 16, 32], dtype=np.float32)
_grids: List[Tuple[np.ndarray, np.ndarray]] = []
for _s in _strides:
    _gh = MODEL_INPUT_SIZE[1] // int(_s)
    _gw = MODEL_INPUT_SIZE[0] // int(_s)
    _cols = np.tile(np.arange(_gw, dtype=np.float32), _gh)
    _rows = np.repeat(np.arange(_gh, dtype=np.float32), _gw)
    _grids.append((_cols, _rows))

_dfl_acc = np.arange(REG_MAX, dtype=np.float32).reshape(1, 1, REG_MAX)


def preprocess(
    img: np.ndarray,
    input_size: Tuple[int, int] = MODEL_INPUT_SIZE,
) -> Tuple[np.ndarray, float, float]:
    src_h, src_w = img.shape[:2]
    target_w, target_h = input_size
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (target_w, target_h))
    return resized[np.newaxis].astype(np.uint8), target_w / src_w, target_h / src_h


def _dfl_batch(box_candidates: np.ndarray) -> np.ndarray:
    y = box_candidates.reshape(-1, 4, REG_MAX)
    y_max = y.max(axis=2, keepdims=True)
    e = np.exp(y - y_max)
    y = e / e.sum(axis=2, keepdims=True)
    return (y * _dfl_acc).sum(axis=2)


def postprocess(
    outputs: List[np.ndarray],
    orig_shape: Tuple[int, int],
    scale_w: float,
    scale_h: float,
) -> List[Dict]:
    if outputs is None or len(outputs) != 9:
        return []

    all_x1, all_y1, all_x2, all_y2, all_scores = [], [], [], [], []

    for i in range(3):
        base = i * 3
        cls_raw = outputs[base + 1]
        gh, gw = cls_raw.shape[2], cls_raw.shape[3]

        cls_flat = cls_raw.reshape(-1)
        mask = cls_flat >= OBJ_THRESH
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue

        box_raw = outputs[base]
        box_flat = box_raw[0].transpose(1, 2, 0).reshape(-1, 64)
        dfl = _dfl_batch(box_flat[idx])

        cols, rows = _grids[i]
        s = _strides[i]

        all_x1.append((cols[idx] + 0.5 - dfl[:, 0]) * s)
        all_y1.append((rows[idx] + 0.5 - dfl[:, 1]) * s)
        all_x2.append((cols[idx] + 0.5 + dfl[:, 2]) * s)
        all_y2.append((rows[idx] + 0.5 + dfl[:, 3]) * s)
        all_scores.append(cls_flat[idx])

    if not all_scores:
        return []

    x1 = np.concatenate(all_x1)
    y1 = np.concatenate(all_y1)
    x2 = np.concatenate(all_x2)
    y2 = np.concatenate(all_y2)
    scores = np.concatenate(all_scores)

    w = x2 - x1
    h = y2 - y1
    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= NMS_THRESH)[0] + 1]

    inv_w = 1.0 / scale_w
    inv_h = 1.0 / scale_h
    results = []
    for i in keep:
        bx1 = int(max(0, min(x1[i] * inv_w, orig_shape[1])))
        by1 = int(max(0, min(y1[i] * inv_h, orig_shape[0])))
        bx2 = int(max(0, min(x2[i] * inv_w, orig_shape[1])))
        by2 = int(max(0, min(y2[i] * inv_h, orig_shape[0])))
        cx = (x1[i] + x2[i]) * 0.5 * inv_w
        cy = (y1[i] + y2[i]) * 0.5 * inv_h
        results.append({
            "box": [bx1, by1, bx2, by2],
            "center": [float(cx), float(cy)],
            "score": float(scores[i]),
        })
    return results


class NPUNozzleDetector:

    MODEL_ID = "nozzle"

    def __init__(
        self,
        engine,
        obj_thresh: float = OBJ_THRESH,
    ) -> None:
        self._obj_thresh = obj_thresh
        self._engine = engine
        self._initialized = False

        if engine is None:
            logging.error("NPU engine is None, cannot initialize NPUNozzleDetector")
            return

        if self.MODEL_ID not in engine.get_loaded_models():
            logging.error(
                f"Model '{self.MODEL_ID}' not loaded in npu_inference engine, "
                f"check [npu_inference nozzle] config"
            )
            return

        self._initialized = True
        logging.info("NPUNozzleDetector initialized successfully")

    def detect(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        if not self._initialized:
            return None

        if img is None or img.size == 0:
            return None

        orig_shape = img.shape[:2]
        input_data, scale_w, scale_h = preprocess(img)

        outputs = self._engine.inference(self.MODEL_ID, input_data)
        if outputs is None:
            return None

        dets = postprocess(outputs, orig_shape, scale_w, scale_h)
        if not dets:
            return None

        best = max(dets, key=lambda d: d["score"])
        cx = int(round(best["center"][0]))
        cy = int(round(best["center"][1]))

        h, w = orig_shape
        if not (5 <= cx <= w - 5 and 5 <= cy <= h - 5):
            return None

        return cx, cy
