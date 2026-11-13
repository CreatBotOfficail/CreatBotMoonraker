# NPU Inference Engine for Moonraker
#
# Copyright (C) 2025 Creatbot
#
# This file may be distributed under the terms of the GNU GPLv3 license.
#
# Backend priority:
#   1) librknnrt.so - vendor C runtime called directly over ctypes
#                     (preallocated outputs, input passed by pointer)
#   2) rknnlite     - previous path (rknn-toolkit-lite2), kept as fallback
#   3) neither      - log a warning, NPU features unavailable (as before)

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Dict, Optional, List

import numpy as np

try:
    from . import rknn_rt
    HAS_RKNN_RT = rknn_rt.is_available()
    RT_ERROR = None if HAS_RKNN_RT else rknn_rt.last_load_error()
except Exception as _e:              # pragma: no cover - unexpected only
    rknn_rt = None
    HAS_RKNN_RT = False
    RT_ERROR = f"import rknn_rt failed: {_e}"

try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    RKNNLite = None
    HAS_RKNN = False

if TYPE_CHECKING:
    from ...confighelper import ConfigHelper


MODEL_BASE_DIR = "/usr/share/model"

class NPUEngine:

    def __init__(self, config: ConfigHelper) -> None:
        self.server = config.get_server()
        self._models: Dict[str, object] = {}
        self._lock = threading.Lock()

        if HAS_RKNN_RT:
            logging.info("NPU backend: librknnrt (direct ctypes)")
        elif HAS_RKNN:
            logging.info("NPU backend: rknnlite (fallback), librknnrt unavailable: %s",
                         RT_ERROR)
        else:
            logging.warning("no NPU backend available "
                            "(librknnrt: %s, rknnlite: not installed)", RT_ERROR)

        prefix_sections = config.get_prefix_sections("npu_inference ")
        for section in prefix_sections:
            model_cfg = config[section]
            model_id = section[len("npu_inference "):]
            model_path = model_cfg.get("path", f"{model_id}.rknn")
            if not os.path.isabs(model_path):
                model_path = os.path.join(MODEL_BASE_DIR, model_path)
            self.load_model(model_id, model_path)

        logging.info("NPUEngine initialized")

    # ------------------------------------------------------------------
    def load_model(self, model_id: str, model_path: str) -> bool:
        if model_id in self._models:
            return True

        with self._lock:
            # 1) vendor runtime over ctypes
            if HAS_RKNN_RT:
                try:
                    model = rknn_rt.RknnRuntime(model_path)
                    self._models[model_id] = model
                    logging.info("NPU model '%s' loaded via librknnrt: input %dx%dx%d, "
                                 "%d outputs", model_id, model.in_w, model.in_h,
                                 model.in_c, model.n_out)
                    return True
                except Exception as e:
                    logging.warning("librknnrt failed to load '%s' (%s), "
                                    "falling back to rknnlite", model_id, e)

            # 2) rknnlite fallback
            if not HAS_RKNN:
                return False
            try:
                rknn = RKNNLite()
                if rknn.load_rknn(model_path) != 0:
                    logging.error(f"load_rknn failed for '{model_id}'")
                    return False
                if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
                    logging.error(f"init_runtime failed for '{model_id}'")
                    rknn.release()
                    return False
                self._models[model_id] = rknn
                logging.info(f"NPU model '{model_id}' loaded via rknnlite")
                return True
            except Exception as e:
                logging.error(f"Failed to load NPU model '{model_id}': {e}")
                return False

    # ------------------------------------------------------------------
    def inference(
        self, model_id: str, input_data: np.ndarray
    ) -> Optional[List[np.ndarray]]:
        model = self._models.get(model_id)
        if model is None:
            return None

        with self._lock:
            try:
                outputs = model.inference(inputs=[input_data])
                if outputs is None or len(outputs) < 1:
                    return None
                return [np.asarray(o) for o in outputs]
            except Exception as e:
                logging.error(f"NPU inference failed for '{model_id}': {e}")
                return None

    # ------------------------------------------------------------------
    def get_loaded_models(self) -> List[str]:
        return list(self._models)


def load_component(config: ConfigHelper) -> NPUEngine:
    return NPUEngine(config)
