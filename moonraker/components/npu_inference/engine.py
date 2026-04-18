# NPU Inference Engine for Moonraker
#
# Copyright (C) 2025 Creatbot
#
# This file may be distributed under the terms of the GNU GPLv3 license.

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Dict, Optional, List

import numpy as np

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
        self._models: Dict[str, RKNNLite] = {}
        self._lock = threading.Lock()

        if not HAS_RKNN:
            logging.warning("rknnlite is not installed, NPU inference will not be available")

        prefix_sections = config.get_prefix_sections("npu_inference ")
        for section in prefix_sections:
            model_cfg = config[section]
            model_id = section[len("npu_inference "):]
            model_path = model_cfg.get("path", f"{model_id}.rknn")
            if not os.path.isabs(model_path):
                model_path = os.path.join(MODEL_BASE_DIR, model_path)
            self.load_model(model_id, model_path)

        logging.info("NPUEngine initialized")

    def load_model(self, model_id: str, model_path: str) -> bool:
        if not HAS_RKNN:
            return False
        if model_id in self._models:
            return True

        with self._lock:
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
                logging.info(f"NPU model '{model_id}' loaded")
                return True
            except Exception as e:
                logging.error(f"Failed to load NPU model '{model_id}': {e}")
                return False

    def inference(
        self, model_id: str, input_data: np.ndarray
    ) -> Optional[List[np.ndarray]]:
        rknn = self._models.get(model_id)
        if rknn is None:
            return None

        with self._lock:
            try:
                outputs = rknn.inference(inputs=[input_data])
                if outputs is None or len(outputs) < 1:
                    return None
                return [np.asarray(o) for o in outputs]
            except Exception as e:
                logging.error(f"NPU inference failed for '{model_id}': {e}")
                return None

    def get_loaded_models(self) -> List[str]:
        return list(self._models.keys())


def load_component(config: ConfigHelper) -> NPUEngine:
    return NPUEngine(config)
