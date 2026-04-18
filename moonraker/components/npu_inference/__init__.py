# NPU Inference Package for Moonraker
#
# Copyright (C) 2025 Creatbot
#
# This file may be distributed under the terms of the GNU GPLv3 license.

from __future__ import annotations
from . import engine as _engine

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...confighelper import ConfigHelper


def load_component(config: ConfigHelper) -> _engine.NPUEngine:
    return _engine.load_component(config)
