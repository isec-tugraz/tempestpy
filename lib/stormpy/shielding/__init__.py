from __future__ import annotations

import importlib
import sys

from tempestpy.shielding import *  # noqa: F401,F403
from tempestpy.shielding import __all__  # noqa: F401


for _name in (
    "_helpers",
    "action_lookup",
    "export",
    "masking",
    "serialization",
    "shield_factory",
    "shield_result",
    "state_lookup",
    "wrappers",
):
    try:
        sys.modules[f"{__name__}.{_name}"] = importlib.import_module(
            f"tempestpy.shielding.{_name}"
        )
    except ImportError:
        if _name != "wrappers":
            raise
