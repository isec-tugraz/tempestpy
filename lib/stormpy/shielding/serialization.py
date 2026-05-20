from __future__ import annotations

import os
import pickle
import tempfile
from typing import Any, TYPE_CHECKING

import stormpy
from loguru import logger

if TYPE_CHECKING:
    from .shield_factory import ShieldFactory


# ---- DRN round-trip helpers ----


def model_to_drn_bytes(model: Any) -> bytes:
    """Export a Storm model to DRN bytes via a temporary file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".drn", delete=False)
    tmp.close()
    try:
        stormpy.export_to_drn(model, tmp.name)
        with open(tmp.name, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp.name)


def model_from_drn_bytes(drn_bytes: bytes) -> Any:
    """Restore a Storm model from DRN bytes via a temporary file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".drn", delete=False)
    try:
        tmp.write(drn_bytes)
        tmp.close()
        return stormpy.build_model_from_drn(tmp.name)
    finally:
        os.unlink(tmp.name)


# ---- Pickle state helpers ----


def get_state(factory: ShieldFactory, slim: bool) -> dict:
    """
    Build the pickle state dict for factory.

    slim=True:  choice values + state lookup only (no Storm model).
    slim=False: DRN-serialised Storm model + prism file path for re-checking,
                plus any already-computed choice values.
    """
    state: dict = {
        "slim": slim,
        "property": factory._property,
        "sparse": factory._sparse,
        "action_map": getattr(factory, "action_map", None),
    }
    if slim:
        factory._extract_choice_values()
        _ = factory.state_lookup
        state["state_lookup"] = factory._state_lookup
        state["choice_values"] = factory._choice_values
        state["choice_offsets"] = factory._choice_offsets
        state["choice_counts"] = factory._choice_counts
    else:
        state["drn_bytes"] = model_to_drn_bytes(factory._model)
        state["prism_file"] = getattr(factory, "_model_path", None)
        state["constants"] = getattr(factory, "_constants", None)
        if (
            factory._check_result is not None
            and getattr(factory, "_choice_values", None) is None
        ):
            factory._extract_choice_values()
        _ = factory.state_lookup
        state["state_lookup"] = getattr(factory, "_state_lookup", None)
        state["choice_values"] = getattr(factory, "_choice_values", None)
        state["choice_offsets"] = getattr(factory, "_choice_offsets", None)
        state["choice_counts"] = getattr(factory, "_choice_counts", None)
    return state


def set_state(factory: ShieldFactory, state: dict) -> None:
    """Restore factory in-place from a pickle state dict."""
    slim = state.get("slim", True)
    factory._property = state["property"]
    factory._sparse = state["sparse"]
    factory.action_map = state.get("action_map")
    factory._check_result = None
    factory._formula = None
    factory._optimality_type = None
    factory._state_lookup = state.get("state_lookup")
    factory._choice_values = state.get("choice_values")
    factory._choice_offsets = state.get("choice_offsets")
    factory._choice_counts = state.get("choice_counts")
    if slim:
        factory._model = None
        factory._prism_program = None
        factory._model_path = None
        factory._constants = None
        logger.info(
            "ShieldFactory loaded (slim) — Storm model not available. "
            "State lookup and choice values are ready; model-dependent methods will fail."
        )
    else:
        factory._model = model_from_drn_bytes(state["drn_bytes"])
        factory._model_path = state.get("prism_file")
        factory._constants = state.get("constants")
        if factory._model_path is not None:
            factory._prism_program = factory._parse_prism_program(
                factory._model_path, factory._constants
            )
            if factory._property is not None:
                factory._prepare_property(factory._property)
        else:
            factory._prism_program = None
        logger.info(
            "ShieldFactory loaded (full) — Storm model restored from DRN. "
            "Call build() to run model checking on the restored model."
        )


# ---- High-level persistence ----


def save(factory: ShieldFactory, path: str, slim: bool = True) -> None:
    """
    Persist factory to a pickle file.

    slim=True  (default): choice values and state lookup only — smallest file.
    slim=False:           DRN model bytes + prism file path — fully functional on load.
    """
    factory._pickle_slim = slim
    try:
        with open(path, "wb") as f:
            pickle.dump(factory, f)
    finally:
        del factory._pickle_slim
    logger.info(f"ShieldFactory saved to {path!r} (slim={slim})")


def load(path: str) -> Any:
    """Load a ShieldFactory from a pickle file produced by save()."""
    with open(path, "rb") as f:
        return pickle.load(f)
