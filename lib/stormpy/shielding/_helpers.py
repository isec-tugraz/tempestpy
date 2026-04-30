from __future__ import annotations

import numpy as np
import warnings

def _sanitize_property(prop: str) -> str:
    """
    Replace single quotes by double quotes for PRISM compatibility.
    """
    if "'" in prop:
        warnings.warn(
            "Property contains single quotes. Replacing ' with \" for PRISM parsing.",
            UserWarning,
            stacklevel=2,
        )
        prop = prop.replace("'", '"')
    return prop



def int_to_bitmask(bits: int, nr_actions: int) -> list[bool]:
    return [(bits & (1 << i)) != 0 for i in range(nr_actions)]


def int_to_bitmask_array(bits: int, nr_actions: int) -> np.ndarray:
    return np.fromiter(
        ((bits & (1 << i)) != 0 for i in range(nr_actions)),
        dtype=bool,
        count=nr_actions,
    )


def bitmask_to_int(mask: list[bool] | np.ndarray) -> int:
    bits = 0
    for i, allowed in enumerate(mask):
        if allowed:
            bits |= (1 << i)
    return bits


def is_action_allowed(bits: int, action_index: int) -> bool:
    return bool(bits & (1 << action_index))


def allowed_action_indices(bits: int, nr_actions: int) -> list[int]:
    return [i for i in range(nr_actions) if bits & (1 << i)]
