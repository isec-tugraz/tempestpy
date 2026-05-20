from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, FrozenSet, Mapping, Tuple

import numpy as np

from .shield_result import ShieldResult


# A hashable, stable key derived from state valuations.
# We deliberately do NOT use stormpy state objects as dict keys.
StateValuationKey = Tuple[Tuple[str, Any], ...]
StateActionValueLookup = (
    Mapping[tuple[int, int], float] | Callable[[int, int], float]
)


# -----------------------------
# State valuation helpers
# -----------------------------

def _state_values(state_lookup: Any, state_id: int) -> Mapping[str, Any]:
    """
    Resolve a state's valuation mapping.

    Supports either:
      - a plain mapping: {state_id -> {var_name: value}}
      - an object exposing one of a few conventional lookup methods
    """
    if isinstance(state_lookup, Mapping):
        return state_lookup[state_id]

    for attr in (
        "get_state_values",
        "get_values",
        "state_values",
        "valuations_for_state",
    ):
        fn = getattr(state_lookup, attr, None)
        if callable(fn):
            return fn(state_id)

    for attr in (
        "state_id_to_values",
        "_state_id_to_values",
        "by_state_id",
    ):
        obj = getattr(state_lookup, attr, None)
        if obj is not None:
            return obj[state_id]

    # Last resort: try __getitem__
    try:
        return state_lookup[state_id]
    except Exception as exc:
        raise TypeError(
            "state_lookup must be either a mapping {state_id -> valuations} "
            "or expose a compatible accessor such as get_state_values(state_id)"
        ) from exc


def valuation_key(state_id: int, state_lookup: Any) -> StateValuationKey:
    """
    Convert a state id into a stable, hashable key based on its valuations.

    The key is a sorted tuple of (var_name, value) pairs so it is deterministic
    and usable as a dictionary key.
    """
    return tuple(sorted(_state_values(state_lookup, state_id).items(), key=lambda kv: kv[0]))


# -----------------------------
# Bitmask / action helpers
# -----------------------------

def _allowed_indices_from_bits(bits: int, n_actions: int) -> list[int]:
    """Return local action indices whose bit is set."""
    return [i for i in range(n_actions) if bits & (1 << i)]


def _action_labels_for_state(state: Any) -> list[FrozenSet[str]]:
    """Return frozenset(labels) in local action order for one state."""
    return [frozenset(a.labels) for a in state.actions]


def _get_state_action_value(
    state_action_values: StateActionValueLookup,
    state_id: int,
    action_index: int,
) -> float:
    """
    Resolve a value for a (state_id, local_action_index) pair.

    Supports either:
      - a mapping: {(state_id, action_index): value}
      - a callable: f(state_id, action_index) -> value
    """
    if callable(state_action_values):
        return float(state_action_values(state_id, action_index))
    return float(state_action_values[(state_id, action_index)])


# -----------------------------
# Basic “ID-keyed” exports
# -----------------------------

def state_id_to_bitmask(mask: ShieldResult) -> dict[int, int]:
    """
    Convert a shield result to a ``{state_id: bitmask}`` dictionary.

    Bit *i* in each bitmask corresponds to local action index *i* in
    ``model.states[state_id].actions``.

    Parameters
    ----------
    mask :
        A `ShieldResult`.

    Returns
    -------
    dict
        Maps each state id to its packed integer bitmask.
    """
    return mask.to_bitmask_dict()


def state_id_to_allowed_action_indices(model: Any, mask: ShieldResult) -> dict[int, list[int]]:
    """
    Map each state id to the list of allowed local action indices.

    Parameters
    ----------
    model :
        The stormpy model associated with *mask*.
    mask :
        A `ShieldResult`.

    Returns
    -------
    dict
        ``{state_id: [action_index, ...]}``.
    """
    out: dict[int, list[int]] = {}
    for state in model.states:
        state_id = int(state.id)
        bits = mask.bitmask_by_state[state_id]
        out[state_id] = _allowed_indices_from_bits(bits, len(state.actions))
    return out


def state_id_to_allowed_action_labels(model: Any, mask: ShieldResult) -> dict[int, list[FrozenSet[str]]]:
    """
    Map each state id to the label-sets of allowed actions.

    Parameters
    ----------
    model :
        The stormpy model associated with *mask*.
    mask :
        A `ShieldResult`.

    Returns
    -------
    dict
        ``{state_id: [frozenset(labels), ...]}``, aligned with allowed local
        action indices.
    """
    out: dict[int, list[FrozenSet[str]]] = {}
    for state in model.states:
        state_id = int(state.id)
        bits = mask.bitmask_by_state[state_id]
        allowed = _allowed_indices_from_bits(bits, len(state.actions))
        labels = _action_labels_for_state(state)
        out[state_id] = [labels[i] for i in allowed]
    return out


def allowed_state_action_pairs(
    model: Any,
    mask: ShieldResult,
) -> list[tuple[int, int]]:
    """
    Return all allowed ``(state_id, local_action_index)`` pairs.

    Parameters
    ----------
    model :
        The stormpy model associated with *mask*.
    mask :
        A `ShieldResult`.

    Returns
    -------
    list of tuple
        Each entry is ``(state_id, action_index)``.
    """
    pairs: list[tuple[int, int]] = []
    for state in model.states:
        state_id = int(state.id)
        bits = mask.bitmask_by_state[state_id]
        for i in _allowed_indices_from_bits(bits, len(state.actions)):
            pairs.append((state_id, i))
    return pairs


# -----------------------------
# “Valuation-keyed” exports
# -----------------------------

def valuation_to_bitmask(
    model: Any,
    mask: ShieldResult,
    state_lookup: Any,
) -> dict[StateValuationKey, int]:
    """
    Map each state's valuation key to its packed bitmask.

    Parameters
    ----------
    model :
        The stormpy model associated with *mask*.
    mask :
        A `ShieldResult`.
    state_lookup :
        A `StateValuationLookup` (or compatible mapping) for the model.

    Returns
    -------
    dict
        ``{StateValuationKey: bitmask}``.
    """
    out: dict[StateValuationKey, int] = {}
    for state in model.states:
        state_id = int(state.id)
        out[valuation_key(state_id, state_lookup)] = mask.bitmask_by_state[state_id]
    return out


def valuation_to_allowed_action_labels(
    model: Any,
    mask: ShieldResult,
    state_lookup: Any,
) -> dict[StateValuationKey, list[FrozenSet[str]]]:
    """
    Map each state's valuation key to the label-sets of allowed actions.

    Parameters
    ----------
    model :
        The stormpy model associated with *mask*.
    mask :
        A `ShieldResult`.
    state_lookup :
        A `StateValuationLookup` for the model.

    Returns
    -------
    dict
        ``{StateValuationKey: [frozenset(labels), ...]}``.
    """
    out: dict[StateValuationKey, list[FrozenSet[str]]] = {}
    for state in model.states:
        state_id = int(state.id)
        bits = mask.bitmask_by_state[state_id]
        allowed = _allowed_indices_from_bits(bits, len(state.actions))
        labels = _action_labels_for_state(state)
        out[valuation_key(state_id, state_lookup)] = [labels[i] for i in allowed]
    return out


def valuation_to_allowed_action_labels_with_values(
    model: Any,
    mask: ShieldResult,
    state_lookup: Any,
    state_action_values: StateActionValueLookup,
) -> dict[StateValuationKey, dict[FrozenSet[str], float]]:
    """
    Map each state's valuation key to a ``{action_labels: value}`` dict for allowed actions.

    Parameters
    ----------
    model :
        The stormpy model associated with *mask*.
    mask :
        A `ShieldResult`.
    state_lookup :
        A `StateValuationLookup` for the model.
    state_action_values :
        Either a mapping ``{(state_id, action_index): float}`` or a callable
        ``f(state_id, action_index) -> float``.

    Returns
    -------
    dict
        ``{StateValuationKey: {frozenset(labels): value}}``.

    Notes
    -----
    If multiple allowed actions in the same state share the same label set,
    later entries overwrite earlier ones (dict keys must be unique).
    """
    out: dict[StateValuationKey, dict[FrozenSet[str], float]] = {}
    for state in model.states:
        state_id = int(state.id)
        bits = mask.bitmask_by_state[state_id]
        allowed = _allowed_indices_from_bits(bits, len(state.actions))
        labels = _action_labels_for_state(state)
        out[valuation_key(state_id, state_lookup)] = {
            labels[i]: _get_state_action_value(state_action_values, state_id, i)
            for i in allowed
        }
    return out


def valuation_to_allowed_action_ids(
    model: Any,
    mask: ShieldResult,
    state_lookup: Any,
) -> dict[StateValuationKey, list[int]]:
    """
    Map each state's valuation key to the Storm action ids of allowed actions.

    Parameters
    ----------
    model :
        The stormpy model associated with *mask*.
    mask :
        A `ShieldResult`.
    state_lookup :
        A `StateValuationLookup` for the model.

    Returns
    -------
    dict
        ``{StateValuationKey: [storm_action_id, ...]}``.
    """
    out: dict[StateValuationKey, list[int]] = {}
    for state in model.states:
        state_id = int(state.id)
        bits = mask.bitmask_by_state[state_id]
        allowed = _allowed_indices_from_bits(bits, len(state.actions))
        action_ids = [int(a.id) for a in state.actions]
        out[valuation_key(state_id, state_lookup)] = [action_ids[i] for i in allowed]
    return out


# -----------------------------
# Pretty / debug helpers
# -----------------------------

def mask_metadata(mask: ShieldResult) -> dict[str, Any]:
    """
    Serialize the non-model parts of a `ShieldResult` for logging or debugging.

    Parameters
    ----------
    mask :
        A `ShieldResult`.

    Returns
    -------
    dict
        Contains ``"config"``, ``"nr_states"``, and (if available)
        ``"best_value_by_state_summary"`` with min/max/mean statistics.
    """
    out = {
        "config": asdict(mask.config),
        "nr_states": len(mask.bitmask_by_state),
    }

    best = getattr(mask, "best_value_by_state", None)
    if best is not None:
        out["best_value_by_state_summary"] = {
            "min": float(np.min(best)) if best.size else 0.0,
            "max": float(np.max(best)) if best.size else 0.0,
            "mean": float(np.mean(best)) if best.size else 0.0,
        }

    return out

def valuation_to_action_label_probability(
    model: Any,
    state_lookup: Any,
    state_action_values: StateActionValueLookup,
) -> dict[StateValuationKey, dict[FrozenSet[str], float]]:
    """
    Map every state to *all* its actions and their model-checked probabilities.

    Unlike `valuation_to_allowed_action_labels_with_values`, this function
    includes blocked actions as well.

    Parameters
    ----------
    model :
        The stormpy model.
    state_lookup :
        A `StateValuationLookup` for the model.
    state_action_values :
        Either a mapping ``{(state_id, action_index): float}`` or a callable
        ``f(state_id, action_index) -> float``.
        `ShieldFactory.get_choice_values_for_state` provides a compatible callable.

    Returns
    -------
    dict
        ``{StateValuationKey: {frozenset(action_labels): probability}}``.

    Notes
    -----
    If multiple actions in the same state share the same label set, later
    entries overwrite earlier ones (dict keys must be unique).
    """
    out: dict[StateValuationKey, dict[FrozenSet[str], float]] = {}
    for state in model.states:
        state_id = int(state.id)
        labels = _action_labels_for_state(state)
        out[valuation_key(state_id, state_lookup)] = {
            labels[i]: _get_state_action_value(state_action_values, state_id, i)
            for i in range(len(state.actions))
        }
    return out


def pretty(
    model: Any,
    mask: ShieldResult,
    state_lookup: Any,
    *,
    max_states: int = 25,
    show_valuations: bool = True,
    action_ref: str = "labels",
) -> str:
    """
    Return a human-readable summary of a shield mask.

    Parameters
    ----------
    model :
        The stormpy model associated with *mask*.
    mask :
        A `ShieldResult`.
    state_lookup :
        A `StateValuationLookup` for the model.
    max_states :
        Maximum number of states to include (default 25).
    show_valuations :
        If ``True`` (default), show PRISM variable values next to each state id.
    action_ref :
        ``"labels"`` (default) to show ``frozenset(labels)`` per action;
        ``"index"`` to show local action indices only.

    Returns
    -------
    str
        Multi-line human-readable shield summary.
    """
    if action_ref not in {"labels", "index"}:
        raise ValueError("action_ref must be one of: 'labels', 'index'")

    lines: list[str] = []
    lines.append(f"ShieldResult(config={asdict(mask.config)})")

    count = 0
    total_states = len(mask.bitmask_by_state)

    for state in model.states:
        if count >= max_states:
            lines.append(f"... ({total_states - max_states} more states)")
            break

        state_id = int(state.id)
        bits = mask.bitmask_by_state[state_id]
        allowed_idx = _allowed_indices_from_bits(bits, len(state.actions))

        if show_valuations:
            vk = valuation_key(state_id, state_lookup)
            lines.append(
                f"state {state_id} {vk}: allowed={len(allowed_idx)}/{len(state.actions)}"
            )
        else:
            lines.append(
                f"state {state_id}: allowed={len(allowed_idx)}/{len(state.actions)}"
            )

        if action_ref == "labels":
            labels = _action_labels_for_state(state)
            for i in allowed_idx:
                lines.append(f"  - a[{i}] labels={set(labels[i])}")
        else:
            for i in allowed_idx:
                lines.append(f"  - a[{i}]")

        count += 1

    return "\n".join(lines)


# -----------------------------
# Storm-compatible text format
# -----------------------------

def storm_format(factory: Any, mask: ShieldResult) -> str:
    """
    Render a shield mask as a Storm-compatible shield text.

    Parameters
    ----------
    factory :
        The `ShieldFactory` that produced *mask*.  Used to resolve choice
        values and state valuations.
    mask :
        A `ShieldResult`.

    Returns
    -------
    str
        Multi-line string matching Storm's shield output format.
        Each line shows ``state_id: [var=val & ...] val: (idx {label}); ...``
        for every action and its model-checked probability.
    """
    cfg = mask.config
    comparison_name = "absolute" if cfg.comparison == "absolute" else "relative"
    header = f"Pre-Safety-Shield with {comparison_name} comparison (gamma = {cfg.threshold:.6f}):"

    factory._extract_choice_values()
    state_lookup = factory.state_lookup

    rows: list[tuple[int, str, str]] = []
    for state in factory._model.states:
        sid = int(state.id)
        vals = state_lookup.values_for_state_id(sid)
        val_str = "    & ".join(f"{v}={vals[v]}" for v in state_lookup.variables)

        pvals = factory.get_choice_values_for_state(sid)
        action_entries = []
        for i, action in enumerate(state.actions):
            label = next(iter(action.labels), str(i))
            action_entries.append(f"{pvals[i]:g}: ({i} {{{label}}})")
        actions_str = ";    ".join(action_entries)

        rows.append((sid, val_str, actions_str))

    if not rows:
        return header

    id_width = max(len(str(sid)) for sid, _, _ in rows)
    val_width = max(len(vs) for _, vs, _ in rows)
    # Align "model state:" header to the right edge of the valuation column.
    # State line layout: "{sid}: [{val_str}]    {actions}"
    state_col = id_width + 4 + val_width  # ": [" (3) + "]" (1)
    col_header = f"{'model state:':>{state_col}}    choice(s) [<value>: (<action {{action label}})>]:"

    lines = [header, col_header]
    for sid, val_str, actions_str in rows:
        lines.append(f"{sid:{id_width}d}: [{val_str}]    {actions_str}")

    return "\n".join(lines)
