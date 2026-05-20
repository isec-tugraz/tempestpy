from dataclasses import dataclass, field
from typing import Callable, Literal, Optional
import numpy as np

import stormpy

from .action_lookup import ActionLookup
from .shield_result import ExplicitShieldResult, ShieldResult, SymbolicShieldResult

ComparisonMode = Literal["absolute", "relative"]

PostSelectorFn = Callable[..., int]

BitmaskView = Literal["local", "global"]
ActionKey = str


@dataclass(frozen=True, slots=True)
class ShieldConfig:
    """
    Immutable configuration for a single shield computation.

    Attributes
    ----------
    threshold :
        The probability threshold. In ``"absolute"`` mode an action is allowed
        when its choice value is ``>= threshold`` (maximise) or
        ``<= threshold`` (minimise).  In ``"relative"`` mode the threshold is
        multiplied by the per-state best value before comparison.
        Defaults to ``1.0`` (only optimal actions pass).
    comparison :
        How to interpret *threshold*: ``"absolute"`` compares directly against
        the model-checked probability; ``"relative"`` compares against
        ``threshold * best_value`` for the current state.
    post_selector :
        Optional callable invoked by `ShieldResult.query_post` when the
        agent's chosen action is blocked.  Signature:
        ``(state_id: int, blocked_action: int) -> int``.
        Required when using `PostShieldWrapper`.
    action_lookup :
        Optional `ActionLookup` that maps PRISM action labels to global RL
        action indices.  When set, bitmask bits correspond to global indices
        rather than local model action indices.

    See Also
    --------
    ShieldFactory.build : Consumes a `ShieldConfig` to produce a `ShieldResult`.
    ActionLookup : Maps PRISM labels to RL action indices.
    """

    threshold: float = 1.0
    comparison: ComparisonMode = "absolute"

    post_selector: Optional[PostSelectorFn] = None
    action_lookup: Optional[ActionLookup] = None

    _bitmask_view: BitmaskView = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self, "_bitmask_view",
            "global" if self.action_lookup is not None else "local"
        )


def compute_action_mask(self, config: ShieldConfig) -> ShieldResult:
    if self.is_sparse:
        return compute_explicit_action_mask(self, config)
    return compute_symbolic_action_mask(self, config)


def compute_symbolic_action_mask(self, config: ShieldConfig) -> SymbolicShieldResult:
    if not hasattr(self, "_check_result") or self._check_result is None:
        self._run_model_check()

    assert hasattr(self, "_model") and self._model is not None, "Model not initialized"

    if config.comparison != "absolute":
        raise ValueError("DD shield masks only support absolute threshold comparison")
    if config.action_lookup is not None:
        raise ValueError("DD shield masks do not support action_lookup remapping")
    if not self._check_result.has_choice_values:
        raise RuntimeError("Cannot compute DD shield mask: check result has no choice values")
    shield_mask = self._check_result.compute_mask_add(self._model, config.threshold)
    return SymbolicShieldResult(
        config=config,
        bitmask_add=shield_mask,
        nr_actions=0,
    )


def compute_explicit_action_mask(self, config: ShieldConfig) -> ExplicitShieldResult:
    if not hasattr(self, "_check_result") or self._check_result is None:
        self._run_model_check()

    assert hasattr(self, "_model") and self._model is not None, "Model not initialized"

    if not hasattr(self, "_choice_values") or self._choice_values is None:
        self._extract_choice_values()

    assert hasattr(self, "_choice_offsets") and self._choice_offsets is not None, "Choice offsets not initialized"

    action_lookup: Optional[ActionLookup] = config.action_lookup
    nr_actions_global: Optional[int] = len(action_lookup) if action_lookup is not None else None
    is_minimize = (
        self.optimality_type is not None
        and self.optimality_type == stormpy.OptimizationDirection.Minimize
    )

    nr_states = int(self._model.nr_states)
    bitmask_by_state: list[int] = [0] * nr_states
    fallback_by_state: list[int] = [0] * nr_states
    nr_actions_by_state: list[int] = [0] * nr_states
    best_by_state = np.zeros(nr_states, dtype=float)

    coalition_states: Optional[frozenset] = None
    if self._model.model_type == stormpy.ModelType.SMG:
        raw = self._formula.raw_formula
        if hasattr(raw, "coalition"):
            bv = self._model.compute_states_of_coalition(raw.coalition)
            coalition_states = frozenset(s for s in range(nr_states) if bv[s])

    for state in self._model.states:
        if coalition_states is not None and state.id not in coalition_states:
            continue
        sid = state.id
        start = int(self._choice_offsets[sid])
        end = int(self._choice_offsets[sid + 1])
        pvals = self._choice_values[start:end]

        if pvals.size > 0:
            p_best = float(pvals.min()) if is_minimize else float(pvals.max())
        else:
            p_best = 0.0
        best_by_state[sid] = p_best
        nr_actions_by_state[sid] = nr_actions_global if action_lookup is not None else int(pvals.size)

        # Base predicate — direction depends on optimality type
        threshold = float(config.threshold)
        if config.comparison == "absolute":
            allowed = pvals <= threshold if is_minimize else pvals >= threshold
        elif config.comparison == "relative":
            if is_minimize:
                if p_best == 1.0:
                    if threshold >= 1.0:
                        allowed = np.ones_like(pvals, dtype=bool)
                    else:
                        allowed = np.zeros_like(pvals, dtype=bool)
                else:
                    allowed = pvals <= (threshold * p_best)
            else:
                if p_best == 0.0:
                    if threshold <= 1.0:
                        allowed = np.ones_like(pvals, dtype=bool)
                    else:
                        allowed = np.zeros_like(pvals, dtype=bool)
                else:
                    allowed = pvals >= (threshold * p_best)
        else:
            raise ValueError(f"Unknown comparison mode: {config.comparison!r}")

        bits = 0
        for action_index, is_allowed in enumerate(allowed):
            if is_allowed:
                if action_lookup is None:
                    bits |= (1 << action_index)
                else:
                    action_labels = frozenset(state.actions[action_index].labels)
                    if len(action_labels) > 1:
                        raise ValueError(
                            f"Actions with multiple labels are ambiguous for global index "
                            f"mapping, got: {action_labels}"
                        )
                    if len(action_labels) == 0:
                        continue
                    action_key = next(iter(action_labels))
                    bits |= (1 << action_lookup.index(action_key))

        bitmask_by_state[sid] = bits

        fallback_bits = 0
        if bits == 0 and pvals.size > 0:
            best_local = int(np.argmin(pvals)) if is_minimize else int(np.argmax(pvals))
            if action_lookup is None:
                fallback_bits = 1 << best_local
            else:
                action_labels = frozenset(state.actions[best_local].labels)
                if len(action_labels) > 1:
                    raise ValueError(
                        f"Fallback action has multiple labels, ambiguous: {action_labels}"
                    )
                if len(action_labels) == 0:
                    continue
                action_key = next(iter(action_labels))
                fallback_bits = 1 << action_lookup.index(action_key)
        fallback_by_state[sid] = fallback_bits

    return ExplicitShieldResult(
        config=config,
        bitmask_by_state=bitmask_by_state,
        fallback_by_state=fallback_by_state,
        nr_actions_by_state=nr_actions_by_state,
        best_value_by_state=best_by_state,
        coalition_states=coalition_states,
    )
