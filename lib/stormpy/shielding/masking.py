from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional
import numpy as np

from loguru import logger
import stormpy

from .action_lookup import ActionLookup

ComparisonMode = Literal["absolute", "relative"]

PostSelectorFn = Callable[..., int]

BitmaskView = Literal["local", "global"]
ActionKey = str


@dataclass(frozen=True, slots=True)
class ShieldConfig:
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


@dataclass(frozen=True, slots=True)
class ShieldResult:
    """
    bitmask_by_state[sid]:
      integer bitmask where bit i corresponds to:
        - local mode:  the local action index i in model.states[sid].actions
        - global mode: the global RL action index i from ActionLookup

    fallback_by_state[sid]:
      non-zero only for dangerous states (no action meets threshold).
      equals bitmask_by_state[sid] in that case, argmax action promoted
      as last resort. zero for safe, critical, and unsafe states.

    nr_actions_by_state[sid]:
      - local mode:  len(state.actions) for state sid
      - global mode: ActionLookup.nr_actions (constant across all states)

    best_value_by_state[sid]:
      the highest choice value across all actions in state sid.

    State classification (derivable from bitmask + fallback + nr_actions):
      unsafe:    bitmask == 0 and fallback == 0   (property violated)
      dangerous: bitmask > 0 and bitmask == fallback  (no action meets threshold)
      critical:  bitmask > 0 and fallback == 0 and popcount(bitmask) < nr_actions
      safe:      bitmask > 0 and fallback == 0 and popcount(bitmask) == nr_actions
    """
    config: ShieldConfig
    bitmask_by_state: list[int]
    fallback_by_state: list[int]
    nr_actions_by_state: list[int]
    best_value_by_state: Optional[np.ndarray] = None
    bit_width: Optional[int] = None

    # ------------------------------------------------------------------
    # Hot-path helpers (RL loop)
    # ------------------------------------------------------------------

    def query_mask(self, sid: int) -> int:
        """Pre-shielding: return the bitmask for the RL wrapper to apply."""
        bits = self.bitmask_by_state[sid]
        if bits != 0:
            return bits
        return self.fallback_by_state[sid]

    def query_action(self, sid: int, agent_action: int, pvals: np.ndarray, **kwargs) -> int:
        """Post-shielding: return agent_action if safe, else invoke post_selector."""
        if self.bitmask_by_state[sid] & (1 << agent_action):
            return agent_action
        return self.config.post_selector(pvals, **kwargs)

    def is_action_allowed(self, sid: int, action_index: int) -> bool:
        return bool(self.bitmask_by_state[sid] & (1 << action_index))

    def to_bitmask_dict(self) -> dict[int, int]:
        return {sid: bits for sid, bits in enumerate(self.bitmask_by_state)}

    # ------------------------------------------------------------------
    # State classification helpers (offline / debug)
    # ------------------------------------------------------------------

    def is_unsafe(self, sid: int) -> bool:
        """property violated, no action available at all."""
        return self.bitmask_by_state[sid] == 0 and self.fallback_by_state[sid] == 0

    def is_dangerous(self, sid: int) -> bool:
        """No action meets threshold."""
        return self.fallback_by_state[sid] != 0

    def is_critical(self, sid: int) -> bool:
        """Some actions blocked by threshold."""
        bits = self.bitmask_by_state[sid]
        return (
            bits > 0
            and self.fallback_by_state[sid] == 0
            and bits.bit_count() < self.nr_actions_by_state[sid]
        )

    def is_safe(self, sid: int) -> bool:
        """All actions permitted."""
        return (
            self.fallback_by_state[sid] == 0
            and self.bitmask_by_state[sid].bit_count() == self.nr_actions_by_state[sid]
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.bitmask_by_state)




def compute_action_mask(self, config: ShieldConfig) -> Any:
    if not hasattr(self, "_check_result") or self._check_result is None:
        self._run_model_check()

    assert hasattr(self, "_model") and self._model is not None, "Model not initialized"

    if not self.is_sparse:
        if config.comparison != "absolute":
            raise ValueError("DD shield masks only support absolute threshold comparison")
        if config.action_lookup is not None:
            raise ValueError("DD shield masks do not support action_lookup remapping")
        if not self._check_result.has_choice_values:
            raise RuntimeError("Cannot compute DD shield mask: check result has no choice values")
        shield_mask = self._check_result.compute_shield_mask(self._model, config.threshold)
        return ShieldResult(
            config=config,
            bitmask_by_state=shield_mask,
            fallback_by_state=[],
            nr_actions_by_state=[],
        )

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

    for state in self._model.states:
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
                    if len(action_labels) != 1:
                        raise ValueError(
                            f"Actions with multiple labels are ambiguous for global index "
                            f"mapping, got: {action_labels}"
                        )
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
                if len(action_labels) != 1:
                    raise ValueError(
                        f"Fallback action has multiple labels, ambiguous: {action_labels}"
                    )
                action_key = next(iter(action_labels))
                fallback_bits = 1 << action_lookup.index(action_key)
        fallback_by_state[sid] = fallback_bits

    return ShieldResult(
        config=config,
        bitmask_by_state=bitmask_by_state,
        fallback_by_state=fallback_by_state,
        nr_actions_by_state=nr_actions_by_state,
        best_value_by_state=best_by_state,
    )
