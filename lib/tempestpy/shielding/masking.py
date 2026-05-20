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


@dataclass(frozen=True, slots=True)
class ShieldResult:
    """
    Per-state action masks produced by `ShieldFactory.build`.

    Attributes
    ----------
    config :
        The `ShieldConfig` used to compute this result.
    bitmask_by_state :
        Integer bitmask per state.  Bit *i* is set when action *i* is allowed.
        In *local* mode bit *i* corresponds to ``model.states[sid].actions[i]``;
        in *global* mode (when `ShieldConfig.action_lookup` is set) bit *i*
        corresponds to the global RL action index *i* from `ActionLookup`.
    fallback_by_state :
        Non-zero only for *dangerous* states (no action meets the threshold).
        Holds a single-bit mask for the argmax/argmin action used as a last
        resort.  Zero for safe, critical, and unsafe states.
    nr_actions_by_state :
        Total action count per state.  Equals ``len(state.actions)`` in local
        mode and ``ActionLookup.nr_actions`` in global mode.
    best_value_by_state :
        The highest model-checked choice value across all actions per state.
        ``None`` for DD (symbolic) shields.
    bit_width :
        Explicit bitmask width override, or ``None`` to infer from action count.
    coalition_states :
        Non-``None`` only for SMG models: the set of ego-coalition state ids.
        States absent from this set are adversary/environment states with no mask.

    Notes
    -----
    State classification is fully determined by ``bitmask``, ``fallback``,
    and ``nr_actions``:

    * **unsafe** — ``bitmask == 0`` and ``fallback == 0`` (property violated)
    * **dangerous** — ``fallback != 0`` (no action meets threshold; fallback used)
    * **critical** — ``bitmask > 0`` and ``fallback == 0`` and ``popcount(bitmask) < nr_actions``
    * **safe** — ``bitmask > 0`` and ``fallback == 0`` and ``popcount(bitmask) == nr_actions``

    See Also
    --------
    ShieldFactory.build : Produces a `ShieldResult`.
    ShieldConfig : Configures the threshold and comparison mode.
    """
    config: ShieldConfig
    bitmask_by_state: list[int]
    fallback_by_state: list[int]
    nr_actions_by_state: list[int]
    best_value_by_state: Optional[np.ndarray] = None
    bit_width: Optional[int] = None
    # Non-None only for SMGs: frozenset of state ids belonging to the ego coalition.
    # States absent from this set are adversary/environment states with no mask.
    coalition_states: Optional[frozenset] = None

    # ------------------------------------------------------------------
    # Hot-path helpers (RL loop)
    # ------------------------------------------------------------------

    def query_mask(self, sid: int) -> int:
        """
        Return the effective bitmask for pre-shielding at state *sid*.

        Falls back to `fallback_by_state` when no action meets the threshold
        (dangerous state).

        Parameters
        ----------
        sid :
            State index.

        Returns
        -------
        int
            Packed bitmask of allowed action indices.
        """
        bits = self.bitmask_by_state[sid]
        if bits != 0:
            return bits
        return self.fallback_by_state[sid]

    def query_post(self, sid: int, agent_action: int, **kwargs) -> int:
        """
        Return *agent_action* if it is allowed, otherwise invoke the post-selector.

        Parameters
        ----------
        sid :
            State index used to look up the bitmask.
        agent_action :
            The action chosen by the agent (local or global index).

        Returns
        -------
        int
            The original action if safe, or the replacement action from
            `ShieldConfig.post_selector`.

        Raises
        ------
        TypeError
            If `ShieldConfig.post_selector` is ``None`` and the action is blocked.
        """
        if self.bitmask_by_state[sid] & (1 << agent_action):
            return agent_action
        return self.config.post_selector(sid, agent_action, **kwargs)

    def is_action_allowed(self, sid: int, action_index: int) -> bool:
        """
        Return ``True`` when *action_index* is permitted in state *sid*.

        Parameters
        ----------
        sid :
            State index.
        action_index :
            Local or global action index depending on the bitmask mode.

        Returns
        -------
        bool
        """
        return bool(self.bitmask_by_state[sid] & (1 << action_index))

    def to_bitmask_dict(self) -> dict[int, int]:
        """
        Return a plain ``{state_id: bitmask}`` dictionary.

        Returns
        -------
        dict
            Maps each state id to its packed integer bitmask.
        """
        return {sid: bits for sid, bits in enumerate(self.bitmask_by_state)}

    # ------------------------------------------------------------------
    # State classification helpers (offline / debug)
    # ------------------------------------------------------------------

    def is_coalition_state(self, sid: int) -> bool:
        """
        Return ``True`` if *sid* belongs to the ego coalition.

        Always ``True`` for non-SMG models.  For SMGs, ``False`` for
        adversary/environment states that carry no mask.
        """
        return self.coalition_states is None or sid in self.coalition_states

    def is_unsafe(self, sid: int) -> bool:
        """
        Return ``True`` when the property is violated in state *sid*.

        An unsafe state has no allowed action and no fallback
        (``bitmask == 0`` and ``fallback == 0``).
        """
        if self.coalition_states is not None and sid not in self.coalition_states:
            return False
        return self.bitmask_by_state[sid] == 0 and self.fallback_by_state[sid] == 0

    def is_dangerous(self, sid: int) -> bool:
        """
        Return ``True`` when no action in state *sid* meets the threshold.

        The fallback (best-value) action is used instead.
        """
        return self.fallback_by_state[sid] != 0

    def is_critical(self, sid: int) -> bool:
        """
        Return ``True`` when some — but not all — actions in state *sid* are blocked.
        """
        bits = self.bitmask_by_state[sid]
        return (
            bits > 0
            and self.fallback_by_state[sid] == 0
            and bits.bit_count() < self.nr_actions_by_state[sid]
        )

    def is_safe(self, sid: int) -> bool:
        """Return ``True`` when all actions in state *sid* pass the threshold."""
        return (
            self.fallback_by_state[sid] == 0
            and self.bitmask_by_state[sid].bit_count() == self.nr_actions_by_state[sid]
        )

    def classify_state(self, sid: int) -> str:
        """
        Return the shield class of state *sid*.

        Returns
        -------
        str
            One of ``"safe"``, ``"critical"``, ``"dangerous"``, or ``"unsafe"``.
        """
        if self.is_unsafe(sid):    return "unsafe"
        if self.is_dangerous(sid): return "dangerous"
        if self.is_critical(sid):  return "critical"
        return "safe"

    def states_by_class(self) -> dict[str, list[int]]:
        """
        Partition all state ids into the four shield classes.

        Returns
        -------
        dict
            ``{"safe": [...], "critical": [...], "dangerous": [...], "unsafe": [...]}``,
            where each value is a list of state ids.
        """
        out = {"safe": [], "critical": [], "dangerous": [], "unsafe": []}
        for sid in range(len(self)):
            out[self.classify_state(sid)].append(sid)
        return out

    def dangerous_states(self) -> list[int]:
        """Return the list of state ids classified as dangerous."""
        return [s for s in range(len(self)) if self.is_dangerous(s)]

    def critical_states(self) -> list[int]:
        """Return the list of state ids classified as critical."""
        return [s for s in range(len(self)) if self.is_critical(s)]

    def safe_states(self) -> list[int]:
        """Return the list of state ids classified as safe."""
        return [s for s in range(len(self)) if self.is_safe(s)]

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
        shield_mask = self._check_result.compute_mask_add(self._model, config.threshold)
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

    return ShieldResult(
        config=config,
        bitmask_by_state=bitmask_by_state,
        fallback_by_state=fallback_by_state,
        nr_actions_by_state=nr_actions_by_state,
        best_value_by_state=best_by_state,
        coalition_states=coalition_states,
    )
