from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True, slots=True)
class ShieldResult(ABC):
    """
    Base class for shield results produced by explicit and symbolic backends.

    Explicit shields are keyed by integer state ids. Symbolic shields are
    keyed by PRISM variable valuations, for example ``{"x": 4, "y": 7}``.
    """

    config: Any
    bit_width: Optional[int] = None
    coalition_states: Optional[frozenset] = None

    @abstractmethod
    def _get_bitmask(self, state: Any) -> int:
        ...

    @abstractmethod
    def _get_fallback(self, state: Any) -> int:
        ...

    @abstractmethod
    def _get_nr_actions(self, state: Any) -> int:
        ...

    @abstractmethod
    def _is_unsafe(self, state: Any) -> bool:
        ...

    def query_mask(self, state: Any) -> int:
        """Return the effective pre-shielding bitmask for *state*."""
        bits = self._get_bitmask(state)
        return bits if bits != 0 else self._get_fallback(state)

    def query_post(self, state: Any, agent_action: int, **kwargs: Any) -> int:
        """Return *agent_action* if allowed, otherwise call the post-selector."""
        if self._get_bitmask(state) & (1 << agent_action):
            return agent_action
        if self.config.post_selector is None:
            raise TypeError("ShieldConfig.post_selector is required for post-shielding")
        return self.config.post_selector(state, agent_action, **kwargs)

    def query_action(self, state: Any, agent_action: int, **kwargs: Any) -> int:
        """Alias for ``query_post`` kept for callers that use action wording."""
        return self.query_post(state, agent_action, **kwargs)

    def is_action_allowed(self, state: Any, action_index: int) -> bool:
        return bool(self._get_bitmask(state) & (1 << action_index))

    def is_coalition_state(self, sid: int) -> bool:
        """True for all non-SMG states; for SMGs, true only for ego states."""
        return self.coalition_states is None or sid in self.coalition_states

    def is_unsafe(self, state: Any) -> bool:
        """Return true when the state violates the property outright (risk 1)."""
        return self._is_unsafe(state)

    def is_dangerous(self, state: Any) -> bool:
        """Return true when no action met the threshold and fallback is used."""
        return not self._is_unsafe(state) and self._get_fallback(state) != 0

    def is_critical(self, state: Any) -> bool:
        """Return true when some but not all actions are blocked."""
        if self._is_unsafe(state):
            return False
        bits = self._get_bitmask(state)
        return (
            bits > 0
            and self._get_fallback(state) == 0
            and bits.bit_count() < self._get_nr_actions(state)
        )

    def is_safe(self, state: Any) -> bool:
        """Return true when all actions are allowed."""
        return (
            not self._is_unsafe(state)
            and self._get_fallback(state) == 0
            and self._get_bitmask(state).bit_count() == self._get_nr_actions(state)
        )

    def classify_state(self, state: Any) -> str:
        """
        Return one of ``safe``, ``critical``, ``dangerous``, or ``unsafe``.

        ``unsafe`` takes precedence, so the four classes partition the states.
        """
        if self.is_unsafe(state):
            return "unsafe"
        if self.is_dangerous(state):
            return "dangerous"
        if self.is_critical(state):
            return "critical"
        return "safe"


@dataclass(frozen=True, slots=True)
class ExplicitShieldResult(ShieldResult):
    """Shield result for sparse, explicit-state model checking."""

    bitmask_by_state: list[int] = field(default_factory=list)
    fallback_by_state: list[int] = field(default_factory=list)
    nr_actions_by_state: list[int] = field(default_factory=list)
    best_value_by_state: Optional[np.ndarray] = None
    unsafe_state_ids: frozenset = frozenset()

    def _get_bitmask(self, sid: int) -> int:
        return self.bitmask_by_state[sid]

    def _get_fallback(self, sid: int) -> int:
        return self.fallback_by_state[sid]

    def _get_nr_actions(self, sid: int) -> int:
        return self.nr_actions_by_state[sid]

    def _is_unsafe(self, sid: int) -> bool:
        return sid in self.unsafe_state_ids

    def to_bitmask_dict(self) -> dict[int, int]:
        return {sid: bits for sid, bits in enumerate(self.bitmask_by_state)}

    def states_by_class(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {
            "safe": [],
            "critical": [],
            "dangerous": [],
            "unsafe": [],
        }
        for sid in range(len(self)):
            out[self.classify_state(sid)].append(sid)
        return out

    def unsafe_states(self) -> list[int]:
        return sorted(self.unsafe_state_ids)

    def dangerous_states(self) -> list[int]:
        return [sid for sid in range(len(self)) if self.is_dangerous(sid)]

    def critical_states(self) -> list[int]:
        return [sid for sid in range(len(self)) if self.is_critical(sid)]

    def safe_states(self) -> list[int]:
        return [sid for sid in range(len(self)) if self.is_safe(sid)]

    def __len__(self) -> int:
        return len(self.bitmask_by_state)


@dataclass(frozen=True, slots=True)
class SymbolicShieldResult(ShieldResult):
    """Shield result for DD-backed symbolic model checking."""

    bitmask_add: Any = None
    fallback_add: Any = None
    nr_actions: int = 0

    def _get_bitmask(self, valuation: dict) -> int:
        return int(self.bitmask_add.query(valuation))

    def _get_fallback(self, valuation: dict) -> int:
        if self.fallback_add is None:
            return 0
        return int(self.fallback_add.query(valuation))

    def _is_unsafe(self, valuation: dict) -> bool:
        # Unsafe states are not tracked for DD shields yet.
        return False

    def _get_nr_actions(self, valuation: dict) -> int:
        if self.nr_actions > 0:
            return self.nr_actions
        return max(1, self._get_bitmask(valuation).bit_length())

    @property
    def bitmask_by_state(self) -> Any:
        """Compatibility alias for the ADD-backed symbolic bitmask."""
        return self.bitmask_add

    @property
    def fallback_by_state(self) -> list[int]:
        """Compatibility alias; symbolic fallback is queried via ADDs."""
        return []

    @property
    def nr_actions_by_state(self) -> list[int]:
        """Compatibility alias; symbolic action count is uniform when known."""
        return []

    @property
    def best_value_by_state(self) -> None:
        return None

    def to_bitmask_dict(self) -> dict[int, int]:
        raise TypeError("SymbolicShieldResult cannot be converted to state-id bitmasks")
