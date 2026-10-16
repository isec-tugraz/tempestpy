from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import stormpy


@dataclass(frozen=True)
class ChoiceData:
    """
    Storm-independent snapshot of everything needed to compute an explicit shield mask.

    Holds the model-checked choice values together with the structural
    information the mask computation needs (per-state offsets, optimisation
    direction, SMG coalition, action labels), so a mask can be computed
    without a Storm model.  This is what ``ShieldFactory.save(slim=True)``
    persists.

    Attributes
    ----------
    values :
        Flat array of choice values, one entry per model choice.
    offsets :
        ``offsets[sid]:offsets[sid + 1]`` is the slice of *values* belonging
        to state ``sid``.  Length is ``nr_states + 1``.
    is_minimize :
        ``True`` when the checked property minimises (allowed means
        ``value <= threshold``).
    coalition_states :
        For SMGs, the states controlled by the coalition; ``None`` otherwise.
        States outside the coalition get an empty mask.
    label_vocab :
        Distinct action-label sets occurring in the model.
    label_ids :
        Per-choice index into *label_vocab* (aligned with *values*).  ``None``
        when the model has no choice labeling.

    See Also
    --------
    tempestpy.shielding.masking.compute_explicit_mask : Consumes this object.
    """

    values: np.ndarray
    offsets: np.ndarray
    is_minimize: bool = False
    coalition_states: Optional[frozenset] = None
    label_vocab: Optional[tuple] = None
    label_ids: Optional[np.ndarray] = None

    @property
    def nr_states(self) -> int:
        return int(self.offsets.size) - 1

    @property
    def counts(self) -> np.ndarray:
        """Number of actions per state."""
        return np.diff(self.offsets)

    def values_for_state(self, sid: int) -> np.ndarray:
        return self.values[self.offsets[sid] : self.offsets[sid + 1]]

    def labels_for_choice(self, sid: int, action_index: int) -> frozenset:
        """Action labels of local action *action_index* in state *sid*."""
        if self.label_ids is None or self.label_vocab is None:
            raise RuntimeError(
                "No action labels stored in this ChoiceData; the model had no choice labeling."
            )
        return self.label_vocab[int(self.label_ids[int(self.offsets[sid]) + action_index])]

    @classmethod
    def from_model(
        cls,
        model: Any,
        check_result: Any,
        formula: Any = None,
        is_minimize: bool = False,
    ) -> "ChoiceData":
        """
        Extract choice data from a sparse model and its check result.

        Parameters
        ----------
        model :
            Sparse stormpy MDP / SMG.
        check_result :
            Check result produced with choice values enabled.
        formula :
            The checked property; used to determine the SMG coalition.
        is_minimize :
            Optimisation direction of the property.
        """
        values = np.asarray(check_result.get_choice_values(), dtype=float)
        nr_states = int(model.nr_states)
        counts = np.empty(nr_states, dtype=int)
        has_labels = model.has_choice_labeling()
        vocab: dict[frozenset, int] = {}
        ids: list[int] = []
        for state in model.states:
            actions = state.actions
            counts[state.id] = len(actions)
            if has_labels:
                for action in actions:
                    ids.append(vocab.setdefault(frozenset(action.labels), len(vocab)))
        offsets = np.concatenate(([0], np.cumsum(counts)))
        if offsets[-1] != values.size:
            raise AssertionError(
                f"Choice value count mismatch: {values.size} values vs "
                f"{offsets[-1]} expected from state action counts"
            )

        coalition_states = None
        if formula is not None and model.model_type == stormpy.ModelType.SMG:
            raw = formula.raw_formula
            if hasattr(raw, "coalition"):
                bv = model.compute_states_of_coalition(raw.coalition)
                coalition_states = frozenset(s for s in range(nr_states) if bv[s])

        return cls(
            values=values,
            offsets=offsets,
            is_minimize=bool(is_minimize),
            coalition_states=coalition_states,
            label_vocab=tuple(vocab) if has_labels else None,
            label_ids=np.asarray(ids, dtype=np.int32) if has_labels else None,
        )
