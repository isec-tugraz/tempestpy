from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(slots=True)
class ActionLookup:
    """
    Canonical mapping between PRISM action labels and global RL action indices.

    Multiple PRISM labels may share the same global index:

        ``{"move_left_fast": 0, "move_left_slow": 0, "right": 1}``

    Indices must be contiguous integers starting at 0.  Gaps are rejected at
    construction time to prevent silent bitmask misalignment with the RL
    action space.

    Attributes
    ----------
    label_to_index :
        Maps each PRISM action label to its global RL action index.
    index_to_labels :
        Inverse map: global action index → list of PRISM labels.
    nr_actions :
        Number of distinct global action indices (= ``len(index_to_labels)``).

    See Also
    --------
    ShieldConfig : Accepts an `ActionLookup` to enable global-index bitmasks.
    ActionLookup.from_dict : Primary constructor.
    """

    label_to_index: dict[str, int]
    index_to_labels: dict[int, list[str]]
    nr_actions: int

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, mapping: Mapping[str, int]) -> "ActionLookup":
        """
        Build from a user-supplied {label: global_index} dict.

        Parameters
        ----------
        mapping:
            Maps each model action label to its global RL action index.
            Indices must be contiguous integers starting at 0.

        Raises
        ------
        ValueError
            If mapping is empty or indices are not contiguous from 0.
        """
        if not mapping:
            raise ValueError("action mapping must not be empty")

        label_to_index = dict(mapping)

        indices = sorted(set(label_to_index.values()))
        if indices != list(range(len(indices))):
            raise ValueError(
                f"Action indices must be contiguous integers starting at 0, "
                f"got {indices}. If you have gaps, renumber your action space."
            )

        index_to_labels: dict[int, list[str]] = {i: [] for i in indices}
        for label, idx in label_to_index.items():
            index_to_labels[idx].append(label)

        return cls(
            label_to_index=label_to_index,
            index_to_labels=index_to_labels,
            nr_actions=len(indices),
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def index(self, label: str) -> int:
        """
        Return the global action index for a model action label.

        Raises
        ------
        KeyError
            If the label is not in the mapping.
        """
        try:
            return self.label_to_index[label]
        except KeyError as e:
            raise KeyError(
                f"Action label {label!r} not found in ActionLookup. "
                f"Known labels: {tuple(self.label_to_index)!r}"
            ) from e

    def labels(self, index: int) -> list[str]:
        """
        Return all model action labels that map to a given global index.

        Raises
        ------
        KeyError
            If the index is out of range.
        """
        try:
            return self.index_to_labels[index]
        except KeyError as e:
            raise KeyError(
                f"Action index {index} out of range [0, {self.nr_actions - 1}]"
            ) from e

    def has_label(self, label: str) -> bool:
        """Return ``True`` when *label* is present in this lookup."""
        return label in self.label_to_index

    def has_index(self, index: int) -> bool:
        """Return ``True`` when *index* is a valid global action index."""
        return index in self.index_to_labels

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_model_labels(self, model: object) -> None:
        """
        Check that every action label appearing in the model is present in
        this lookup. Raises ValueError listing any missing labels.

        Parameters
        ----------
        model:
            A stormpy sparse model with state_valuations and choice_labeling.
        """
        model_labels: set[str] = set()
        for state in model.states:
            for action in state.actions:
                for label in action.labels:
                    model_labels.add(label)

        missing = model_labels - set(self.label_to_index)
        if missing:
            raise ValueError(
                f"The following model action labels are missing from ActionLookup: "
                f"{sorted(missing)!r}. Add them to your mapping."
            )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.nr_actions

    def __repr__(self) -> str:
        return (
            f"ActionLookup(nr_actions={self.nr_actions}, "
            f"labels={tuple(self.label_to_index)!r})"
        )
