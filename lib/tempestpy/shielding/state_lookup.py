from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


StateValuationKey = tuple[tuple[str, Any], ...]


@dataclass(slots=True)
class StateValuationLookup:
    """
    Bidirectional map between model state ids and PRISM variable valuations.

    Built once from a stormpy model and its PRISM program via `from_model`.
    Supports forward lookup (valuation → state id) and reverse lookup
    (state id → valuation dict / key).

    Attributes
    ----------
    variables :
        Ordered tuple of PRISM variable names, in the canonical order used
        for all keys.
    state_id_to_values :
        List of ``{var_name: value}`` dicts, indexed by state id.
    state_id_to_key :
        List of `StateValuationKey` tuples, indexed by state id.
    key_to_state_id :
        Inverse map from `StateValuationKey` to state id.

    See Also
    --------
    ShieldFactory.state_lookup : Lazy accessor that builds this object on demand.
    """

    variables: tuple[str, ...]
    state_id_to_values: list[dict[str, Any]]
    state_id_to_key: list[StateValuationKey]
    key_to_state_id: dict[StateValuationKey, int]

    @staticmethod
    def make_key(
        values: Mapping[str, Any],
        variables: tuple[str, ...],
    ) -> StateValuationKey | None:
        """
        Build a canonical valuation key in fixed PRISM-variable order.

        Extra keys in *values* are silently ignored.  Returns ``None`` when
        a required variable is missing.

        Parameters
        ----------
        values :
            Mapping of variable name to value.
        variables :
            Ordered sequence of variable names that defines the key structure.

        Returns
        -------
        StateValuationKey or None
            Sorted tuple of ``(var, value)`` pairs, or ``None`` on missing key.
        """
        try:
            return tuple((var, values[var]) for var in variables)
        except KeyError as e:
            return None

    @classmethod
    def _extract_state_id_to_values(
        cls,
        model: Any,
        prism_variables: tuple[Any, ...],
    ) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
        """
        Bulk-extract valuations for all model states from model.state_valuations.

        Assumes:
        - `prism_variables` already came from
          prism_program.get_variables(include_constants=False)
        """
        if model is None:
            raise RuntimeError("Model not initialized; cannot build state valuation lookup.")

        sv = getattr(model, "state_valuations", None)
        if sv is None:
            raise RuntimeError(
                "Model has no state_valuations; cannot build state valuation lookup."
            )

        nr_states = int(getattr(model, "nr_states"))
        variables = tuple(getattr(var, "name", str(var)) for var in prism_variables)

        state_id_to_values: list[dict[str, Any]] = [{} for _ in range(nr_states)]

        for var, var_name in zip(prism_variables, variables):
            if var.has_boolean_type():
                values = sv._get_boolean_values_states(var)
                for sid in range(nr_states):
                    state_id_to_values[sid][var_name] = bool(values[sid])

            elif var.has_integer_type():
                values = sv._get_integer_values_states(var)
                for sid in range(nr_states):
                    state_id_to_values[sid][var_name] = int(values[sid])

            elif var.has_rational_type():
                values = sv._get_rational_values_states(var)
                for sid in range(nr_states):
                    state_id_to_values[sid][var_name] = values[sid]

            else:
                raise ValueError(f"Unsupported variable type for {var_name!r}: {var}")

        return variables, state_id_to_values

    @classmethod
    def from_model(
        cls,
        model: Any,
        prism_program: Any,
    ) -> "StateValuationLookup":
        """
        Build the full valuation lookup from a model and its PRISM program.

        Parameters
        ----------
        model :
            A stormpy sparse model built with ``set_build_state_valuations()``.
        prism_program :
            The parsed ``storm::prism::Program`` for the model.

        Returns
        -------
        StateValuationLookup

        Raises
        ------
        ValueError
            If no PRISM variables are found, or if two states share the same
            valuation key (non-unique state space).
        RuntimeError
            If the model has no ``state_valuations`` attribute.
        """
        prism_variables = tuple(prism_program.get_variables(include_constants=False))
        if not prism_variables:
            raise ValueError("No PRISM variables found")

        variables, state_id_to_values = cls._extract_state_id_to_values(
            model=model,
            prism_variables=prism_variables,
        )

        state_id_to_key: list[StateValuationKey] = []
        key_to_state_id: dict[StateValuationKey, int] = {}

        for sid, values in enumerate(state_id_to_values):
            key = cls.make_key(values, variables)
            if key is None:
                continue

            if key in key_to_state_id:
                other_sid = key_to_state_id[key]
                raise ValueError(
                    "Non-unique valuation key encountered for state ids "
                    f"{other_sid} and {sid}: {key!r}"
                )

            state_id_to_key.append(key)
            key_to_state_id[key] = sid

        return cls(
            variables=variables,
            state_id_to_values=state_id_to_values,
            state_id_to_key=state_id_to_key,
            key_to_state_id=key_to_state_id,
        )

    def get_state_id(self, values: Mapping[str, Any]) -> int:
        """
        Return the state id for a given PRISM variable valuation.

        Parameters
        ----------
        values :
            Mapping of variable name to value.  Extra keys are ignored.

        Returns
        -------
        int
            The model state id.

        Raises
        ------
        KeyError
            If any required PRISM variable is missing, or if no state matches
            the supplied valuation.
        """
        key = self.make_key(values, self.variables)
        if key is None:
            raise KeyError(f"Incomplete valuation (missing variables): {values!r} and {self.variables!r}")
        try:
            return self.key_to_state_id[key]
        except KeyError as e:
            raise KeyError(f"Unknown state valuation key: {key!r}") from e

    def has_state(self, values: Mapping[str, Any]) -> bool:
        """
        Return ``True`` when the given valuation maps to a known state.

        Parameters
        ----------
        values :
            Mapping of variable name to value.

        Returns
        -------
        bool
        """
        key = self.make_key(values, self.variables)
        if key is None:
            return False
        return key in self.key_to_state_id

    def get_key(self, values: Mapping[str, Any]) -> StateValuationKey | None:
        """
        Convert *values* to a `StateValuationKey`, or ``None`` on missing variable.

        Parameters
        ----------
        values :
            Mapping of variable name to value.

        Returns
        -------
        StateValuationKey or None
        """
        return self.make_key(values, self.variables)

    def key_for_state_id(self, state_id: int) -> StateValuationKey:
        """
        Return the canonical valuation key for *state_id*.

        Parameters
        ----------
        state_id :
            Model state index.

        Returns
        -------
        StateValuationKey

        Raises
        ------
        KeyError
            If *state_id* is out of range.
        """
        try:
            return self.state_id_to_key[state_id]
        except IndexError as e:
            raise KeyError(f"Unknown state id: {state_id}") from e

    def values_for_state_id(self, state_id: int) -> dict[str, Any]:
        """
        Return the PRISM variable valuation dict for *state_id*.

        Parameters
        ----------
        state_id :
            Model state index.

        Returns
        -------
        dict
            Mapping of PRISM variable name to value.

        Raises
        ------
        KeyError
            If *state_id* is out of range.
        """
        try:
            return self.state_id_to_values[state_id]
        except IndexError as e:
            raise KeyError(f"Unknown state id: {state_id}") from e
