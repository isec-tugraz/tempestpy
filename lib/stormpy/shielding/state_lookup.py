from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


StateValuationKey = tuple[tuple[str, Any], ...]


@dataclass(slots=True)
class StateValuationLookup:
    """
    Canonical lookup between model-state valuations and state ids.
    """
    variables: tuple[str, ...]
    state_id_to_values: list[dict[str, Any]]
    state_id_to_key: list[StateValuationKey]
    key_to_state_id: dict[StateValuationKey, int]

    @staticmethod
    def make_key(
        values: Mapping[str, Any],
        variables: tuple[str, ...],
    ) -> StateValuationKey:
        """
        Build a canonical valuation key in fixed PRISM-variable order.

        Extra keys in `values` are ignored.
        Missing required variables raise KeyError.
        """
        try:
            return tuple((var, values[var]) for var in variables)
        except KeyError as e:
            missing = e.args[0]
            raise KeyError(
                f"Missing valuation for variable {missing!r}. "
                f"Expected variables: {variables!r}"
            ) from e

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
        Build the full valuation lookup directly from model + prism_program.
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
        key = self.make_key(values, self.variables)
        try:
            return self.key_to_state_id[key]
        except KeyError as e:
            raise KeyError(f"Unknown state valuation key: {key!r}") from e

    def has_state(self, values: Mapping[str, Any]) -> bool:
        key = self.make_key(values, self.variables)
        return key in self.key_to_state_id

    def get_key(self, values: Mapping[str, Any]) -> StateValuationKey:
        return self.make_key(values, self.variables)

    def key_for_state_id(self, state_id: int) -> StateValuationKey:
        try:
            return self.state_id_to_key[state_id]
        except IndexError as e:
            raise KeyError(f"Unknown state id: {state_id}") from e

    def values_for_state_id(self, state_id: int) -> dict[str, Any]:
        try:
            return self.state_id_to_values[state_id]
        except IndexError as e:
            raise KeyError(f"Unknown state id: {state_id}") from e

