from __future__ import annotations

from typing import Any, Optional, Dict
from pathlib import Path

from ._helpers import _sanitize_property
from .masking import compute_action_mask, ShieldConfig
from .state_lookup import StateValuationLookup
from . import export as export_mod

import stormpy
from stormpy import Environment

import numpy as np

from loguru import logger

SparseMDP = stormpy.SparseMdp


def _build_constant_definitions(constants: Dict[str, Any]) -> str:
    """
    Convert a user-supplied {name: value} dict into a PRISM constant definition
    string, e.g. {"N": 3, "slippery": False} -> "N=3,slippery=false".

    Booleans are lowercased to match PRISM syntax.
    """
    parts = []
    for name, value in constants.items():
        if isinstance(value, bool):
            parts.append(f"{name}={'true' if value else 'false'}")
        else:
            parts.append(f"{name}={value}")
    return ",".join(parts)


class ShieldFactory:
    """
    Construct a shield factory.

    Construction modes:
    1. Path + property  (builds model and runs model checking)
    2. Pre-built model + prism program + property  (skips model build, runs model checking)
    3. Pre-built model + prism program + check result  (skips model build and model checking)
    """

    def __init__(
        self,
        model: Any,
        *,
        property: Optional[str] = None,
        constants: Optional[Dict[str, Any]] = None,
        prism_program: Optional[Any] = None,
        check_result: Optional[Any] = None,
        engine: str = "sparse",
    ) -> None:
        self._property = property
        self._check_result = check_result
        if engine not in {"sparse", "dd"}:
            raise ValueError(f"Selected engine {engine} is not supported")
        self._sparse = engine == "sparse"

        if isinstance(model, str):
            # --- Mode 1: path string ---
            logger.trace(f"ShieldFactory.__init__: mode 1 (path), {model=}")
            self._model_path = str(model)
            self._prism_program = self._parse_prism_program(self._model_path, constants)
            if property is not None:
                self._prepare_property(property)
            self._model = self._build_model_from_prism()
        else:
            # --- Mode 2 / 3: pre-built model ---
            logger.trace("ShieldFactory.__init__: mode 2/3 (pre-built model)")
            if prism_program is None:
                raise TypeError(
                    "prism_program is required when passing a pre-built model. "
                    "Use shield.prism_program to obtain it from an existing ShieldFactory, "
                    "or pass a path string."
                )
            # Accept either a parsed program or a path string
            if isinstance(prism_program, str):
                self._prism_program = self._parse_prism_program(prism_program, constants)
            else:
                self._prism_program = prism_program

            self._model = model
            if property is not None:
                self._prepare_property(property)
            # Store labeling references for consistency where sparse shielding needs them.
            if self.is_sparse:
                self._state_valuations = model.state_valuations
                self._choice_labeling = model.choice_labeling

        self._assert_supported_model(self._model)
        if property is not None:
            self._assert_supported_property(property)

    # ---- Alternative constructors ----

    @classmethod
    def from_check_result(
        cls,
        model: Any,
        *,
        prism_program: Any,
        result: Any,
        engine: str = "sparse",
    ) -> "ShieldFactory":
        """
        Build a ShieldFactory from a pre-built model and an already-computed check result.
        No model building or model checking will be performed; compute_mask can be
        called directly.

        prism_program may be a parsed storm::prism::Program or a path string.

        Example::

            shield2 = ShieldFactory.from_check_result(
                shield.model,
                prism_program=shield.prism_program,
                result=my_result,
            )
            masked = shield2.compute_mask(ShieldConfig(...))
        """
        logger.trace("ShieldFactory.from_check_result called")
        return cls(
            model,
            prism_program=prism_program,
            check_result=result,
            engine=engine,
        )

    # ---- Public properties ----

    @property
    def model(self) -> Any:
        return self._model

    @property
    def prism_program(self) -> Any:
        return self._prism_program

    @property
    def prop(self) -> Optional[str]:
        return self._property

    @prop.setter
    def prop(self, value: str) -> None:
        logger.trace(f"ShieldFactory.prop.setter: setting property to {value!r}")
        if value == self._property:
            logger.trace("ShieldFactory.prop.setter: property unchanged, skipping")
            return
        self._property = value
        self._prepare_property(value)
        # Invalidate cached results since the property changed
        self._check_result = None
        self._choice_values = None
        self._choice_offsets = None
        self._choice_counts = None

    @property
    def formula(self) -> Any:
        return getattr(self, "_formula", None)

    @property
    def optimality_type(self) -> Any:
        return getattr(self, "_optimality_type", None)

    # ---- Build ----

    def build(self, config: ShieldConfig) -> Any:
        """
        Run model checking (if not already done) and compute the action mask.
        """
        logger.trace(f"ShieldFactory.build called with {config=}")
        if self._check_result is None and getattr(self, "_formula", None) is None:
            raise RuntimeError(
                "Cannot build shield: no formula and no check result available. "
                "Pass a property string or use ShieldFactory.from_check_result."
            )
        if self._check_result is None:
            self._run_model_check()
        return self.compute_mask(config)

    def compute_mask(self, config: ShieldConfig) -> Any:
        """
        Compute the action mask from already-extracted choice values.
        Calls _extract_choice_values() if not already done.
        """
        logger.trace(f"ShieldFactory.compute_mask called with {config=}")
        if self._check_result is None:
            raise RuntimeError(
                "Cannot compute mask: no check result available. "
                "Call build() first or use ShieldFactory.from_check_result."
            )
        return compute_action_mask(self, config)

    # ---- Prism / model building internals ----

    @staticmethod
    def _parse_prism_program(path: str, constants: Optional[Dict[str, Any]]) -> Any:
        logger.trace(f"ShieldFactory._parse_prism_program: {path=}, {constants=}")
        program = stormpy.parse_prism_program(path)
        if constants:
            definition_string = _build_constant_definitions(constants)
            logger.info(f"defined constants: {definition_string}")
            constant_map = stormpy.parse_constants_string(
                program.expression_manager, definition_string
            )
            program = program.define_constants(constant_map)
        if not program.has_undefined_constants:
            program = program.substitute_constants()
        return program

    def _build_model_from_prism(self) -> Any:
        logger.trace("ShieldFactory._build_model_from_prism called")
        if self.formula is not None:
            options = stormpy.BuilderOptions([self.formula.raw_formula])
        else:
            options = stormpy.BuilderOptions(True, True)
        options.set_build_state_valuations()
        options.set_build_choice_labels()
        if self.is_sparse:
            model = stormpy.build_sparse_model_with_options(self._prism_program, options)
            self._state_valuations = model.state_valuations
            self._choice_labeling = model.choice_labeling
        else:
            properties = [self.formula] if self.formula is not None else None
            model = stormpy.build_symbolic_model(self._prism_program, properties)
        return model

    def _prepare_property(self, property: str) -> None:
        logger.trace(f"ShieldFactory._prepare_property: {property!r}")
        formula = stormpy.parse_properties_for_prism_program(
            _sanitize_property(property), self._prism_program
        )
        assert len(formula) == 1, f"Expected exactly one property, got {len(formula)}"
        self._formula = formula[0]
        self._optimality_type = formula[0].raw_formula.optimality_type

    # ---- Model checking internals ----

    def _run_model_check(self) -> None:
        if self.formula is None:
            raise RuntimeError(
                "Cannot run model checking: no property set. "
                "Assign a property via ShieldFactory.prop = '...'"
            )
        logger.trace(f"ShieldFactory._run_model_check: {str(self.formula.raw_formula)=}")
        logger.info(f"ShieldFactory._run_model_check: {str(self.formula.raw_formula)=}")
        env = Environment()
        env.modelchecker_environment.set_produce_choice_values()
        result = stormpy.model_checking(self._model, self._formula, environment=env)
        assert result.has_choice_values, (
            f"Choice values were not computed for '{self._formula.raw_formula}'"
        )
        self._check_result = result

    def _extract_choice_values(self) -> None:
        """
        Extract choice values, offsets and counts from the stored check result.
        Guarded: no-op if already extracted.
        """
        if getattr(self, "_choice_values", None) is not None:
            logger.trace("ShieldFactory._extract_choice_values: already extracted, skipping")
            return
        logger.trace("ShieldFactory._extract_choice_values: extracting from check result")
        result = self._check_result
        choice_vals = np.asarray(result.get_choice_values(), dtype=float)
        counts = np.fromiter(
            (len(s.actions) for s in self._model.states), dtype=int
        )
        offsets = np.concatenate(([0], np.cumsum(counts)))
        assert offsets[-1] == choice_vals.size, (
            f"Choice value count mismatch: {choice_vals.size} values vs "
            f"{offsets[-1]} expected from state action counts"
        )
        self._choice_values = choice_vals
        self._choice_offsets = offsets
        self._choice_counts = counts

    # ---- Assertions ----

    def _assert_supported_model(self, model: Any) -> None:
        if model is None:
            raise TypeError("model must not be None")
        if model.model_type != stormpy.ModelType.MDP:
            raise TypeError("the model must be an MDP")
        if self.is_sparse:
            if not model.is_sparse_model:
                raise TypeError("engine='sparse' requires a sparse model")
            if not model.has_choice_labeling():
                raise TypeError("the model must be built with choice labeling")
            if not model.has_state_valuations():
                raise TypeError("the model must be built with state valuations")
        elif not model.is_symbolic_model:
            raise TypeError("engine='dd' requires a symbolic model")

    def _assert_supported_property(self, prop: Any) -> None:
        if prop is None:
            raise TypeError("property must not be None")

    @property
    def is_sparse(self) -> bool:
        return self._sparse

    # ---- State lookup ----

    @property
    def state_lookup(self) -> StateValuationLookup:
        if getattr(self, "_state_lookup", None) is None:
            self._state_lookup = StateValuationLookup.from_model(
                self._model,
                self._prism_program,
            )
        return self._state_lookup

    def get_choice_values_for_state(self, sid: int) -> np.ndarray:
        start = self._choice_offsets[sid]
        end = self._choice_offsets[sid + 1]
        return self._choice_values[start:end]

    def get_state_id(self, values: dict) -> int:
        return self.state_lookup.get_state_id(values)

    # ---- Export helpers ----

    def to_bitmask(self, mask: Any) -> Any:
        return export_mod.state_id_to_bitmask(mask)

    def to_allowed_action_indices(self, mask: Any) -> Any:
        return export_mod.state_id_to_allowed_action_indices(self._model, mask)

    def to_allowed_action_labels(self, mask: Any) -> Any:
        return export_mod.state_id_to_allowed_action_labels(self._model, mask)

    def to_allowed_state_action_pairs(self, mask: Any) -> Any:
        return export_mod.allowed_state_action_pairs(self._model, mask)

    def to_valuation_bitmask(self, mask: Any) -> Any:
        return export_mod.valuation_to_bitmask(
            self._model,
            mask,
            self.state_lookup,
        )

    def to_valuation_allowed_action_labels(self, mask: Any) -> Any:
        return export_mod.valuation_to_allowed_action_labels(
            self._model,
            mask,
            self.state_lookup,
        )

    def to_valuation_action_label_probability(self, mask: Any) -> dict:
        """
        { valuation_key(state) -> { frozenset(action_labels): probability } }
        for ALL actions, not just shield-allowed ones.
        """
        return export_mod.valuation_to_action_label_probability(
            self._model,
            mask,
            self.state_lookup,
            lambda sid, i: float(self.get_choice_values_for_state(sid)[i]),
        )

    def pretty_mask(
        self,
        mask: Any,
        *,
        max_states: int = 25,
        show_valuations: bool = True,
        action_ref: str = "labels",
    ) -> str:
        return export_mod.pretty(
            self._model,
            mask,
            self.state_lookup,
            max_states=max_states,
            show_valuations=show_valuations,
            action_ref=action_ref,
        )
