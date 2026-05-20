from __future__ import annotations

from typing import Any, Optional, Dict

from ._helpers import _sanitize_property
from .masking import compute_action_mask, ShieldConfig, ShieldResult
from .state_lookup import StateValuationLookup
from . import export as export_mod
from . import serialization as serial_mod

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
            self._constants = constants
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
                self._model_path = prism_program
                self._prism_program = self._parse_prism_program(
                    prism_program, constants
                )
            else:
                self._model_path = None
                self._prism_program = prism_program
            self._constants = constants

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

    # ---- Public properties ----

    @property
    def model(self) -> Any:
        return self._model

    @property
    def prism_program(self) -> Any:
        return self._prism_program

    @property
    def prism_file(self) -> Optional[str]:
        return getattr(self, "_model_path", None)

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

    # ---- Copy / clone ----

    def copy(self) -> "ShieldFactory":
        """Shallow-clone config, share the underlying model."""
        new = object.__new__(ShieldFactory)
        new._model_path = getattr(self, "_model_path", None)
        new._constants = getattr(self, "_constants", None)
        new._prism_program = self._prism_program   # shared
        new._model = self._model                   # intentionally shared
        new._sparse = self._sparse
        new._property = self._property             # str, immutable
        new._formula = getattr(self, "_formula", None)
        new._optimality_type = getattr(self, "_optimality_type", None)
        new._state_valuations = getattr(self, "_state_valuations", None)
        new._choice_labeling = getattr(self, "_choice_labeling", None)
        new._state_lookup = getattr(self, "_state_lookup", None)  # shared
        new._choice_values = getattr(self, "_choice_values", None)
        new._choice_offsets = getattr(self, "_choice_offsets", None)
        new._choice_counts = getattr(self, "_choice_counts", None)
        new._check_result = None  # not carried over; recomputed on demand
        new.action_map = getattr(self, "action_map", None)
        return new

    def __copy__(self) -> "ShieldFactory":
        return self.copy()

    def __deepcopy__(self, memo: dict) -> "ShieldFactory":
        # deepcopy still shares the Storm model — treated as immutable data
        new = self.copy()
        memo[id(self)] = new
        return new

    def with_prop(self, prop: str) -> "ShieldFactory":
        """Return a copy of this factory with a different property string."""
        new = self.copy()
        new.prop = prop
        return new

    # ---- Build ----

    def build(self, config: ShieldConfig) -> Any:
        """
        Run model checking (if not already done) and compute the action mask.
        Returns None when loaded from a slim pickle (model not available).
        """
        logger.trace(f"ShieldFactory.build called with {config=}")
        if self._model is None:
            logger.trace("ShieldFactory.build called on a loaded factory.")
            return None
        if self._check_result is None and getattr(self, "_formula", None) is None:
            raise RuntimeError(
                "Cannot build shield: no formula and no check result available. "
                "Pass a property string."
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
        if self._model is None:
            raise RuntimeError(
                "Cannot compute mask: Storm model not available. "
                "Load with slim=False to include the model in the pickle."
            )
        if self._check_result is None and getattr(self, "_choice_values", None) is None:
            raise RuntimeError(
                "Cannot compute mask: no check result or pre-extracted choice values. "
                "Call build() first."
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
        return program

    def _build_model_from_prism(self) -> Any:
        logger.trace("ShieldFactory._build_model_from_prism called")
        options = stormpy.BuilderOptions(True, True)
        options.set_build_state_valuations()
        options.set_build_choice_labels()
        if self.is_sparse:
            model = stormpy.build_sparse_model_with_options(
                self._prism_program, options
            )
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
        raw = formula[0].raw_formula
        # For GameFormula (SMGs) the optimality type sits on the inner operator
        if getattr(raw, "is_game_formula", False):
            self._optimality_type = raw.subformula.optimality_type
        else:
            self._optimality_type = raw.optimality_type

    # ---- Model checking internals ----

    def _run_model_check(self) -> None:
        if self.formula is None:
            raise RuntimeError(
                "Cannot run model checking: no property set. "
                "Assign a property via ShieldFactory.prop = '...'"
            )
        logger.trace(
            f"ShieldFactory._run_model_check: {str(self.formula.raw_formula)=}"
        )
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
            logger.trace(
                "ShieldFactory._extract_choice_values: already extracted, skipping"
            )
            return
        logger.trace(
            "ShieldFactory._extract_choice_values: extracting from check result"
        )
        result = self._check_result
        choice_vals = np.asarray(result.get_choice_values(), dtype=float)
        counts = np.fromiter((len(s.actions) for s in self._model.states), dtype=int)
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
        if model.model_type not in {stormpy.ModelType.MDP, stormpy.ModelType.SMG}:
            raise TypeError("the model must be an MDP or SMG")
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

    # ---- Dangerous / Critical / Safe / Unsafe helpers

    def _states_with_valuations(self, sids: list[int]) -> list[dict]:
        return [self.state_lookup.values_for_state_id(sid) for sid in sids]

    def dangerous_states(self, *, mask: ShieldResult) -> list[dict]:
        return self._states_with_valuations(mask.dangerous_states())

    def critical_states(self, *, mask: ShieldResult) -> list[dict]:
        return self._states_with_valuations(mask.critical_states())

    def safe_states(self, *, mask: ShieldResult) -> list[dict]:
        return self._states_with_valuations(mask.safe_states())

    def states_by_class(self, *, mask: ShieldResult) -> dict[str, list[dict]]:
        raw = mask.states_by_class()
        return {cls: self._states_with_valuations(sids) for cls, sids in raw.items()}

    # ---- Persistence ----

    def __getstate__(self) -> dict:
        return serial_mod.get_state(self, slim=getattr(self, "_pickle_slim", True))

    def __setstate__(self, state: dict) -> None:
        serial_mod.set_state(self, state)

    def save(self, path: str, *, slim: bool = True) -> None:
        """
        Persist this factory to a pickle file.

        slim=True  (default): choice values and state lookup only — smallest file.
        slim=False:           DRN model bytes + prism file path — fully functional on load.
        """
        serial_mod.save(self, path, slim=slim)

    @classmethod
    def load(cls, path: str) -> "ShieldFactory":
        """
        Load a ShieldFactory saved with save().

        slim=True saves:  state lookup and choice values ready for inference;
                          Storm model not available.
        slim=False saves: Storm model restored from DRN; call build() to
                          re-run model checking if fresh results are needed.
        """
        return serial_mod.load(path)

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

    def to_valuation_action_label_probability(self) -> dict:
        """
        { valuation_key(state) -> { frozenset(action_labels): probability } }
        for ALL actions, not just shield-allowed ones.
        """
        return export_mod.valuation_to_action_label_probability(
            self._model,
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

    def to_storm_format(self, mask: Any) -> str:
        return export_mod.storm_format(self, mask)
