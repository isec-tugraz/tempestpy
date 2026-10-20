from __future__ import annotations

import re
from typing import Any, Optional, Dict

from ._helpers import _sanitize_property
from .choice_data import ChoiceData
from .masking import compute_action_mask, ShieldConfig
from .shield_result import ShieldResult
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
    Entry point for computing probabilistic shields from PRISM models.

    A `ShieldFactory` parses or receives a PRISM MDP/SMG, runs model
    checking to obtain per-action choice values, and converts those values
    into `ShieldResult` bitmasks via `build` or `compute_mask`.

    See Also
    --------
    ShieldConfig : Threshold and comparison-mode settings passed to `build`.
    ShieldResult : The bitmask container returned by `build`.
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
        """
        Initialise a ShieldFactory.

        Three construction modes are supported, selected automatically based
        on the type of *model* and the keyword arguments supplied:

        1. **Path mode** — pass a PRISM file path as *model*.
           The factory parses the program, builds the model, and (if *property*
           is given) runs model checking immediately.
        2. **Pre-built model mode** — pass a stormpy model object as *model*
           together with *prism_program* and *property*.
           Model building is skipped; model checking runs on `build`.
        3. **Pre-computed result mode** — pass a stormpy model object,
           *prism_program*, and *check_result*.
           Both model building and model checking are skipped; `compute_mask`
           can be called directly.

        Parameters
        ----------
        model :
            Either a path string to a ``.prism`` / ``.nm`` file (mode 1),
            or a pre-built stormpy sparse/symbolic MDP or SMG (modes 2 & 3).
        property :
            A PCTL/PRCTL property string, e.g. ``"Pmax=? [G safety]"``.
            Required for modes 1 and 2; omit only when supplying *check_result*.
        constants :
            PRISM constant definitions, e.g. ``{"N": 3, "slippery": False}``.
            Only used when the factory parses the PRISM program itself
            (mode 1, or mode 2 when *prism_program* is a path string).
        prism_program :
            A parsed ``storm::prism::Program`` or a path string to the PRISM
            source file. Required for modes 2 and 3.
        check_result :
            A pre-computed stormpy check result that already contains choice
            values. When provided, model checking is skipped entirely (mode 3).
        engine :
            Backend representation. ``"sparse"`` (default) uses a sparse
            explicit-state model; ``"dd"`` uses a symbolic BDD-based model.

        Raises
        ------
        ValueError
            If *engine* is not ``"sparse"`` or ``"dd"``.
        TypeError
            If a pre-built model is passed without *prism_program*, or if the
            model type or engine/model combination is unsupported.

        Examples
        --------
        Mode 1 — build everything from a file:

        >>> factory = ShieldFactory("model.prism", property="Pmax=? [G safe]",
        ...                         constants={"N": 5})
        >>> result = factory.build(ShieldConfig(threshold=0.9))

        Mode 3 — reuse an existing check result:

        """
        self._property = property
        self._check_result = check_result
        self._choice_data: Optional[ChoiceData] = None
        self._unsafe_state_ids = None
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
        """The stormpy model held by this factory."""
        return self._model

    @property
    def prism_program(self) -> Any:
        """The parsed ``storm::prism::Program`` held by this factory."""
        return self._prism_program

    @property
    def prism_file(self) -> Optional[str]:
        return getattr(self, "_model_path", None)

    @property
    def prop(self) -> Optional[str]:
        """The active PCTL property string, or ``None`` if not set."""
        return self._property

    @prop.setter
    def prop(self, value: str) -> None:
        logger.trace(f"ShieldFactory.prop.setter: setting property to {value!r}")
        if value == self._property:
            logger.trace("ShieldFactory.prop.setter: property unchanged, skipping")
            return
        if self._prism_program is None:
            raise RuntimeError(
                "Cannot change the property of a slim-loaded ShieldFactory: "
                "no PRISM program available. Load with slim=False or create a new factory."
            )
        self._property = value
        self._prepare_property(value)
        # Invalidate cached results since the property changed
        self._check_result = None
        self._choice_data = None
        self._unsafe_state_ids = None

    @property
    def formula(self) -> Any:
        """The parsed stormpy formula, or ``None`` before a property is set."""
        return getattr(self, "_formula", None)

    @property
    def optimality_type(self) -> Any:
        """
        The optimality direction (``Minimize`` / ``Maximize``) derived from
        the formula, or ``None`` if no formula has been set.
        """
        return getattr(self, "_optimality_type", None)

    # ---- Copy ----

    def copy(self) -> "ShieldFactory":
        new = object.__new__(ShieldFactory)
        new._model_path = getattr(self, "_model_path", None)
        new._constants = getattr(self, "_constants", None)
        new._prism_program = self._prism_program
        new._model = self._model
        new._sparse = self._sparse
        new._property = self._property
        new._formula = getattr(self, "_formula", None)
        new._optimality_type = getattr(self, "_optimality_type", None)
        new._state_valuations = getattr(self, "_state_valuations", None)
        new._choice_labeling = getattr(self, "_choice_labeling", None)
        new._state_lookup = getattr(self, "_state_lookup", None)
        new._choice_data = getattr(self, "_choice_data", None)
        new._check_result = None
        new._unsafe_state_ids = None
        new.action_map = getattr(self, "action_map", None)
        return new

    def __copy__(self) -> "ShieldFactory":
        return self.copy()

    def __deepcopy__(self, memo: dict) -> "ShieldFactory":
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
        Run model checking (if needed) and compute the action mask.

        If model checking has not been performed yet, it is triggered here.
        Subsequent calls with the same property reuse the cached result.

        Parameters
        ----------
        config :
            Threshold, comparison mode, and optional post-selector / action
            lookup settings for the mask computation.

        Returns
        -------
        ShieldResult
            Per-state bitmasks encoding which actions are permitted.

        Raises
        ------
        RuntimeError
            If neither a formula nor a check result is available.
        """
        logger.trace(f"ShieldFactory.build called with {config=}")
        if self._check_result is None and self._choice_data is None:
            if self._model is None:
                raise RuntimeError(
                    "Cannot build shield: this factory has neither a Storm model nor "
                    "stored choice values."
                )
            if getattr(self, "_formula", None) is None:
                raise RuntimeError(
                    "Cannot build shield: no formula and no check result available. "
                    "Pass a property string."
                )
            self._run_model_check()
        return self.compute_mask(config)

    def compute_mask(self, config: ShieldConfig) -> Any:
        """
        Compute the action mask from an already-available check result.

        Unlike `build`, this method does *not* run model checking.
        A check result must be present.

        Parameters
        ----------
        config :
            Threshold, comparison mode, and optional post-selector / action
            lookup settings.

        Returns
        -------
        ShieldResult
            Per-state bitmasks encoding which actions are permitted.

        Raises
        ------
        RuntimeError
            If no check result is available.
        """
        logger.trace(f"ShieldFactory.compute_mask called with {config=}")
        if not self.is_sparse and self._model is None:
            raise RuntimeError("Cannot compute mask: Storm model not available.")
        if self._check_result is None and self._choice_data is None:
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
        if not program.has_undefined_constants:
            program = program.substitute_constants()
        return program

    def _build_model_from_prism(self) -> Any:
        logger.trace("ShieldFactory._build_model_from_prism called")
        if self.formula is not None:
            options = stormpy.BuilderOptions([self.formula.raw_formula])
            options.set_build_all_labels()
            options.set_build_all_reward_models()
        else:
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
            model = stormpy.build_symbolic_model(
                self._prism_program, properties, build_all_labels_and_rewards=True
            )
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
        """Extract and cache the `ChoiceData` snapshot from the check result.

        No-op if already extracted. Called lazily by `compute_action_mask`.
        """
        if self._choice_data is not None:
            logger.trace(
                "ShieldFactory._extract_choice_values: already extracted, skipping"
            )
            return
        if self._check_result is None:
            self._run_model_check()
        logger.trace(
            "ShieldFactory._extract_choice_values: extracting from check result"
        )
        is_minimize = (
            self.optimality_type is not None
            and self.optimality_type == stormpy.OptimizationDirection.Minimize
        )
        self._choice_data = ChoiceData.from_model(
            self._model,
            self._check_result,
            formula=self.formula,
            is_minimize=is_minimize,
        )

    @property
    def choice_data(self) -> Optional[ChoiceData]:
        """The stored `ChoiceData`, or ``None`` if no model checking has happened yet."""
        return self._choice_data

    _PROB_OPERATOR = re.compile(r"\b(Pmax|Pmin)\s*=\s*\?")

    def _qualitative_unsafe_property(self) -> str:
        """
        Derive the qualitative query whose truth vector identifies unsafe states.

        ``Pmax=? [phi]`` becomes ``Pmax>0 [phi]`` and ``Pmin=? [phi]`` becomes
        ``Pmin>=1 [phi]``; everything else (e.g. a ``<<coalition>>`` prefix) is kept.
        """
        prop = _sanitize_property(self._property)
        match = self._PROB_OPERATOR.search(prop)
        if match is None:
            raise ValueError(
                f"Cannot derive unsafe-state query from property {self._property!r}: "
                "expected a 'Pmax=?' or 'Pmin=?' operator"
            )
        is_minimize = self.optimality_type == stormpy.OptimizationDirection.Minimize
        operator = "Pmin>=1" if is_minimize else "Pmax>0"
        return prop[: match.start()] + operator + prop[match.end() :]

    def _get_unsafe_state_ids(self) -> frozenset:
        """
        Return (and cache) the ids of all unsafe states, i.e. states with risk 1.

        The property ``Pmax=? [phi]`` is checked as ``Pmax>0 [phi]`` (unsafe iff
        false) and ``Pmin=? [phi]`` as ``Pmin>=1 [phi]`` (unsafe iff true); a
        ``<<coalition>>`` prefix is kept. Storm's own qualitative results are
        used and no tolerance is added. Unbounded F/G are exact; bounded F is
        exact on the zero side and numeric on the one side for MDPs.
        """
        if getattr(self, "_unsafe_state_ids", None) is not None:
            return self._unsafe_state_ids
        if not self.is_sparse:
            raise NotImplementedError("unsafe states are not supported for engine='dd'")
        if self._property is None or getattr(self, "_formula", None) is None:
            raise RuntimeError("Cannot compute unsafe states: no property set.")
        if self._model is None:
            raise RuntimeError("Cannot compute unsafe states: Storm model not available.")

        is_minimize = self.optimality_type == stormpy.OptimizationDirection.Minimize
        qualitative = self._qualitative_unsafe_property()
        logger.trace(f"ShieldFactory._get_unsafe_state_ids: checking {qualitative!r}")
        formula = stormpy.parse_properties_for_prism_program(
            qualitative, self._prism_program
        )[0]
        result = stormpy.model_checking(self._model, formula)
        if result is None:
            raise NotImplementedError(
                f"Storm cannot check {qualitative!r} on this model "
                "(e.g. step-bounded properties on SMGs)"
            )
        truth = [bool(result.at(s)) for s in range(int(self._model.nr_states))]
        # Pmin>=1 holds exactly for unsafe states; Pmax>0 fails exactly for them.
        self._unsafe_state_ids = frozenset(
            s for s, holds in enumerate(truth) if holds == is_minimize
        )
        return self._unsafe_state_ids

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
        """``True`` when the factory operates in sparse-engine mode."""
        return self._sparse

    # ---- State lookup ----

    @property
    def state_lookup(self) -> StateValuationLookup:
        """Lazy-initialised bidirectional map between state ids and PRISM valuations."""
        if getattr(self, "_state_lookup", None) is None:
            self._state_lookup = StateValuationLookup.from_model(
                self._model,
                self._prism_program,
            )
        return self._state_lookup

    def get_choice_values_for_state(self, sid: int) -> np.ndarray:
        """
        Return the model-checked choice values for all actions in state *sid*.

        Parameters
        ----------
        sid :
            State index (0-based).

        Returns
        -------
        numpy.ndarray
            1-D array of floats, one entry per local action, in action order.
        """
        self._extract_choice_values()
        return self._choice_data.values_for_state(sid)

    def get_state_id(self, values: dict) -> int:
        """
        Look up the state id for a given PRISM variable valuation.

        Parameters
        ----------
        values :
            Mapping of PRISM variable name to value,
            e.g. ``{"x": 2, "y": 0, "done": False}``.

        Returns
        -------
        int
            The model state id corresponding to *values*.

        Raises
        ------
        KeyError
            If the valuation does not match any state.
        """
        return self.state_lookup.get_state_id(values)

    # ---- Dangerous / Critical / Safe / Unsafe helpers

    def _states_with_valuations(self, sids: list[int]) -> list[dict]:
        return [self.state_lookup.values_for_state_id(sid) for sid in sids]

    def unsafe_states(self) -> list[dict]:
        """
        Return the PRISM valuations of all unsafe states.

        A state is *unsafe* when it cannot satisfy the property at all
        (risk 1): ``Pmax(phi) = 0`` for maximising properties, ``Pmin = 1``
        for minimising ones. Unsafe takes precedence over the other classes.
        This does not depend on a `ShieldConfig`, so no mask is needed.

        The result comes from Storm's qualitative check (``Pmax>0`` /
        ``Pmin>=1``) without any added tolerance. Unbounded F/G are exact;
        bounded F on MDPs is exact on the zero side and numeric on the one
        side; bounded SMG properties are unsupported by Storm here.
        ``engine="dd"`` raises ``NotImplementedError``.

        Returns
        -------
        list of dict
            Each dict maps PRISM variable names to their values for one state.
        """
        return self._states_with_valuations(sorted(self._get_unsafe_state_ids()))

    def dangerous_states(self, *, mask: ShieldResult) -> list[dict]:
        """
        Return the PRISM valuations of all dangerous states in *mask*.

        A state is *dangerous* when no action meets the shield threshold and
        the fallback (best-value) action is used instead. Unsafe states are
        excluded.

        Parameters
        ----------
        mask :
            A `ShieldResult` produced by `build` or `compute_mask`.

        Returns
        -------
        list of dict
            Each dict maps PRISM variable names to their values for one state.
        """
        return self._states_with_valuations(mask.dangerous_states())

    def critical_states(self, *, mask: ShieldResult) -> list[dict]:
        """
        Return the PRISM valuations of all critical states in *mask*.

        A state is *critical* when at least one action is blocked by the
        threshold but at least one action still passes.

        Parameters
        ----------
        mask :
            A `ShieldResult` produced by `build` or `compute_mask`.

        Returns
        -------
        list of dict
            Each dict maps PRISM variable names to their values for one state.
        """
        return self._states_with_valuations(mask.critical_states())

    def safe_states(self, *, mask: ShieldResult) -> list[dict]:
        """
        Return the PRISM valuations of all safe states in *mask*.

        A state is *safe* when every action passes the shield threshold.

        Parameters
        ----------
        mask :
            A `ShieldResult` produced by `build` or `compute_mask`.

        Returns
        -------
        list of dict
            Each dict maps PRISM variable names to their values for one state.
        """
        return self._states_with_valuations(mask.safe_states())

    def states_by_class(self, *, mask: ShieldResult) -> dict[str, list[dict]]:
        """
        Partition all states into the four shield classes with their valuations.

        Parameters
        ----------
        mask :
            A `ShieldResult` produced by `build` or `compute_mask`.

        Returns
        -------
        dict
            Keys are ``"safe"``, ``"critical"``, ``"dangerous"``, ``"unsafe"``.
            Values are lists of valuation dicts (same format as `dangerous_states`).
        """
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

    def to_bitmask(self, mask: Any) -> dict[int, int]:
        """
        Convert *mask* to a ``{state_id: bitmask}`` dictionary.

        Parameters
        ----------
        mask :
            A `ShieldResult`.

        Returns
        -------
        dict
            Maps each state id to its packed integer bitmask.
        """
        return export_mod.state_id_to_bitmask(mask)

    def to_allowed_action_indices(self, mask: Any) -> dict[int, list[int]]:
        """
        Convert *mask* to a ``{state_id: [action_index, ...]}`` dictionary.

        Parameters
        ----------
        mask :
            A `ShieldResult`.

        Returns
        -------
        dict
            Maps each state id to the list of allowed local action indices.
        """
        return export_mod.state_id_to_allowed_action_indices(self._model, mask)

    def to_allowed_action_labels(self, mask: Any) -> dict[int, list]:
        """
        Convert *mask* to a ``{state_id: [frozenset(labels), ...]}`` dictionary.

        Parameters
        ----------
        mask :
            A `ShieldResult`.

        Returns
        -------
        dict
            Maps each state id to the list of label-sets for allowed actions.
        """
        return export_mod.state_id_to_allowed_action_labels(self._model, mask)

    def to_allowed_state_action_pairs(self, mask: Any) -> list[tuple[int, int]]:
        """
        Return all allowed ``(state_id, local_action_index)`` pairs.

        Parameters
        ----------
        mask :
            A `ShieldResult`.

        Returns
        -------
        list of tuple
            Each entry is ``(state_id, action_index)``.
        """
        return export_mod.allowed_state_action_pairs(self._model, mask)

    def to_valuation_bitmask(self, mask: Any) -> dict:
        """
        Convert *mask* to a ``{valuation_key: bitmask}`` dictionary.

        Parameters
        ----------
        mask :
            A `ShieldResult`.

        Returns
        -------
        dict
            Maps each state's `StateValuationKey` tuple to its packed bitmask.
        """
        return export_mod.valuation_to_bitmask(
            self._model,
            mask,
            self.state_lookup,
        )

    def to_valuation_allowed_action_labels(self, mask: Any) -> dict:
        """
        Convert *mask* to a ``{valuation_key: [frozenset(labels), ...]}`` dictionary.

        Parameters
        ----------
        mask :
            A `ShieldResult`.

        Returns
        -------
        dict
            Maps each state's `StateValuationKey` to the list of label-sets
            for allowed actions.
        """
        return export_mod.valuation_to_allowed_action_labels(
            self._model,
            mask,
            self.state_lookup,
        )

    def to_valuation_action_label_probability(self) -> dict:
        """
        Map every state to all its actions and their model-checked probabilities.

        Returns
        -------
        dict
            ``{valuation_key: {frozenset(action_labels): probability}}`` for
            *all* actions in each state, regardless of whether the shield
            allows or blocks them.
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
        """
        Return a human-readable summary of *mask* for debugging.

        Parameters
        ----------
        mask :
            A `ShieldResult`.
        max_states :
            Maximum number of states to include in the output (default 25).
        show_valuations :
            If ``True`` (default), show PRISM variable values next to each state.
        action_ref :
            How to identify actions: ``"labels"`` (default) shows
            ``frozenset(labels)``; ``"index"`` shows local action indices.

        Returns
        -------
        str
            Multi-line human-readable shield summary.
        """
        return export_mod.pretty(
            self._model,
            mask,
            self.state_lookup,
            max_states=max_states,
            show_valuations=show_valuations,
            action_ref=action_ref,
        )

    def to_storm_format(self, mask: Any) -> str:
        """
        Render *mask* as a Storm-compatible shield text.

        Parameters
        ----------
        mask :
            A `ShieldResult`.

        Returns
        -------
        str
            Multi-line string matching Storm's shield output format, with
            per-action choice values and valuation annotations.
        """
        return export_mod.storm_format(self, mask)
