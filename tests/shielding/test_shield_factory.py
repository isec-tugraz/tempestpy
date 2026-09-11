import os

import stormpy
import stormpy.examples.files
import pytest

from tempestpy.shielding import ShieldFactory, ShieldConfig

slipgrid_sketch_path = os.path.join(
    stormpy.examples.files.testfile_dir, "mdp", "slipgrid_sketch.nm"
)


class TestConstructionModePath:
    def test_builds_model_and_prepares_property(self, slipgrid_path):
        factory = ShieldFactory(slipgrid_path, property='Pmax=? [F "goal"]')
        assert factory.model.model_type == stormpy.ModelType.MDP
        assert factory.formula is not None
        assert factory.prop == 'Pmax=? [F "goal"]'

    def test_without_property_has_no_formula(self, slipgrid_path):
        factory = ShieldFactory(slipgrid_path)
        assert factory.formula is None
        assert factory.prop is None

    def test_constants_define_undefined_prism_constant(self):
        # slipgrid_sketch.nm declares an undefined constant V (bound on 'south').
        factory = ShieldFactory(
            slipgrid_sketch_path,
            property='Pmax=? [F "goal"]',
            constants={"V": 5},
        )
        assert not factory.prism_program.has_undefined_constants

    def test_invalid_engine_raises_valueerror(self, slipgrid_path):
        with pytest.raises(ValueError):
            ShieldFactory(slipgrid_path, property='Pmax=? [F "goal"]', engine="bogus")


class TestConstructionModePreBuiltModel:
    def test_reuse_model_and_program(self, sparse_factory):
        factory2 = ShieldFactory(
            sparse_factory.model,
            property='Pmax=? [F "goal"]',
            prism_program=sparse_factory.prism_program,
        )
        assert factory2.model is sparse_factory.model
        assert factory2.formula is not None

    def test_missing_prism_program_raises_typeerror(self, sparse_factory):
        with pytest.raises(TypeError):
            ShieldFactory(sparse_factory.model, property='Pmax=? [F "goal"]')

    def test_prism_program_as_path_string(self, slipgrid_path, sparse_factory):
        factory2 = ShieldFactory(
            sparse_factory.model,
            property='Pmax=? [F "goal"]',
            prism_program=slipgrid_path,
        )
        assert factory2.formula is not None


class TestConstructionModePreComputedResult:
    def test_skips_model_checking(self, sparse_factory):
        mask = sparse_factory.build(ShieldConfig(threshold=0.9))
        check_result = sparse_factory._check_result

        factory3 = ShieldFactory(
            sparse_factory.model,
            prism_program=sparse_factory.prism_program,
            check_result=check_result,
        )
        assert factory3.formula is None
        result = factory3.compute_mask(ShieldConfig(threshold=0.9))
        assert result.bitmask_by_state == mask.bitmask_by_state


class TestAssertSupportedModel:
    def test_non_mdp_model_type_raises_typeerror(self, sparse_factory):
        with pytest.raises(TypeError):
            sparse_factory._assert_supported_model(None)

    def test_sparse_engine_requires_sparse_model(self, sparse_factory, dd_factory):
        # Exercised directly: constructing a sparse-engine ShieldFactory from a
        # symbolic model fails earlier (AttributeError) while wiring up
        # state_valuations/choice_labeling, before this check would even run.
        with pytest.raises(TypeError):
            sparse_factory._assert_supported_model(dd_factory.model)

    def test_dd_engine_requires_symbolic_model(self, sparse_factory, dd_factory):
        with pytest.raises(TypeError):
            dd_factory._assert_supported_model(sparse_factory.model)


class TestPropSetter:
    def test_setting_same_property_is_noop(self, sparse_factory):
        sparse_factory.build(ShieldConfig(threshold=0.9))
        cached = sparse_factory._check_result
        sparse_factory.prop = sparse_factory.prop
        assert sparse_factory._check_result is cached

    def test_changing_property_invalidates_cache(self, sparse_factory):
        sparse_factory.build(ShieldConfig(threshold=0.9))
        assert sparse_factory._check_result is not None
        # Reuse the same "goal" label: the model was built to only retain
        # labels referenced by the *initial* formula, so switching to an
        # unrelated label ("target") would fail model checking entirely.
        sparse_factory.prop = 'Pmin=? [F "goal"]'
        assert sparse_factory._check_result is None
        assert sparse_factory._choice_values is None
        assert sparse_factory._choice_offsets is None
        assert sparse_factory._choice_counts is None


class TestCopy:
    def test_copy_produces_independent_instance(self, sparse_factory):
        sparse_factory.build(ShieldConfig(threshold=0.9))
        clone = sparse_factory.copy()
        assert clone is not sparse_factory
        assert clone.model is sparse_factory.model
        assert clone.prop == sparse_factory.prop
        clone.prop = 'Pmax=? [F "target"]'
        assert sparse_factory.prop == 'Pmax=? [F "goal"]'

    def test_dunder_copy_and_deepcopy(self, sparse_factory):
        import copy

        c1 = copy.copy(sparse_factory)
        c2 = copy.deepcopy(sparse_factory)
        assert c1 is not sparse_factory
        assert c2 is not sparse_factory

    def test_with_prop_returns_new_factory_with_fresh_mask(self, sparse_factory):
        base_mask = sparse_factory.build(ShieldConfig(threshold=0.9))
        other = sparse_factory.with_prop('Pmin=? [F "goal"]')
        assert other is not sparse_factory
        assert other.prop == 'Pmin=? [F "goal"]'
        assert sparse_factory.prop == 'Pmax=? [F "goal"]'
        other_mask = other.build(ShieldConfig(threshold=0.9))
        assert isinstance(other_mask.bitmask_by_state, list)
        assert other_mask.bitmask_by_state != base_mask.bitmask_by_state


class TestBuildAndComputeMaskErrors:
    def test_build_without_formula_or_check_result_raises_runtimeerror(self, slipgrid_path):
        factory = ShieldFactory(slipgrid_path)
        with pytest.raises(RuntimeError):
            factory.build(ShieldConfig(threshold=0.9))

    def test_compute_mask_without_model_raises_runtimeerror(self, sparse_factory):
        sparse_factory._model = None
        with pytest.raises(RuntimeError):
            sparse_factory.compute_mask(ShieldConfig(threshold=0.9))

    def test_compute_mask_without_check_result_raises_runtimeerror(self, slipgrid_path):
        factory = ShieldFactory(slipgrid_path, property='Pmax=? [F "goal"]')
        with pytest.raises(RuntimeError):
            factory.compute_mask(ShieldConfig(threshold=0.9))


class TestStateAndChoiceAccessors:
    def test_get_choice_values_for_state(self, sparse_factory):
        sparse_factory.build(ShieldConfig(threshold=0.9))
        values = sparse_factory.get_choice_values_for_state(0)
        assert len(values) == len(list(sparse_factory.model.states)[0].actions)

    def test_get_state_id_round_trips(self, sparse_factory):
        values = sparse_factory.state_lookup.values_for_state_id(3)
        assert sparse_factory.get_state_id(values) == 3


class TestStateClassification:
    def test_dangerous_critical_safe_states(self, smg_factory):
        mask = smg_factory.build(ShieldConfig(threshold=0.5))
        dangerous = smg_factory.dangerous_states(mask=mask)
        safe = smg_factory.safe_states(mask=mask)
        critical = smg_factory.critical_states(mask=mask)
        assert {tuple(sorted(v.items())) for v in dangerous} == {
            (("s", 1),), (("s", 3),)
        }
        assert {tuple(sorted(v.items())) for v in safe} == {
            (("s", 0),), (("s", 2),)
        }
        assert critical == []

    def test_states_by_class(self, smg_factory):
        mask = smg_factory.build(ShieldConfig(threshold=0.5))
        classes = smg_factory.states_by_class(mask=mask)
        assert set(classes.keys()) == {"safe", "critical", "dangerous"}
        assert len(classes["safe"]) == 2
        assert len(classes["dangerous"]) == 2


class TestExportDelegation:
    def test_to_bitmask_delegates(self, sparse_factory):
        mask = sparse_factory.build(ShieldConfig(threshold=0.9))
        assert sparse_factory.to_bitmask(mask) == mask.to_bitmask_dict()

    def test_pretty_mask_smoke(self, sparse_factory):
        mask = sparse_factory.build(ShieldConfig(threshold=0.9))
        text = sparse_factory.pretty_mask(mask, max_states=2)
        assert "ShieldResult" in text

    def test_to_storm_format_smoke(self, sparse_factory):
        mask = sparse_factory.build(ShieldConfig(threshold=0.9))
        text = sparse_factory.to_storm_format(mask)
        assert "Pre-Safety-Shield" in text
