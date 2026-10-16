import pickle

import numpy as np
import pytest

from tempestpy.shielding import ShieldConfig, ShieldFactory
from tempestpy.shielding.choice_data import ChoiceData
from tempestpy.shielding.masking import compute_explicit_mask
from tempestpy.shielding.serialization import get_state, set_state


def _data(**overrides):
    kwargs = dict(
        values=np.array([0.1, 0.9, 0.5, 1.0]),
        offsets=np.array([0, 2, 3, 4]),
    )
    kwargs.update(overrides)
    return ChoiceData(**kwargs)


class TestChoiceDataStructure:
    def test_nr_states_and_counts(self):
        data = _data()
        assert data.nr_states == 3
        assert list(data.counts) == [2, 1, 1]

    def test_values_for_state(self):
        data = _data()
        assert list(data.values_for_state(0)) == [0.1, 0.9]
        assert list(data.values_for_state(2)) == [1.0]

    def test_frozen(self):
        with pytest.raises(Exception):
            _data().is_minimize = True

    def test_labels_for_choice_requires_labeling(self):
        with pytest.raises(RuntimeError):
            _data().labels_for_choice(0, 0)

    def test_labels_for_choice(self):
        data = _data(
            label_vocab=(frozenset({"a"}), frozenset({"b"})),
            label_ids=np.array([0, 1, 1, 0], dtype=np.int32),
        )
        assert data.labels_for_choice(0, 1) == frozenset({"b"})
        assert data.labels_for_choice(2, 0) == frozenset({"a"})


class TestChoiceDataFromModel:
    def test_matches_check_result(self, sparse_factory):
        sparse_factory.build(ShieldConfig(threshold=0.9))
        data = sparse_factory._choice_data
        model = sparse_factory.model
        assert data.nr_states == int(model.nr_states)
        assert data.values.size == len(sparse_factory._check_result.get_choice_values())
        assert data.is_minimize is False
        assert data.coalition_states is None
        for state in model.states:
            assert data.counts[state.id] == len(state.actions)

    def test_minimize_direction(self, slipgrid_path):
        factory = ShieldFactory(slipgrid_path, property='Pmin=? [F "goal"]')
        factory.build(ShieldConfig(threshold=0.9))
        assert factory._choice_data.is_minimize is True

    def test_action_labels_are_recorded(self, sparse_factory):
        sparse_factory.build(ShieldConfig(threshold=0.9))
        data = sparse_factory._choice_data
        assert data.label_ids is not None
        assert data.label_ids.size == data.values.size
        for state in sparse_factory.model.states:
            for i, action in enumerate(state.actions):
                assert data.labels_for_choice(state.id, i) == frozenset(action.labels)

    def test_smg_records_coalition(self, smg_factory):
        smg_factory.build(ShieldConfig(threshold=0.5))
        coalition = smg_factory._choice_data.coalition_states
        assert coalition is not None
        assert coalition < frozenset(range(smg_factory._choice_data.nr_states))


class TestChoiceDataMasks:
    def test_mask_is_pure_function_of_choice_data(self, sparse_factory):
        config = ShieldConfig(threshold=0.9)
        expected = sparse_factory.build(config)
        recomputed = compute_explicit_mask(
            sparse_factory._choice_data, config, expected.unsafe_state_ids
        )
        assert recomputed.bitmask_by_state == expected.bitmask_by_state
        assert recomputed.fallback_by_state == expected.fallback_by_state
        assert recomputed.nr_actions_by_state == expected.nr_actions_by_state

    def test_synthetic_absolute_mask(self):
        result = compute_explicit_mask(_data(), ShieldConfig(threshold=0.5))
        assert result.bitmask_by_state == [0b10, 0b1, 0b1]
        assert result.fallback_by_state == [0, 0, 0]

    def test_synthetic_minimize_mask(self):
        result = compute_explicit_mask(
            _data(is_minimize=True), ShieldConfig(threshold=0.5)
        )
        assert result.bitmask_by_state == [0b01, 0b1, 0b0]
        assert result.fallback_by_state[2] == 0b1

    def test_unsafe_ids_are_supplied_by_caller(self):
        data = _data()
        assert compute_explicit_mask(data, ShieldConfig()).unsafe_state_ids == frozenset()
        result = compute_explicit_mask(data, ShieldConfig(), frozenset({2}))
        assert result.is_unsafe(2)
        assert not result.is_unsafe(0)

    def test_coalition_excludes_other_states(self):
        data = _data(coalition_states=frozenset({0}))
        result = compute_explicit_mask(data, ShieldConfig(threshold=0.5))
        assert result.bitmask_by_state == [0b10, 0, 0]


class TestChoiceDataPersistence:
    def test_pickle_round_trip(self, sparse_factory):
        sparse_factory.build(ShieldConfig(threshold=0.9))
        data = sparse_factory._choice_data
        restored = pickle.loads(pickle.dumps(data))
        np.testing.assert_array_equal(restored.values, data.values)
        np.testing.assert_array_equal(restored.offsets, data.offsets)
        np.testing.assert_array_equal(restored.label_ids, data.label_ids)
        assert restored.label_vocab == data.label_vocab
        assert restored.is_minimize == data.is_minimize

    @pytest.mark.parametrize("slim", [True, False])
    def test_factory_save_load_preserves_choice_data(self, sparse_factory, tmp_path, slim):
        sparse_factory.build(ShieldConfig(threshold=0.9))
        path = str(tmp_path / "factory.pkl")
        sparse_factory.save(path, slim=slim)
        loaded = ShieldFactory.load(path)
        original = sparse_factory._choice_data
        np.testing.assert_array_equal(loaded._choice_data.values, original.values)
        np.testing.assert_array_equal(loaded._choice_data.offsets, original.offsets)
        assert loaded._choice_data.label_vocab == original.label_vocab

    def test_slim_state_has_no_model(self, sparse_factory):
        sparse_factory.build(ShieldConfig(threshold=0.9))
        state = get_state(sparse_factory, slim=True)
        assert state["slim"] is True
        assert "drn_bytes" not in state
        assert isinstance(state["choice_data"], ChoiceData)
        assert "drn_bytes" in get_state(sparse_factory, slim=False)

    def test_slim_is_smaller_than_full(self, sparse_factory, tmp_path):
        sparse_factory.build(ShieldConfig(threshold=0.9))
        slim = tmp_path / "slim.pkl"
        full = tmp_path / "full.pkl"
        sparse_factory.save(str(slim), slim=True)
        sparse_factory.save(str(full), slim=False)
        assert slim.stat().st_size < full.stat().st_size

    def test_unknown_format_version_rejected(self, sparse_factory):
        sparse_factory.build(ShieldConfig(threshold=0.9))
        state = get_state(sparse_factory, slim=True)
        state["format_version"] = 1
        with pytest.raises(ValueError, match="format version"):
            set_state(ShieldFactory.__new__(ShieldFactory), state)
