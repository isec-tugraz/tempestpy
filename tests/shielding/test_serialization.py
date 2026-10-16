import pickle

import pytest

from tempestpy.shielding import ShieldFactory, ShieldConfig
from tempestpy.shielding.serialization import model_to_drn_bytes, model_from_drn_bytes


class TestDrnRoundTrip:
    def test_model_round_trips_through_drn_bytes(self, sparse_factory):
        drn_bytes = model_to_drn_bytes(sparse_factory.model)
        assert isinstance(drn_bytes, bytes)
        assert len(drn_bytes) > 0

        restored = model_from_drn_bytes(drn_bytes)
        assert restored.nr_states == sparse_factory.model.nr_states
        assert restored.model_type == sparse_factory.model.model_type


class TestSlimSaveLoad:
    def test_slim_round_trip_preserves_mask_and_lookup(self, sparse_factory, tmp_path):
        mask = sparse_factory.build(ShieldConfig(threshold=0.9))
        path = str(tmp_path / "shield_slim.pkl")
        sparse_factory.save(path, slim=True)

        loaded = ShieldFactory.load(path)
        assert loaded._model is None
        assert loaded._choice_data is not None

        loaded_values = loaded.get_choice_values_for_state(3)
        original_values = sparse_factory.get_choice_values_for_state(3)
        assert list(loaded_values) == list(original_values)

        assert loaded.get_state_id(sparse_factory.state_lookup.values_for_state_id(5)) == 5

    def test_slim_load_computes_masks_but_fails_on_model_dependent_calls(
        self, sparse_factory, tmp_path
    ):
        expected = sparse_factory.build(ShieldConfig(threshold=0.9))
        path = str(tmp_path / "shield_slim.pkl")
        sparse_factory.save(path, slim=True)

        loaded = ShieldFactory.load(path)
        # masks are computed from the stored ChoiceData, no Storm model needed
        mask = loaded.compute_mask(ShieldConfig(threshold=0.9))
        assert mask.bitmask_by_state == expected.bitmask_by_state
        assert mask.fallback_by_state == expected.fallback_by_state
        # re-checking needs the PRISM program
        with pytest.raises(RuntimeError):
            loaded.prop = 'Pmin=? [F "goal"]'


class TestFullSaveLoad:
    def test_full_round_trip_restores_model_and_rebuilds(self, sparse_factory, tmp_path):
        sparse_factory.build(ShieldConfig(threshold=0.9))
        path = str(tmp_path / "shield_full.pkl")
        sparse_factory.save(path, slim=False)

        loaded = ShieldFactory.load(path)
        assert loaded._model is not None
        assert loaded._prism_program is not None
        assert loaded.prop == sparse_factory.prop

        new_mask = loaded.build(ShieldConfig(threshold=0.9))
        original_mask = sparse_factory.build(ShieldConfig(threshold=0.9))
        assert new_mask.bitmask_by_state == original_mask.bitmask_by_state


class TestPickleProtocolDirectly:
    def test_pickle_dumps_loads_round_trip(self, sparse_factory):
        sparse_factory.build(ShieldConfig(threshold=0.9))
        sparse_factory._pickle_slim = True
        try:
            blob = pickle.dumps(sparse_factory)
        finally:
            del sparse_factory._pickle_slim

        restored = pickle.loads(blob)
        assert restored.prop == sparse_factory.prop
        assert list(restored.get_choice_values_for_state(0)) == list(
            sparse_factory.get_choice_values_for_state(0)
        )
