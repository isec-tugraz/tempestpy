import os
import pickle

import pytest
import stormpy.examples.files as files

from tempestpy.shielding import ShieldFactory
from tempestpy.shielding.masking import ShieldConfig

SMG = os.path.join(os.path.dirname(files.__file__), "files", "smg", "example_smg.nm")
CLASSES = {"safe", "critical", "dangerous", "unsafe"}

MDP_CASES = [
    ('Pmax=? [F "goal"]', 0),
    ('Pmax=? [G !"goal"]', 1),
    ('Pmin=? [F "goal"]', 1),
    ('Pmax=? [F<=3 "goal"]', 6),
]
CONFIGS = [
    ShieldConfig(threshold=0.0),
    ShieldConfig(threshold=0.5),
    ShieldConfig(threshold=1.0),
    ShieldConfig(threshold=0.0, comparison="relative"),
    ShieldConfig(threshold=1.0, comparison="relative"),
]


def _assert_partition(result):
    classes = result.states_by_class()
    assert set(classes) == CLASSES
    all_ids = [sid for ids in classes.values() for sid in ids]
    assert sorted(all_ids) == list(range(len(result)))


@pytest.mark.parametrize("prop,nr_unsafe", MDP_CASES)
@pytest.mark.parametrize("config", CONFIGS)
def test_mdp_classes_partition_states(prop, nr_unsafe, config):
    factory = ShieldFactory(files.prism_mdp_slipgrid, property=prop)
    result = factory.build(config)
    _assert_partition(result)
    assert len(result.states_by_class()["unsafe"]) == nr_unsafe
    assert len(factory.unsafe_states()) == nr_unsafe


def test_unsafe_not_counted_as_safe_with_zero_threshold():
    factory = ShieldFactory(files.prism_mdp_slipgrid, property='Pmax=? [F<=3 "goal"]')
    result = factory.build(ShieldConfig(threshold=0.0))
    unsafe = set(result.unsafe_states())
    assert unsafe
    assert unsafe.isdisjoint(result.safe_states())
    assert unsafe.isdisjoint(result.critical_states())
    assert unsafe.isdisjoint(result.dangerous_states())


def test_unsafe_not_counted_as_safe_in_relative_mode_with_best_value_zero():
    factory = ShieldFactory(files.prism_mdp_slipgrid, property='Pmax=? [F<=3 "goal"]')
    result = factory.build(ShieldConfig(threshold=1.0, comparison="relative"))
    unsafe = set(result.unsafe_states())
    assert all(result.best_value_by_state[sid] == 0.0 for sid in unsafe)
    assert unsafe.isdisjoint(result.safe_states())
    _assert_partition(result)


@pytest.mark.parametrize("config", CONFIGS)
def test_smg_classes_partition_states(config):
    factory = ShieldFactory(SMG, property='<<maxP>> Pmax=? [F "goal"]')
    result = factory.build(config)
    _assert_partition(result)
    classes = result.states_by_class()
    assert classes["unsafe"] == [3]
    assert len(factory.states_by_class(mask=result)) == 4


def test_states_by_class_has_unsafe_valuations():
    factory = ShieldFactory(SMG, property='<<maxP>> Pmax=? [F "goal"]')
    result = factory.build(ShieldConfig(threshold=0.0))
    assert factory.states_by_class(mask=result)["unsafe"] == [{"s": 3}]
    assert factory.unsafe_states() == [{"s": 3}]


def test_unsafe_cache_reset_on_property_change_and_copy():
    factory = ShieldFactory(files.prism_mdp_slipgrid, property='Pmax=? [F "goal"]')
    assert factory.unsafe_states() == []
    factory.prop = 'Pmax=? [F<=3 "goal"]'
    assert len(factory.unsafe_states()) == 6
    assert factory.copy()._unsafe_state_ids is None


@pytest.mark.parametrize("slim", [True, False])
def test_unsafe_ids_survive_serialization(tmp_path, slim):
    factory = ShieldFactory(files.prism_mdp_slipgrid, property='Pmax=? [F<=3 "goal"]')
    factory.build(ShieldConfig(threshold=0.5))
    expected = factory._get_unsafe_state_ids()
    path = str(tmp_path / "factory.pkl")
    factory.save(path, slim=slim)
    assert ShieldFactory.load(path)._unsafe_state_ids == expected


def test_dd_engine_not_supported():
    factory = ShieldFactory.__new__(ShieldFactory)
    factory._sparse = False
    factory._unsafe_state_ids = None
    with pytest.raises(NotImplementedError):
        factory.unsafe_states()

