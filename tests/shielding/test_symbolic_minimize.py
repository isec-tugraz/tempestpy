import pytest
from helpers.helper import get_example_path

from tempestpy.shielding import ShieldConfig, ShieldFactory

MAZE = get_example_path("mdp", "maze_2.nm")
STATE_VALUES = range(0, 13)

CASES = [
    pytest.param('Pmax=? [ F "goal" ]', 0.7, id="pmax-0.7"),
    pytest.param('Pmin=? [ F "goal" ]', 0.0, id="pmin-0.0"),
    pytest.param('Pmin=? [ F "goal" ]', 0.5, id="pmin-0.5"),
    pytest.param('Pmin=? [ F<=3 "goal" ]', 0.0, id="pmin-bounded-0.0"),
    pytest.param('Pmin=? [ F<=3 "goal" ]', 0.5, id="pmin-bounded-0.5"),
]


def build_masks(prop, threshold):
    config = ShieldConfig(threshold=threshold)
    sparse_factory = ShieldFactory(MAZE, property=prop)
    sparse_result = sparse_factory.build(config)
    dd_factory = ShieldFactory(MAZE, property=prop, engine="dd")
    dd_result = dd_factory.build(config)
    return sparse_factory, sparse_result, dd_factory, dd_result


def compared_masks(prop, threshold):
    sparse_factory, sparse_result, dd_factory, dd_result = build_masks(prop, threshold)
    masks = {}
    for s in STATE_VALUES:
        try:
            state_id = sparse_factory.state_lookup.get_state_id({"s": s})
        except KeyError:
            continue
        masks[s] = (
            dd_result.bitmask_by_state.query({"s": s}),
            sparse_result.bitmask_by_state[state_id],
        )
    del dd_factory
    return masks


@pytest.mark.parametrize("prop, threshold", CASES)
def test_dd_mask_matches_sparse_mask(prop, threshold):
    masks = compared_masks(prop, threshold)
    assert masks
    mismatches = {s: m for s, m in masks.items() if m[0] != m[1]}
    assert not mismatches, f"(dd, sparse) masks differ: {mismatches}"


@pytest.mark.parametrize("prop, threshold", CASES[1:])
def test_masks_are_not_trivial(prop, threshold):
    masks = compared_masks(prop, threshold)
    assert len({sparse for _, sparse in masks.values()}) > 1

