import os

import pytest
import stormpy
import stormpy.examples.files

from tempestpy.shielding import ShieldFactory, ShieldConfig
from tempestpy.shielding.shield_result import ExplicitShieldResult

smg_example_path = os.path.join(
    stormpy.examples.files.testfile_dir, "smg", "example_smg.nm"
)

slipgrid_goal_property = 'Pmax=? [F "goal"]'
smg_goal_property = '<<maxP>> Pmax=? [F "goal"]'


@pytest.fixture
def slipgrid_path():
    return stormpy.examples.files.prism_mdp_slipgrid


@pytest.fixture
def sparse_factory(slipgrid_path):
    return ShieldFactory(slipgrid_path, property=slipgrid_goal_property)


@pytest.fixture
def dd_factory(slipgrid_path):
    return ShieldFactory(slipgrid_path, property=slipgrid_goal_property, engine="dd")


@pytest.fixture
def bounded_max_factory(slipgrid_path):
    """Bounded reachability gives non-trivial (0,1) probabilities, unlike
    the unbounded 'goal' property which is 1.0 almost everywhere."""
    factory = ShieldFactory(slipgrid_path, property='Pmax=? [F<=3 "goal"]')
    factory.build(ShieldConfig(threshold=1.0))
    return factory


@pytest.fixture
def bounded_min_factory(slipgrid_path):
    factory = ShieldFactory(slipgrid_path, property='Pmin=? [F<=3 "goal"]')
    factory.build(ShieldConfig(threshold=1.0))
    return factory


@pytest.fixture
def smg_factory():
    return ShieldFactory(smg_example_path, property=smg_goal_property)


@pytest.fixture
def sparse_mask(sparse_factory):
    return sparse_factory.build(ShieldConfig(threshold=0.9))


@pytest.fixture
def smg_mask(smg_factory):
    return smg_factory.build(ShieldConfig(threshold=0.5))


def make_explicit_result(
    *,
    bitmask_by_state,
    fallback_by_state=None,
    nr_actions_by_state=None,
    config=None,
    coalition_states=None,
):
    """Build a synthetic ExplicitShieldResult without running model checking."""
    n = len(bitmask_by_state)
    if fallback_by_state is None:
        fallback_by_state = [0] * n
    if nr_actions_by_state is None:
        nr_actions_by_state = [bits.bit_length() or 1 for bits in bitmask_by_state]
    if config is None:
        config = ShieldConfig()
    return ExplicitShieldResult(
        config=config,
        bitmask_by_state=list(bitmask_by_state),
        fallback_by_state=list(fallback_by_state),
        nr_actions_by_state=list(nr_actions_by_state),
        coalition_states=coalition_states,
    )
