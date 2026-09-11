import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from tempestpy.shielding import ShieldConfig
from tempestpy.shielding.action_lookup import ActionLookup
from tempestpy.shielding.wrappers import (
    PreShieldWrapper,
    PostShieldWrapper,
    make_best_value_selector,
    make_random_safe_selector,
    make_nearest_action_selector,
)

from conftest import make_explicit_result

# Global action indices used throughout: 0=north, 1=south, 2=east, 3=west
ACTION_LOOKUP = ActionLookup.from_dict({"north": 0, "south": 1, "east": 2, "west": 3})


class GridEnv(gym.Env):
    """Minimal deterministic Discrete-action grid env mirroring slipgrid's
    (x, y) state layout, just enough to drive the wrappers under test."""

    def __init__(self, n=4, m=4):
        super().__init__()
        self.n = n
        self.m = m
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=1, high=max(n, m), shape=(2,), dtype=int)
        self.x = 1
        self.y = 1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.x, self.y = 1, 1
        return np.array([self.x, self.y]), {}

    def step(self, action):
        if action == 0 and self.x > 1:
            self.x -= 1
        elif action == 1 and self.x < self.n:
            self.x += 1
        elif action == 2 and self.y > 1:
            self.y -= 1
        elif action == 3 and self.y < self.m:
            self.y += 1
        terminated = self.x == self.n and self.y == self.m
        return np.array([self.x, self.y]), 0.0, terminated, False, {}


def obs_to_values(obs, info=None):
    return {"x": int(obs[0]), "y": int(obs[1])}


@pytest.fixture
def shield_config():
    return ShieldConfig(threshold=0.5, action_lookup=ACTION_LOOKUP)


@pytest.fixture
def built_factory(bounded_max_factory, shield_config):
    bounded_max_factory.build(shield_config)
    return bounded_max_factory


class TestPreShieldWrapper:
    def test_requires_discrete_action_space(self, built_factory, shield_config):
        env = GridEnv()
        env.action_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        with pytest.raises(TypeError):
            PreShieldWrapper(env, built_factory, shield_config, obs_to_values=obs_to_values)

    def test_reset_injects_mask_info(self, built_factory, shield_config):
        wrapper = PreShieldWrapper(
            GridEnv(), built_factory, shield_config, obs_to_values=obs_to_values
        )
        obs, info = wrapper.reset()
        assert info["shield_state_id"] == 0
        # only 'south' (global index 1) is allowed/fallback at state (x=1, y=1)
        assert info["shield_bitmask"] == 0b0010
        np.testing.assert_array_equal(
            info["action_mask"], np.array([False, True, False, False])
        )

    def test_action_masks_before_reset_raises_runtimeerror(self, built_factory, shield_config):
        wrapper = PreShieldWrapper(
            GridEnv(), built_factory, shield_config, obs_to_values=obs_to_values
        )
        with pytest.raises(RuntimeError):
            wrapper.action_masks()

    def test_action_masks_matches_last_reset(self, built_factory, shield_config):
        wrapper = PreShieldWrapper(
            GridEnv(), built_factory, shield_config, obs_to_values=obs_to_values
        )
        obs, info = wrapper.reset()
        np.testing.assert_array_equal(wrapper.action_masks(), info["action_mask"])

    def test_step_updates_mask(self, built_factory, shield_config):
        wrapper = PreShieldWrapper(
            GridEnv(), built_factory, shield_config, obs_to_values=obs_to_values
        )
        wrapper.reset()
        obs, reward, terminated, truncated, info = wrapper.step(1)  # move south
        assert info["shield_state_id"] == built_factory.get_state_id({"x": 2, "y": 1})

    def test_rebuild_resets_last_mask(self, built_factory, shield_config):
        wrapper = PreShieldWrapper(
            GridEnv(), built_factory, shield_config, obs_to_values=obs_to_values
        )
        wrapper.reset()
        wrapper.rebuild(ShieldConfig(threshold=0.0, action_lookup=ACTION_LOOKUP))
        with pytest.raises(RuntimeError):
            wrapper.action_masks()


class TestPostShieldWrapper:
    def test_requires_discrete_action_space(self, built_factory):
        env = GridEnv()
        env.action_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        config = ShieldConfig(
            threshold=0.5, action_lookup=ACTION_LOOKUP, post_selector=lambda s, a: a
        )
        with pytest.raises(TypeError):
            PostShieldWrapper(env, built_factory, config, obs_to_values=obs_to_values)

    def test_requires_post_selector(self, built_factory, shield_config):
        with pytest.raises(ValueError):
            PostShieldWrapper(
                GridEnv(), built_factory, shield_config, obs_to_values=obs_to_values
            )

    def test_allowed_action_passes_through(self, built_factory):
        config = ShieldConfig(
            threshold=0.5,
            action_lookup=ACTION_LOOKUP,
            post_selector=lambda sid, blocked: 1,  # always fall back to 'south'
        )
        wrapper = PostShieldWrapper(
            GridEnv(), built_factory, config, obs_to_values=obs_to_values
        )
        wrapper.reset()
        # 'south' (1) is allowed at state (x=1, y=1) -> passes through unmodified
        obs, reward, terminated, truncated, info = wrapper.step(1)
        assert info["shield_safe_action"] == 1
        assert info["shield_corrected"] is False

    def test_blocked_action_replaced_by_selector(self, built_factory):
        config = ShieldConfig(
            threshold=0.5,
            action_lookup=ACTION_LOOKUP,
            post_selector=lambda sid, blocked: 1,
        )
        wrapper = PostShieldWrapper(
            GridEnv(), built_factory, config, obs_to_values=obs_to_values
        )
        wrapper.reset()
        # 'north' (0) is blocked at state (x=1, y=1) -> selector replaces it
        obs, reward, terminated, truncated, info = wrapper.step(0)
        assert info["shield_safe_action"] == 1
        assert info["shield_corrected"] is True

    def test_step_before_reset_raises_runtimeerror(self, built_factory):
        config = ShieldConfig(
            threshold=0.5, action_lookup=ACTION_LOOKUP, post_selector=lambda s, a: a
        )
        wrapper = PostShieldWrapper(
            GridEnv(), built_factory, config, obs_to_values=obs_to_values
        )
        with pytest.raises(RuntimeError):
            wrapper.step(0)

    def test_rebuild_requires_post_selector(self, built_factory):
        config = ShieldConfig(
            threshold=0.5, action_lookup=ACTION_LOOKUP, post_selector=lambda s, a: a
        )
        wrapper = PostShieldWrapper(
            GridEnv(), built_factory, config, obs_to_values=obs_to_values
        )
        with pytest.raises(ValueError):
            wrapper.rebuild(ShieldConfig(threshold=0.5, action_lookup=ACTION_LOOKUP))


class TestPostSelectorFactories:
    def test_make_best_value_selector_maximize(self, bounded_max_factory, shield_config):
        result = bounded_max_factory.build(shield_config)
        selector = make_best_value_selector(bounded_max_factory, result, is_minimize=False)
        # state 10 pvals = [0.144, 0.144, 0.648] -> argmax is local index 2
        assert selector(10, blocked_action=0) == 2

    def test_make_best_value_selector_minimize(self, bounded_min_factory):
        config = ShieldConfig(threshold=0.1)
        result = bounded_min_factory.build(config)
        selector = make_best_value_selector(bounded_min_factory, result, is_minimize=True)
        # state 13 pvals = [0.0, 0.0, 0.6] -> argmin is local index 0
        assert selector(13, blocked_action=2) == 0

    def test_make_random_safe_selector_picks_allowed_action(self):
        result = make_explicit_result(bitmask_by_state=[0b0110])
        rng = np.random.default_rng(42)
        selector = make_random_safe_selector(result, rng=rng)
        for _ in range(20):
            assert selector(0, blocked_action=0) in (1, 2)

    def test_make_random_safe_selector_uses_fallback_when_blocked(self):
        result = make_explicit_result(
            bitmask_by_state=[0b0000], fallback_by_state=[0b0001]
        )
        selector = make_random_safe_selector(result, rng=np.random.default_rng(0))
        assert selector(0, blocked_action=3) == 0

    def test_make_random_safe_selector_falls_back_to_blocked_when_nothing_allowed(self):
        result = make_explicit_result(bitmask_by_state=[0b0000], fallback_by_state=[0b0000])
        selector = make_random_safe_selector(result, rng=np.random.default_rng(0))
        assert selector(0, blocked_action=3) == 3

    def test_make_nearest_action_selector_picks_closest_index(self):
        result = make_explicit_result(bitmask_by_state=[0b1001])  # actions 0 and 3 allowed
        selector = make_nearest_action_selector(result)
        assert selector(0, blocked_action=1) == 0
        assert selector(0, blocked_action=2) == 3

    def test_make_nearest_action_selector_falls_back_to_blocked_when_nothing_allowed(self):
        result = make_explicit_result(bitmask_by_state=[0b0000], fallback_by_state=[0b0000])
        selector = make_nearest_action_selector(result)
        assert selector(0, blocked_action=1) == 1
