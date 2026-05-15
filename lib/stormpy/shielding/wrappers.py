from __future__ import annotations

try:
    import gymnasium as gym
except ImportError as _exc:
    raise ImportError(
        "stormpy.shielding.wrappers requires gymnasium. "
        "Install it with:  pip install gymnasium"
    ) from _exc

from typing import Any, Callable, Mapping, Optional

import numpy as np

from .shield_factory import ShieldFactory
from .masking import ShieldConfig, ShieldResult

ObsToValuesFn = Callable[[Any, Optional[dict]], Mapping[str, Any]]


def _bitmask_to_bool_array(bits: int, n_actions: int) -> np.ndarray:
    return np.array([(bits >> i) & 1 for i in range(n_actions)], dtype=bool)


# ---------------------------------------------------------------------------
# PreShieldWrapper
# ---------------------------------------------------------------------------

class PreShieldWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that adds shield-based action masking.

    Requires:
      - env.action_space is gym.spaces.Discrete
      - obs_to_values maps obs -> {prism_var: value} for every PRISM state variable

    After reset() / step(), info contains:
      "action_mask"     -> np.ndarray[bool], shape (n_actions,)
      "shield_state_id" -> int
      "shield_bitmask"  -> int

    action_masks() returns the current mask for SB3 MaskablePPO.
    """

    def __init__(
        self,
        env: gym.Env,
        factory: ShieldFactory,
        config: ShieldConfig,
        *,
        obs_to_values: ObsToValuesFn,
    ) -> None:
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("PreShieldWrapper requires a Discrete action space.")

        self._factory = factory
        self._config = config
        self._obs_to_values = obs_to_values
        self._n_actions = int(env.action_space.n)

        self._result: ShieldResult = factory.build(config)
        self._last_mask: Optional[np.ndarray] = None

    def _query(self, obs: Any, info: Optional[dict] = None) -> tuple[int, np.ndarray, int]:
        values = self._obs_to_values(obs, info)
        sid = self._factory.state_lookup.get_state_id(values)
        bits = self._result.query_mask(sid)
        mask = _bitmask_to_bool_array(bits, self._n_actions)
        return sid, mask, bits

    def _augment(self, info: Optional[dict], sid: int, mask: np.ndarray, bits: int) -> dict:
        out = {} if info is None else dict(info)
        out["action_mask"] = mask
        out["shield_state_id"] = sid
        out["shield_bitmask"] = bits
        return out

    def action_masks(self) -> np.ndarray:
        """SB3 MaskablePPO hook — returns current boolean action mask."""
        if self._last_mask is None:
            raise RuntimeError("action_masks() called before reset().")
        return self._last_mask

    def rebuild(self, config: ShieldConfig) -> None:
        """Recompute the shield with a new config (e.g. different threshold)."""
        self._config = config
        self._result = self._factory.build(config)
        self._last_mask = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        sid, mask, bits = self._query(obs, info)
        self._last_mask = mask
        return obs, self._augment(info, sid, mask, bits)

    def step(self, action: int):
        if self._last_mask is None:
            raise RuntimeError("step() called before reset().")
        obs, reward, terminated, truncated, info = self.env.step(action)
        sid, mask, bits = self._query(obs, info)
        self._last_mask = mask
        return obs, reward, terminated, truncated, self._augment(info, sid, mask, bits)


# ---------------------------------------------------------------------------
# post_selector factory functions
# ---------------------------------------------------------------------------

def make_best_value_selector(
    factory: ShieldFactory,
    result: ShieldResult,
    *,
    is_minimize: bool = False,
) -> Callable[[int, int], int]:
    """
    Replace a blocked action with the one that has the best model-checked value.

    Closes over factory to retrieve per-state choice values at call time.
    Works in both normal and dangerous (bitmask == 0) states; in dangerous
    states the fallback action (argmax/argmin) is returned.

    Example::

        selector = make_best_value_selector(factory, result, is_minimize=False)
        config = ShieldConfig(threshold=0.9, comparison="absolute",
                              post_selector=selector)
    """
    def selector(sid: int, blocked_action: int) -> int:
        pvals = factory.get_choice_values_for_state(sid)
        if pvals.size == 0:
            return blocked_action
        return int(np.argmin(pvals) if is_minimize else np.argmax(pvals))
    return selector


def make_random_safe_selector(
    result: ShieldResult,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Callable[[int, int], int]:
    """
    Replace a blocked action with a uniformly random allowed action.

    Falls back to the model's best action (fallback_by_state) when no action
    meets the threshold (dangerous state).

    Example::

        selector = make_random_safe_selector(result, rng=np.random.default_rng(42))
        config = ShieldConfig(threshold=0.9, comparison="absolute",
                              post_selector=selector)
    """
    _rng = np.random.default_rng(rng)

    def selector(sid: int, blocked_action: int) -> int:
        bits = result.bitmask_by_state[sid]
        if bits == 0:
            bits = result.fallback_by_state[sid]
        allowed = [i for i in range(result.bit_width or 64) if bits & (1 << i)]
        if not allowed:
            return blocked_action
        return int(_rng.choice(allowed))
    return selector


def make_nearest_action_selector(
    result: ShieldResult,
) -> Callable[[int, int], int]:
    """
    Replace a blocked action with the allowed action nearest by index.

    Useful when action indices have a natural ordering (e.g. discretised velocity).
    Falls back to the model's best action (fallback_by_state) in dangerous states.

    Example::

        selector = make_nearest_action_selector(result)
        config = ShieldConfig(threshold=0.9, comparison="absolute",
                              post_selector=selector)
    """
    def selector(sid: int, blocked_action: int) -> int:
        bits = result.bitmask_by_state[sid]
        if bits == 0:
            bits = result.fallback_by_state[sid]
        allowed = [i for i in range(result.bit_width or 64) if bits & (1 << i)]
        if not allowed:
            return blocked_action
        return min(allowed, key=lambda a: abs(a - blocked_action))
    return selector


# ---------------------------------------------------------------------------
# PostShieldWrapper
# ---------------------------------------------------------------------------

class PostShieldWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that enforces shield safety via post-shielding.

    The agent receives the unmasked observation and selects any action.
    Before env.step the wrapper calls query_post(sid, action): if the action
    is allowed it passes through unchanged; if blocked the post_selector
    from ShieldConfig is invoked to supply a replacement.

    Requires:
      - env.action_space is gym.spaces.Discrete
      - ShieldConfig.post_selector is set (query_post raises otherwise)
      - obs_to_values maps obs -> {prism_var: value} for every PRISM variable

    After reset() / step(), info contains:
      "shield_state_id"    -> int   state id used for the *next* step
      "shield_safe_action" -> int   action actually passed to env.step
      "shield_corrected"   -> bool  True when the agent's action was replaced
    """

    def __init__(
        self,
        env: gym.Env,
        factory: ShieldFactory,
        config: ShieldConfig,
        *,
        obs_to_values: ObsToValuesFn,
    ) -> None:
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("PostShieldWrapper requires a Discrete action space.")
        if config.post_selector is None:
            raise ValueError(
                "PostShieldWrapper requires ShieldConfig.post_selector to be set."
            )

        self._factory = factory
        self._config = config
        self._obs_to_values = obs_to_values

        self._result: ShieldResult = factory.build(config)
        self._last_sid: Optional[int] = None

    def _obs_to_sid(self, obs: Any, info: Optional[dict] = None) -> int:
        values = self._obs_to_values(obs, info)
        return self._factory.state_lookup.get_state_id(values)

    def rebuild(self, config: ShieldConfig) -> None:
        """Recompute the shield with a new config (e.g. different threshold)."""
        if config.post_selector is None:
            raise ValueError("post_selector must be set when rebuilding PostShieldWrapper.")
        self._config = config
        self._result = self._factory.build(config)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_sid = self._obs_to_sid(obs, info)
        if info is None:
            info = {}
        info["shield_state_id"] = self._last_sid
        return obs, info

    def step(self, action: int):
        if self._last_sid is None:
            raise RuntimeError("step() called before reset().")

        safe_action = self._result.query_post(self._last_sid, action)
        obs, reward, terminated, truncated, info = self.env.step(safe_action)

        self._last_sid = self._obs_to_sid(obs, info)
        if info is None:
            info = {}
        info["shield_state_id"] = self._last_sid
        info["shield_safe_action"] = safe_action
        info["shield_corrected"] = safe_action != action
        return obs, reward, terminated, truncated, info
