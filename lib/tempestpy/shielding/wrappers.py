from __future__ import annotations

try:
    import gymnasium as gym
except ImportError as _exc:
    raise ImportError(
        "tempestpy.shielding.wrappers requires gymnasium. "
        "Install it with:  pip install gymnasium"
    ) from _exc

from typing import Any, Callable, Mapping, Optional

import numpy as np

from .shield_factory import ShieldFactory
from .masking import ShieldConfig
from .shield_result import ShieldResult

ObsToValuesFn = Callable[[Any, Optional[dict]], Mapping[str, Any]]


def _bitmask_to_bool_array(bits: int, n_actions: int) -> np.ndarray:
    return np.array([(bits >> i) & 1 for i in range(n_actions)], dtype=bool)


# ---------------------------------------------------------------------------
# PreShieldWrapper
# ---------------------------------------------------------------------------

class PreShieldWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that adds pre-shield action masking.

    On every ``reset()`` and ``step()``, the wrapper queries the shield and
    injects an ``"action_mask"`` boolean array into ``info``.  The mask can
    be consumed directly or via the ``action_masks()`` hook for SB3
    ``MaskablePPO``.

    The ``info`` dict after ``reset()`` / ``step()`` contains:

    * ``"action_mask"``     — ``np.ndarray[bool]``, shape ``(n_actions,)``
    * ``"shield_state_id"`` — ``int``, state id used for the current mask
    * ``"shield_bitmask"``  — ``int``, raw packed bitmask

    See Also
    --------
    PostShieldWrapper : Corrects the agent's chosen action after the fact.
    ShieldFactory.build : Produces the `ShieldResult` used internally.
    """

    def __init__(
        self,
        env: gym.Env,
        factory: ShieldFactory,
        config: ShieldConfig,
        *,
        obs_to_values: ObsToValuesFn,
    ) -> None:
        """
        Initialise a PreShieldWrapper.

        Parameters
        ----------
        env :
            The Gymnasium environment to wrap.  Must have a
            ``gym.spaces.Discrete`` action space.
        factory :
            A `ShieldFactory` with a built model and property.
        config :
            `ShieldConfig` specifying the threshold and comparison mode.
            ``post_selector`` is not used here.
        obs_to_values :
            Callable ``(obs, info) -> {prism_var: value}`` that maps an
            observation (and optionally ``info``) to the PRISM state variables
            needed to look up the shield state id.

        Raises
        ------
        TypeError
            If ``env.action_space`` is not ``gym.spaces.Discrete``.
        """
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
        """
        Return the current boolean action mask for SB3 ``MaskablePPO``.

        Returns
        -------
        numpy.ndarray
            Boolean array of shape ``(n_actions,)``.

        Raises
        ------
        RuntimeError
            If called before the first ``reset()``.
        """
        if self._last_mask is None:
            raise RuntimeError("action_masks() called before reset().")
        return self._last_mask

    def rebuild(self, config: ShieldConfig) -> None:
        """
        Recompute the shield with a new `ShieldConfig`.

        Useful for changing the threshold mid-training without re-creating the
        wrapper.  Resets the cached action mask.

        Parameters
        ----------
        config :
            New shield configuration.
        """
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
    Create a post-selector that replaces a blocked action with the best-value action.

    The returned callable closes over *factory* to retrieve per-state choice
    values at call time.  Works in both normal and dangerous states; in
    dangerous states it falls back to the argmax/argmin action.

    Parameters
    ----------
    factory :
        The `ShieldFactory` used to look up choice values.
    result :
        The `ShieldResult` (currently unused; kept for a consistent selector API).
    is_minimize :
        If ``True``, select the action with the *lowest* choice value.
        Default is ``False`` (maximise).

    Returns
    -------
    Callable[[int, int], int]
        ``selector(state_id, blocked_action) -> replacement_action``.

    Examples
    --------
    >>> selector = make_best_value_selector(factory, result, is_minimize=False)
    >>> config = ShieldConfig(threshold=0.9, post_selector=selector)
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
    Create a post-selector that replaces a blocked action with a random allowed action.

    In dangerous states (no action meets the threshold) the fallback action
    from ``ShieldResult.fallback_by_state`` is used instead.

    Parameters
    ----------
    result :
        The `ShieldResult` used to look up allowed actions.
    rng :
        Optional NumPy random generator.  If ``None``, a fresh default
        generator is created.

    Returns
    -------
    Callable[[int, int], int]
        ``selector(state_id, blocked_action) -> replacement_action``.

    Examples
    --------
    >>> selector = make_random_safe_selector(result, rng=np.random.default_rng(42))
    >>> config = ShieldConfig(threshold=0.9, post_selector=selector)
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
    Create a post-selector that replaces a blocked action with the nearest allowed action.

    Nearest is measured by absolute index distance, making this useful when
    action indices have a natural ordering (e.g. discretised velocity).
    Falls back to ``ShieldResult.fallback_by_state`` in dangerous states.

    Parameters
    ----------
    result :
        The `ShieldResult` used to look up allowed actions.

    Returns
    -------
    Callable[[int, int], int]
        ``selector(state_id, blocked_action) -> replacement_action``.

    Examples
    --------
    >>> selector = make_nearest_action_selector(result)
    >>> config = ShieldConfig(threshold=0.9, post_selector=selector)
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

    The agent selects any action from the unmasked observation.  Before the
    action reaches ``env.step``, the wrapper checks the shield: if the action
    is allowed it passes through unchanged; if blocked, ``ShieldConfig.post_selector``
    is called to supply a safe replacement.

    The ``info`` dict after ``reset()`` / ``step()`` contains:

    * ``"shield_state_id"``    — ``int``, state id used for the *next* step
    * ``"shield_safe_action"`` — ``int``, the action actually passed to ``env.step``
    * ``"shield_corrected"``   — ``bool``, ``True`` when the agent's action was replaced

    See Also
    --------
    PreShieldWrapper : Injects an action mask before the agent acts.
    make_best_value_selector : A ready-made post-selector for this wrapper.
    """

    def __init__(
        self,
        env: gym.Env,
        factory: ShieldFactory,
        config: ShieldConfig,
        *,
        obs_to_values: ObsToValuesFn,
    ) -> None:
        """
        Initialise a PostShieldWrapper.

        Parameters
        ----------
        env :
            The Gymnasium environment to wrap.  Must have a
            ``gym.spaces.Discrete`` action space.
        factory :
            A `ShieldFactory` with a built model and property.
        config :
            `ShieldConfig` with a non-``None`` ``post_selector``.
        obs_to_values :
            Callable ``(obs, info) -> {prism_var: value}`` that maps an
            observation to the PRISM state variables needed for shield lookup.

        Raises
        ------
        TypeError
            If ``env.action_space`` is not ``gym.spaces.Discrete``.
        ValueError
            If ``config.post_selector`` is ``None``.
        """
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
        """
        Recompute the shield with a new `ShieldConfig`.

        Parameters
        ----------
        config :
            New configuration.  Must have a non-``None`` ``post_selector``.

        Raises
        ------
        ValueError
            If ``config.post_selector`` is ``None``.
        """
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
