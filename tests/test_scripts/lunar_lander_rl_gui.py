#!/usr/bin/env python3
"""Train and render a Gymnasium LunarLander agent.

Examples:
    python3 tests/test_scripts/lunar_lander_rl_gui.py train --timesteps 200000 --shield --render-after
    python3 tests/test_scripts/lunar_lander_rl_gui.py enjoy --model models/lunar_lander_ppo.zip --shield
    python3 tests/test_scripts/lunar_lander_rl_gui.py heuristic --shield
    python3 tests/test_scripts/lunar_lander_rl_gui.py train --benchmark --benchmark-start 10 --benchmark-stop 100 --benchmark-step 10

Dependencies:
    pip install "gymnasium[box2d]" stable-baselines3 sb3-contrib

The heuristic mode only needs Gymnasium with Box2D/Pygame and is useful to
verify that the GUI works before running reinforcement learning.

Shielding:
    The optional --shield flag builds the symbolic PRISM shield from
    lib/stormpy/examples/files/mdp/lunar_lander_angle_shield.prism and exposes
    it as an action mask. Training then uses sb3-contrib MaskablePPO.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL_PATH = Path("models/lunar_lander_ppo.zip")
DEFAULT_BENCHMARK_MODEL_DIR = Path("models/lunar_lander_granularity_benchmark")
DEFAULT_BENCHMARK_CSV_PATH = DEFAULT_BENCHMARK_MODEL_DIR / "benchmark.csv"
TEMPESTPY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHIELD_MODEL_PATH = TEMPESTPY_ROOT / "lib/stormpy/examples/files/mdp/lunar_lander_angle_shield_v2.prism"
DEFAULT_SHIELD_PROPERTY = 'Pmax=? [ G "safe" ]'
ANGULAR_OBS_TO_DEGREES_PER_STEP = 180.0 / (20.0 * math.pi)

# Gymnasium LunarLander discrete actions:
#   0 noop, 1 left engine, 2 main engine, 3 right engine
# The PRISM abstraction uses the same engine-side action labels as Gymnasium.
GYM_TO_SYMBOLIC_BIT: dict[int, int] = {
    0: 0,  # [noop]
    1: 1,  # [thrust_left]
    2: 2,  # [thrust_main]
    3: 3,  # [thrust_right]
}
GYM_ACTION_LABEL: dict[int, str] = {
    0: "noop",
    1: "thrust_left",
    2: "thrust_main",
    3: "thrust_right",
}


def import_gymnasium():
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise SystemExit(
            "Missing Gymnasium. Install it with:\n"
            '  pip install "gymnasium[box2d]"\n'
            "For the RL training mode also install:\n"
            "  pip install stable-baselines3"
        ) from exc
    return gym


def import_ppo():
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "Missing stable-baselines3. Install it with:\n"
            "  pip install stable-baselines3\n"
            'and make sure Gymnasium Box2D is installed with:\n'
            '  pip install "gymnasium[box2d]"'
        ) from exc
    return PPO


def import_maskable_ppo():
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise SystemExit(
            "Missing sb3-contrib. Install it with:\n"
            "  pip install sb3-contrib\n"
            "Shielded training uses sb3-contrib MaskablePPO."
        ) from exc
    return MaskablePPO


def import_shielding():
    try:
        from stormpy.shielding import ShieldConfig, ShieldFactory
    except ImportError as exc:
        raise SystemExit(
            "Missing stormpy/tempestpy shielding bindings. Run this script from "
            "the TempestPy environment that has stormpy.shielding available."
        ) from exc
    return ShieldConfig, ShieldFactory


def make_lunar_lander(render: bool, seed: int | None = None):
    """Create LunarLander with a small compatibility fallback for env versions."""
    gym = import_gymnasium()
    render_mode = "human" if render else None
    errors: list[Exception] = []
    for env_id in ("LunarLander-v3", "LunarLander-v2"):
        try:
            env = gym.make(env_id, render_mode=render_mode)
            if seed is not None:
                env.reset(seed=seed)
            return env
        except Exception as exc:  # Gymnasium raises dependency errors here too.
            errors.append(exc)
    raise SystemExit(
        "Could not create LunarLander-v3 or LunarLander-v2.\n"
        "Most likely Box2D/Pygame is missing. Install with:\n"
        '  pip install "gymnasium[box2d]"\n'
        f"Last error: {errors[-1]}"
    )


def obs_to_angle_shield_valuation(obs: Any, granularity: int, angular_scale: float) -> dict[str, int]:
    """Project a Gymnasium LunarLander observation to the PRISM shield state."""
    angle_rad = float(obs[4])
    angular_velocity = float(obs[5])
    angle = int(round(math.degrees(angle_rad) * granularity))
    v_angle = int(round(angular_velocity * angular_scale * granularity))
    return {
        "angle": max(-90 * granularity, min(90 * granularity, angle)),
        "v_angle": max(-30 * granularity, min(30 * granularity, v_angle)),
    }


def symbolic_bits_to_gym_mask(symbolic_bits: int) -> int:
    """Translate the PRISM action bitmask to Gymnasium's 4-action mask."""
    gym_bits = 0
    for gym_action, symbolic_bit in GYM_TO_SYMBOLIC_BIT.items():
        if symbolic_bits & (1 << symbolic_bit):
            gym_bits |= 1 << gym_action
    return gym_bits


def gym_bits_to_action_mask(gym_bits: int, n_actions: int) -> np.ndarray:
    """Convert packed Gym action bits to the bool mask expected by MaskablePPO."""
    mask = np.array([bool(gym_bits & (1 << action)) for action in range(n_actions)], dtype=bool)
    if not mask.any():
        # print("[shield] Warning: symbolic shield blocked all actions; using noop fallback.")
        # MaskablePPO requires at least one valid action. If the symbolic shield
        # cannot provide one, noop is the least invasive emergency action.
        mask[0] = True
    return mask


class LunarLanderAngleShieldWrapperBase:
    """Expose the symbolic angle shield as a Gymnasium action mask."""

    def __init__(
        self,
        env: Any,
        factory: Any,
        shield: Any,
        *,
        granularity: int,
        angular_scale: float,
        debug: bool = False,
        step_debug: bool = False,
    ) -> None:
        super().__init__(env)
        # Keep the factory alive: the symbolic ADDs in the shield depend on
        # the DD/model context owned by the factory.
        self._shield_factory = factory
        self._shield = shield
        self._granularity = granularity
        self._angular_scale = angular_scale
        self._debug = debug
        self._step_debug = step_debug
        self._last_obs = None
        self._last_mask = np.ones(int(env.action_space.n), dtype=bool)
        self._last_raw_bits = 0
        self._last_fallback_bits = 0
        self._last_symbolic_bits = 0
        self._last_gym_bits = (1 << int(env.action_space.n)) - 1
        self._last_valuation: dict[str, int] | None = None
        self._mask_query_count = 0
        self._fallback_use_count = 0
        self._emergency_mask_count = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _compute_mask(self) -> tuple[dict[str, int], int, int, int, int, np.ndarray]:
        if self._last_obs is None:
            raise RuntimeError("action_masks() called before reset()")

        valuation = obs_to_angle_shield_valuation(
            self._last_obs,
            self._granularity,
            self._angular_scale,
        )
        raw_bits = int(self._shield.bitmask_by_state.query(valuation))
        fallback_bits = int(self._shield.fallback_by_state.query(valuation)) if self._shield.fallback_by_state is not None else 0
        symbolic_bits = raw_bits if raw_bits != 0 else fallback_bits
        gym_bits = symbolic_bits_to_gym_mask(symbolic_bits)
        mask = gym_bits_to_action_mask(gym_bits, int(self.action_space.n))
        return valuation, raw_bits, fallback_bits, symbolic_bits, gym_bits, mask

    def _refresh_mask(self) -> None:
        valuation, raw_bits, fallback_bits, symbolic_bits, gym_bits, mask = self._compute_mask()
        self._last_valuation = valuation
        self._last_raw_bits = raw_bits
        self._last_fallback_bits = fallback_bits
        self._last_symbolic_bits = symbolic_bits
        self._last_gym_bits = gym_bits
        self._last_mask = mask
        self._mask_query_count += 1
        if raw_bits == 0 and fallback_bits != 0:
            self._fallback_use_count += 1
        if gym_bits == 0:
            self._emergency_mask_count += 1

        if self._debug:
            print(
                "[shield] "
                f"valuation={valuation}, "
                f"raw={bin(raw_bits)}, "
                f"fallback={bin(fallback_bits)}, "
                f"effective={bin(symbolic_bits)}, "
                f"gym={bin(gym_bits)}, "
                f"mask={mask.astype(int).tolist()}"
            )

    def _format_choice_values(self, valuation: dict[str, int] | None) -> str:
        choice_values_add = getattr(self._shield, "choice_values_add", None)
        if choice_values_add is None:
            return "unavailable"
        if valuation is None:
            return "unavailable: no valuation"

        meta_variable_names = set(choice_values_add.meta_variable_names)
        action_labels = set(GYM_ACTION_LABEL.values())
        query_base = {name: value for name, value in valuation.items() if name in meta_variable_names}
        missing_state_variables = [name for name in valuation if name not in meta_variable_names]

        parts = []
        for gym_action, action_label in GYM_ACTION_LABEL.items():
            query = dict(query_base)
            for label in action_labels:
                if label in meta_variable_names:
                    query[label] = int(label == action_label)
            for name in meta_variable_names:
                if name not in query:
                    query[name] = 0
            try:
                value = choice_values_add.query(query)
                value_text = f"{float(value):.6g}"
            except Exception as exc:
                value_text = f"error({type(exc).__name__}: {exc})"
            parts.append(f"{gym_action}:{action_label}={value_text}")

        if missing_state_variables:
            parts.append(f"ignored_state_vars={missing_state_variables}")
        return ", ".join(parts)

    def _augment_info(self, info: dict[str, Any]) -> dict[str, Any]:
        out_info = dict(info)
        out_info.update({
            "shield_valuation": self._last_valuation,
            "shield_action_mask": self._last_mask.copy(),
            "shield_raw_bitmask": self._last_raw_bits,
            "shield_fallback_bitmask": self._last_fallback_bits,
            "shield_symbolic_bitmask": self._last_symbolic_bits,
            "shield_gym_bitmask": self._last_gym_bits,
            "shield_used_fallback": self._last_raw_bits == 0 and self._last_fallback_bits != 0,
            "shield_used_emergency": self._last_gym_bits == 0,
        })
        return out_info

    def action_masks(self) -> np.ndarray:
        return self._last_mask.copy()

    def shield_stats(self) -> dict[str, int | float]:
        fallback_rate = self._fallback_use_count / self._mask_query_count if self._mask_query_count else 0.0
        emergency_rate = self._emergency_mask_count / self._mask_query_count if self._mask_query_count else 0.0
        return {
            "mask_queries": self._mask_query_count,
            "fallback_uses": self._fallback_use_count,
            "fallback_rate": fallback_rate,
            "emergency_masks": self._emergency_mask_count,
            "emergency_rate": emergency_rate,
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._refresh_mask()
        return obs, self._augment_info(info)

    def step(self, action: int):
        previous_obs = np.asarray(self._last_obs, dtype=float).tolist() if self._last_obs is not None else None
        previous_valuation = self._last_valuation
        previous_mask = self._last_mask.copy()
        previous_raw_bits = self._last_raw_bits
        previous_fallback_bits = self._last_fallback_bits
        previous_symbolic_bits = self._last_symbolic_bits
        previous_gym_bits = self._last_gym_bits

        obs, reward, terminated, truncated, info = self.env.step(int(action))
        current_obs = np.asarray(obs, dtype=float).tolist()
        self._last_obs = obs
        self._refresh_mask()
        if self._step_debug:
            previous_choice_values = self._format_choice_values(previous_valuation)
            current_choice_values = self._format_choice_values(self._last_valuation)
            print(
                "[step-debug] "
                f"action={int(action)}, "
                f"reward={reward:.6f}, "
                f"terminated={terminated}, "
                f"truncated={truncated}"
            )
            print(
                "[step-debug] before "
                f"obs={previous_obs}, "
                f"valuation={previous_valuation}, "
                f"raw={bin(previous_raw_bits)}, "
                f"fallback={bin(previous_fallback_bits)}, "
                f"effective={bin(previous_symbolic_bits)}, "
                f"gym={bin(previous_gym_bits)}, "
                f"mask={previous_mask.astype(int).tolist()}, "
                f"choice_values=[{previous_choice_values}]"
            )
            print(
                "[step-debug] after  "
                f"obs={current_obs}, "
                f"valuation={self._last_valuation}, "
                f"raw={bin(self._last_raw_bits)}, "
                f"fallback={bin(self._last_fallback_bits)}, "
                f"effective={bin(self._last_symbolic_bits)}, "
                f"gym={bin(self._last_gym_bits)}, "
                f"mask={self._last_mask.astype(int).tolist()}, "
                f"choice_values=[{current_choice_values}]"
            )
            input("[step-debug] press Enter for next environment step...")
        return obs, reward, terminated, truncated, self._augment_info(info)

    def close(self) -> None:
        self.env.close()


class StepDebugWrapperBase:
    """Pause after each environment step and print raw/shield debug data."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self._step_debug_last_obs = None
        self._step_debug_enabled = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _find_shield_debug_source(self) -> Any | None:
        env = self.env
        seen: set[int] = set()
        while env is not None and id(env) not in seen:
            seen.add(id(env))
            if hasattr(env, "_last_valuation") and hasattr(env, "_last_mask"):
                return env
            env = getattr(env, "env", None)
        return None

    def _collect_step_debug_state(self) -> dict[str, Any]:
        shield_source = self._find_shield_debug_source()
        if shield_source is None:
            return {
                "shield": "absent",
                "valuation": None,
                "mask": None,
                "raw_bits": None,
                "fallback_bits": None,
                "symbolic_bits": None,
                "gym_bits": None,
                "choice_values": "unavailable: run with --shield",
            }

        valuation = getattr(shield_source, "_last_valuation", None)
        formatter = getattr(shield_source, "_format_choice_values", None)
        return {
            "shield": "present",
            "valuation": valuation,
            "mask": getattr(shield_source, "_last_mask", None),
            "raw_bits": getattr(shield_source, "_last_raw_bits", None),
            "fallback_bits": getattr(shield_source, "_last_fallback_bits", None),
            "symbolic_bits": getattr(shield_source, "_last_symbolic_bits", None),
            "gym_bits": getattr(shield_source, "_last_gym_bits", None),
            "choice_values": formatter(valuation) if formatter is not None else "unavailable",
        }

    @staticmethod
    def _format_bits(bits: Any) -> str:
        return bin(int(bits)) if bits is not None else "unavailable"

    @staticmethod
    def _format_mask(mask: Any) -> str:
        if mask is None:
            return "unavailable"
        return np.asarray(mask, dtype=int).tolist()

    @staticmethod
    def _format_action(action: Any) -> Any:
        action_array = np.asarray(action)
        if action_array.size == 1:
            return int(action_array.item())
        return action_array.tolist()

    def _print_step_debug_state(self, label: str, obs: Any, state: dict[str, Any]) -> None:
        obs_text = np.asarray(obs, dtype=float).tolist() if obs is not None else None
        print(
            f"[step-debug] {label:<6} "
            f"obs={obs_text}, "
            f"shield={state['shield']}, "
            f"valuation={state['valuation']}, "
            f"raw={self._format_bits(state['raw_bits'])}, "
            f"fallback={self._format_bits(state['fallback_bits'])}, "
            f"effective={self._format_bits(state['symbolic_bits'])}, "
            f"gym={self._format_bits(state['gym_bits'])}, "
            f"mask={self._format_mask(state['mask'])}, "
            f"choice_values=[{state['choice_values']}]"
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._step_debug_last_obs = obs
        return obs, info

    def step(self, action: int):
        previous_obs = self._step_debug_last_obs
        previous_debug_state = self._collect_step_debug_state()

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_debug_last_obs = obs

        if self._step_debug_enabled:
            current_debug_state = self._collect_step_debug_state()
            print(
                "[step-debug] "
                f"action={self._format_action(action)}, "
                f"reward={float(reward):.6f}, "
                f"terminated={terminated}, "
                f"truncated={truncated}"
            )
            self._print_step_debug_state("before", previous_obs, previous_debug_state)
            self._print_step_debug_state("after", obs, current_debug_state)
            try:
                input("[step-debug] press Enter for next environment step...")
            except EOFError:
                print("[step-debug] stdin unavailable; disabling step-debug pauses.")
                self._step_debug_enabled = False

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()


def build_angle_shield(args: argparse.Namespace) -> tuple[Any, Any]:
    ShieldConfig, ShieldFactory = import_shielding()
    factory = ShieldFactory(
        str(args.shield_model),
        property=args.shield_property,
        engine="dd",
        constants={"GRAN": args.shield_granularity},
    )
    shield = factory.build(ShieldConfig(threshold=args.shield_threshold))
    return factory, shield


def get_training_hyperparameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "learning_rate": getattr(args, "learning_rate", 3e-4),
        "n_steps": getattr(args, "n_steps", 2048),
        "batch_size": getattr(args, "batch_size", 64),
        "gamma": getattr(args, "gamma", 0.99),
    }


def benchmark_shield(args: argparse.Namespace) -> None:
    ShieldConfig, ShieldFactory = import_shielding()
    MaskablePPO = import_maskable_ppo()
    args.benchmark_model_dir.mkdir(parents=True, exist_ok=True)
    args.benchmark_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "granularity",
        "build_model_seconds",
        "build_shield_seconds",
        "train_seconds",
        "total_seconds",
        "sample_mask",
        "mask_queries",
        "fallback_uses",
        "fallback_rate",
        "emergency_masks",
        "emergency_rate",
        "model_path",
    ]
    print(",".join(fieldnames))

    with args.benchmark_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for granularity in range(args.benchmark_start, args.benchmark_stop + 1, args.benchmark_step):
            total_start = time.perf_counter()

            model_start = time.perf_counter()
            factory = ShieldFactory(
                str(args.shield_model),
                property=args.shield_property,
                engine="dd",
                constants={"GRAN": granularity},
            )
            model_seconds = time.perf_counter() - model_start

            shield_start = time.perf_counter()
            shield = factory.build(ShieldConfig(threshold=args.shield_threshold))
            shield_seconds = time.perf_counter() - shield_start

            sample_mask = int(shield.query_mask({"angle": 0, "v_angle": 0}))
            train_start = time.perf_counter()
            env = maybe_wrap_with_angle_shield(
                make_lunar_lander(render=False, seed=args.seed),
                args,
                factory=factory,
                shield=shield,
                granularity=granularity,
            )
            model = MaskablePPO(
                "MlpPolicy",
                env,
                verbose=1,
                seed=args.seed,
                **get_training_hyperparameters(args),
            )
            model.learn(total_timesteps=args.benchmark_timesteps)
            model_path = args.benchmark_model_dir / f"lunar_lander_maskable_ppo_gran_{granularity}.zip"
            model.save(model_path)
            shield_stats = env.shield_stats()
            env.close()
            train_seconds = time.perf_counter() - train_start

            total_seconds = time.perf_counter() - total_start
            row = {
                "granularity": granularity,
                "build_model_seconds": f"{model_seconds:.6f}",
                "build_shield_seconds": f"{shield_seconds:.6f}",
                "train_seconds": f"{train_seconds:.6f}",
                "total_seconds": f"{total_seconds:.6f}",
                "sample_mask": bin(sample_mask),
                "mask_queries": shield_stats["mask_queries"],
                "fallback_uses": shield_stats["fallback_uses"],
                "fallback_rate": f"{shield_stats['fallback_rate']:.6f}",
                "emergency_masks": shield_stats["emergency_masks"],
                "emergency_rate": f"{shield_stats['emergency_rate']:.6f}",
                "model_path": str(model_path),
            }
            writer.writerow(row)
            csv_file.flush()
            print(",".join(str(row[name]) for name in fieldnames))

            del model
            del shield
            del factory
            del env
            gc.collect()

    print(f"saved benchmark csv to {args.benchmark_csv}")


def maybe_wrap_with_angle_shield(
    env: Any,
    args: argparse.Namespace,
    *,
    factory: Any | None = None,
    shield: Any | None = None,
    granularity: int | None = None,
) -> Any:
    gym = import_gymnasium()

    if args.shield or (factory is not None and shield is not None):
        if factory is None or shield is None:
            factory, shield = build_angle_shield(args)

        class LunarLanderAngleShieldWrapper(LunarLanderAngleShieldWrapperBase, gym.Wrapper):
            pass

        env = LunarLanderAngleShieldWrapper(
            env,
            factory,
            shield,
            granularity=args.shield_granularity if granularity is None else granularity,
            angular_scale=args.shield_angular_scale,
            debug=args.shield_debug,
            step_debug=False,
        )

    if args.step_debug:
        class StepDebugWrapper(StepDebugWrapperBase, gym.Wrapper):
            pass

        env = StepDebugWrapper(env)

    return env


def heuristic_action(obs: Any) -> int:
    """A tiny hand-written controller for GUI smoke tests.

    Actions are Gymnasium LunarLander discrete actions:
      0 noop, 1 left engine, 2 main engine, 3 right engine.
    """
    x, y, vx, vy, angle, angular_velocity, left_contact, right_contact = obs
    del y

    target_angle = x * 0.5 + vx * 1.0
    target_angle = max(-0.4, min(0.4, target_angle))
    angle_error = target_angle - angle
    hover = abs(vy) > 0.35 or left_contact or right_contact

    if hover and abs(angle_error) < 0.05 and abs(angular_velocity) < 0.15:
        return 2
    if angle_error > 0.05 or angular_velocity < -0.25:
        return 3
    if angle_error < -0.05 or angular_velocity > 0.25:
        return 1
    return 0


def apply_action_mask(action: int, action_mask: np.ndarray | None) -> int:
    if action_mask is None or bool(action_mask[action]):
        return action
    for fallback_action, allowed in enumerate(action_mask):
        if allowed:
            return fallback_action
    return 0


def run_policy(env, predict_action, episodes: int, deterministic: bool = True) -> None:
    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated):
            action_mask = env.action_masks() if hasattr(env, "action_masks") else None
            action = predict_action(obs, deterministic, action_mask)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1
        print(f"episode={episode + 1} reward={total_reward:.2f} steps={steps}")


def train(args: argparse.Namespace) -> None:
    Algorithm = import_maskable_ppo() if args.shield else import_ppo()
    train_env = maybe_wrap_with_angle_shield(
        make_lunar_lander(render=args.render_train, seed=args.seed),
        args,
    )
    model = Algorithm(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=args.seed,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
    )
    model.learn(total_timesteps=args.timesteps)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model)
    train_env.close()
    print(f"saved model to {args.model}")

    if args.render_after:
        render_env = maybe_wrap_with_angle_shield(
            make_lunar_lander(render=True, seed=args.seed),
            args,
        )
        run_policy(
            render_env,
            lambda obs, deterministic, action_mask: model.predict(
                obs,
                deterministic=deterministic,
                action_masks=action_mask,
            )[0] if action_mask is not None else model.predict(obs, deterministic=deterministic)[0],
            args.episodes,
        )
        render_env.close()


def enjoy(args: argparse.Namespace) -> None:
    Algorithm = import_maskable_ppo() if args.shield else import_ppo()
    env = maybe_wrap_with_angle_shield(make_lunar_lander(render=True, seed=args.seed), args)
    model = Algorithm.load(args.model, env=env)
    run_policy(
        env,
        lambda obs, deterministic, action_mask: model.predict(
            obs,
            deterministic=deterministic,
            action_masks=action_mask,
        )[0] if action_mask is not None else model.predict(obs, deterministic=deterministic)[0],
        args.episodes,
    )
    env.close()


def heuristic(args: argparse.Namespace) -> None:
    env = maybe_wrap_with_angle_shield(make_lunar_lander(render=True, seed=args.seed), args)
    run_policy(
        env,
        lambda obs, deterministic, action_mask: apply_action_mask(heuristic_action(obs), action_mask),
        args.episodes,
    )
    env.close()


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--shield", action="store_true", help="Apply the symbolic LunarLander angle shield.")
    parser.add_argument("--shield-model", type=Path, default=DEFAULT_SHIELD_MODEL_PATH)
    parser.add_argument("--shield-property", default=DEFAULT_SHIELD_PROPERTY)
    parser.add_argument("--shield-threshold", type=float, default=1.0)
    parser.add_argument("--shield-granularity", type=int, default=50)
    parser.add_argument(
        "--shield-angular-scale",
        type=float,
        default=ANGULAR_OBS_TO_DEGREES_PER_STEP,
        help=(
            "Scale Gym obs[5] before multiplying by shield granularity. "
            "The default converts Gym's angular-velocity observation to "
            "degrees per environment step."
        ),
    )
    parser.add_argument("--shield-debug", action="store_true", help="Print blocked action replacements.")
    parser.add_argument("--step-debug", action="store_true", help="Pause after every environment step and print the current shield state.")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark shield construction plus MaskablePPO training for increasing GRAN values.")
    parser.add_argument("--benchmark-start", type=int, default=10)
    parser.add_argument("--benchmark-stop", type=int, default=100)
    parser.add_argument("--benchmark-step", type=int, default=10)
    parser.add_argument("--benchmark-timesteps", type=int, default=200_000)
    parser.add_argument("--benchmark-model-dir", type=Path, default=DEFAULT_BENCHMARK_MODEL_DIR)
    parser.add_argument("--benchmark-csv", type=Path, default=DEFAULT_BENCHMARK_CSV_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gymnasium LunarLander with RL and GUI rendering.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a PPO agent.")
    train_parser.add_argument("--timesteps", type=int, default=200_000)
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    train_parser.add_argument("--render-train", action="store_true", help="Render during training; very slow but useful for debugging.")
    train_parser.add_argument("--render-after", action="store_true", help="Render a few episodes after training.")
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--n-steps", type=int, default=2048)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--gamma", type=float, default=0.99)
    add_common_arguments(train_parser)
    train_parser.set_defaults(func=train)

    enjoy_parser = subparsers.add_parser("enjoy", help="Load a trained PPO model and render it.")
    enjoy_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    add_common_arguments(enjoy_parser)
    enjoy_parser.set_defaults(func=enjoy)

    heuristic_parser = subparsers.add_parser("heuristic", help="Render a simple hand-written controller.")
    add_common_arguments(heuristic_parser)
    heuristic_parser.set_defaults(func=heuristic)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.benchmark:
        benchmark_shield(args)
        return 0
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
