from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


TEMPESTPY_ROOT = Path(__file__).resolve().parents[2]
LIB_DIR = TEMPESTPY_ROOT / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from stormpy.shielding import ShieldConfig, ShieldFactory


EXAMPLE_DIR = LIB_DIR / "stormpy" / "examples" / "files" / "mdp"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    prop: str
    seed_valuations: tuple[dict[str, int], ...]


MODEL_SPECS: dict[str, ModelSpec] = {
    "grid": ModelSpec(
        name="grid",
        path=EXAMPLE_DIR / "grid_slippery.prism",
        prop="Pmax=? [ !crashed U goal ]",
        seed_valuations=(
            {"x": 1, "y": 1},
            {"x": 5, "y": 5},
            {"x": 250, "y": 250},
            {"x": 500, "y": 500},
        ),
    ),
    "adv": ModelSpec(
        name="adv",
        path=EXAMPLE_DIR / "fourroomsheist_6_advs.prism",
        prop="Pmax=? [ !collision U AgentIsOnGoal ]", 
        seed_valuations=(
            {
                "xAgent": 1,
                "yAgent": 1,
                "xAdvBlue": 1,
                "yAdvBlue": 1,
                "xAdvGreen": 1,
                "yAdvGreen": 1,
                "xAdvGrey": 1,
                "yAdvGrey": 1,
                "xAdvLightblue": 1,
                "yAdvLightblue": 1,
                "xAdvPurple": 1,
                "yAdvPurple": 1,
                "xAdvYellow": 1,
                "yAdvYellow": 1,
                "turn": 0,
            },
        ),
    ),
}


@dataclass
class TimingRow:
    model: str
    engine: str
    stage: str
    seconds: float
    repeats: int = 1
    query_count: int = 0
    sample_count: int = 0
    result: str = "ok"
    detail: str = ""


def time_once(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    value = fn()
    end = time.perf_counter()
    return value, end - start


def time_repeated(fn: Callable[[], Any], repeats: int) -> tuple[Any, float]:
    timings: list[float] = []
    last_value: Any = None
    for _ in range(repeats):
        last_value, elapsed = time_once(fn)
        timings.append(elapsed)
    return last_value, statistics.median(timings)


def cycle(values: list[Any], count: int) -> list[Any]:
    if not values:
        return []
    return [values[i % len(values)] for i in range(count)]


def sample_sparse_state_ids(factory: ShieldFactory, sample_count: int, seed: int) -> list[int]:
    nr_states = int(factory.model.nr_states)
    sample_count = max(1, sample_count)
    if nr_states <= sample_count:
        return list(range(nr_states))
    rng = random.Random(seed)
    must_include = [0] if sample_count == 1 else [0, nr_states - 1]
    random_ids = rng.sample(range(nr_states), sample_count - len(must_include))
    return must_include + random_ids


def valuations_from_sparse_factory(factory: ShieldFactory, state_ids: list[int]) -> list[dict[str, Any]]:
    lookup = factory.state_lookup
    return [lookup.values_for_state_id(sid) for sid in state_ids]


def benchmark_factory_and_build(
    spec: ModelSpec,
    engine: str,
    threshold: float,
) -> tuple[ShieldFactory, Any, list[TimingRow]]:
    rows: list[TimingRow] = []

    print("Beginning with modelchecking...", spec.name, "engine", engine)

    factory, init_seconds = time_once(
        lambda: ShieldFactory(str(spec.path), property=spec.prop, engine=engine)
    )
    rows.append(TimingRow(spec.name, engine, "factory_init", init_seconds))

    shield, build_seconds = time_once(lambda: factory.build(ShieldConfig(threshold=threshold)))
    rows.append(TimingRow(spec.name, engine, "shield_build", build_seconds))
    rows.append(TimingRow(spec.name, engine, "generation_total", init_seconds + build_seconds))

    print("Done with modelchecking...", spec.name, "engine", engine)

    return factory, shield, rows


def benchmark_sparse_inference(
    spec: ModelSpec,
    factory: ShieldFactory,
    shield: Any,
    state_ids: list[int],
    query_count: int,
    repeats: int,
) -> list[TimingRow]:
    rows: list[TimingRow] = []
    ids = cycle(state_ids, query_count)

    def query_by_id() -> int:
        acc = 0
        for sid in ids:
            acc ^= int(shield.query_mask(sid))
        return acc

    _, seconds = time_repeated(query_by_id, repeats)
    rows.append(
        TimingRow(spec.name, "sparse", "inference_state_id", seconds, repeats, query_count, len(state_ids))
    )

    return rows


def benchmark_symbolic_inference(
    spec: ModelSpec,
    shield: Any,
    valuations: list[dict[str, Any]],
    query_count: int,
    repeats: int,
) -> list[TimingRow]:
    vals = cycle(valuations, query_count)

    def query_adds() -> int:
        acc = 0
        for valuation in vals:
            acc ^= int(shield.bitmask_by_state.query(valuation))
            if shield.fallback_by_state is not None:
                acc ^= int(shield.fallback_by_state.query(valuation))
        return acc

    _, seconds = time_repeated(query_adds, repeats)
    return [
        TimingRow(spec.name, "dd", "inference_valuation", seconds, repeats, query_count, len(valuations))
    ]


def run_model(
    spec: ModelSpec,
    engines: list[str],
    threshold: float,
    sample_count: int,
    query_count: int,
    repeats: int,
    seed: int,
) -> list[TimingRow]:
    rows: list[TimingRow] = []
    sparse_state_ids: list[int] = []
    valuations: list[dict[str, Any]] = list(spec.seed_valuations)

    for engine in engines:
        gc.collect()
        try:
            factory, shield, build_rows = benchmark_factory_and_build(spec, engine, threshold)
            rows.extend(build_rows)

            if engine == "sparse":
                sparse_state_ids = sample_sparse_state_ids(factory, sample_count, seed)
                rows.extend(
                    benchmark_sparse_inference(
                        spec,
                        factory,
                        shield,
                        sparse_state_ids,
                        query_count,
                        repeats,
                    )
                )
            else:
                rows.extend(
                    benchmark_symbolic_inference(
                        spec,
                        shield,
                        valuations,
                        query_count,
                        repeats,
                    )
                )
        except Exception as exc:
            rows.append(
                TimingRow(
                    model=spec.name,
                    engine=engine,
                    stage="failed",
                    seconds=0.0,
                    repeats=repeats,
                    query_count=query_count,
                    sample_count=sample_count,
                    result=type(exc).__name__,
                    detail=str(exc),
                )
            )

    return rows


def print_table(rows: list[TimingRow]) -> None:
    headers = [
        "model",
        "engine",
        "stage",
        "seconds",
        "repeats",
        "query_count",
        "sample_count",
        "result",
        "detail",
    ]
    widths = {header: len(header) for header in headers}
    rendered: list[dict[str, str]] = []
    for row in rows:
        item = asdict(row)
        item["seconds"] = f"{row.seconds:.6f}"
        rendered_item = {key: str(item[key]) for key in headers}
        rendered.append(rendered_item)
        for key, value in rendered_item.items():
            widths[key] = max(widths[key], len(value))

    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for item in rendered:
        print(" | ".join(item[header].ljust(widths[header]) for header in headers))


def write_csv(rows: list[TimingRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(rows: list[TimingRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump([asdict(row) for row in rows], handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare explicit and symbolic TempestPy shield generation/inference runtime."
    )
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_SPECS), default=["grid", "adv"])
    parser.add_argument("--engines", nargs="+", choices=["sparse", "dd"], default=["sparse", "dd"])
    parser.add_argument("--threshold", type=float, default=1.1)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--query-count", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--grid-property", default=MODEL_SPECS["grid"].prop)
    parser.add_argument("--adv-property", default=MODEL_SPECS["adv"].prop)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return exit code 0 even if an engine fails, useful for very large explicit runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = dict(MODEL_SPECS)
    specs["grid"] = ModelSpec(
        name="grid",
        path=MODEL_SPECS["grid"].path,
        prop=args.grid_property,
        seed_valuations=MODEL_SPECS["grid"].seed_valuations,
    )
    specs["adv"] = ModelSpec(
        name="adv",
        path=MODEL_SPECS["adv"].path,
        prop=args.adv_property,
        seed_valuations=MODEL_SPECS["adv"].seed_valuations,
    )

    rows: list[TimingRow] = []
    for model_name in args.models:
        rows.extend(
            run_model(
                specs[model_name],
                args.engines,
                args.threshold,
                args.sample_count,
                args.query_count,
                args.repeats,
                args.seed,
            )
        )

    print_table(rows)
    if args.csv:
        write_csv(rows, args.csv)
    if args.json:
        write_json(rows, args.json)
    if args.allow_failures:
        return 0
    return 0 if all(row.stage != "failed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
