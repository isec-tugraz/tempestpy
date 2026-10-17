[![Tempest upstream](https://img.shields.io/badge/tempest--upstream-dc451dc8-blue)](https://github.com/tempestchecker/tempest/commit/dc451dc83)

# Tempestpy

Tempestpy is a Python interface for shield synthesis, built as a fork of [stormpy](https://github.com/stormchecker/stormpy). It provides Python bindings for [Tempest](https://github.com/tempestchecker/tempest) — a probabilistic model checker extended with shield synthesis capabilities.

## Dependencies

Tempestpy requires [Tempest](https://gitlab.tugraz.at/trustedai/tempest) to be built from source before installation. Follow the Tempest README to build Tempest, then note the path to your build directory (e.g. `<tempest-root>/build`).

We require a python version 3.10 or newer.

## Installation

Tempestpy uses [scikit-build-core](https://github.com/scikit-build/scikit-build-core) (CMake + pybind11).

```bash
# Clone the repository
git clone <tempestpy-repo-url>
cd tempestpy

# Install (adjust STORM_DIR_HINT to your Tempest build path)
pip install . \
  --config-settings=cmake.define.STORM_DIR_HINT=<tempest-root>/build \
  --config-settings=cmake.define.CMAKE_BUILD_PARALLEL_LEVEL=4
```

Replace `<tempest-root>/build` with the actual path to your Tempest build directory.

### Tempest version

The Tempest version that Tempestpy is tested against is pinned in `pyproject.toml` (`STORM_GIT_TAG` and `STORM_GIT_COMMIT`, with the repository in `STORM_GIT_REPO`). If you do not pass `STORM_DIR_HINT` and no Tempest build is found, CMake fetches and builds exactly this version from the repository, which takes a long time.

When an existing Tempest build is used, CMake checks it while configuring:

- The Tempest version must be at least the minimum required by Tempestpy (`TEMPEST_MIN_VERSION` in `CMakeLists.txt`). Otherwise the configuration fails.
- If the build is not at the pinned commit, was built from a dirty working tree (uncommitted changes), or its revision or state is unknown, a warning is printed, and the installation continues. Such builds may or may not work.
- With `--config-settings=cmake.define.STORM_STRICT=ON`, these warnings are errors instead. This is used for building wheels.

The revision and state Tempest was built from can also be seen with `tempest --version`.

## Shielding

The primary extension over stormpy is the `stormpy.shielding` module, which synthesizes action masks (shields) from probabilistic PRISM models.

```python
import stormpy.shielding as shielding

factory = shielding.ShieldFactory(
    model_path="model.prism",
    property_str="Pmin=? [F unsafe]",
)
config = shielding.ShieldConfig(threshold=0.05)
result = factory.compute(config)
```

See `lib/stormpy/shielding/` for the full module.

## Authors
- Stefan Pranger (tempestpy)

