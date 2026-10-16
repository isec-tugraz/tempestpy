import pytest

from tempestpy.shielding.action_lookup import ActionLookup
from tempestpy.shielding.masking import ShieldConfig
from tempestpy.shielding.shield_result import ExplicitShieldResult, SymbolicShieldResult


class TestShieldConfig:
    def test_defaults(self):
        config = ShieldConfig()
        assert config.threshold == 1.0
        assert config.comparison == "absolute"
        assert config.post_selector is None
        assert config.action_lookup is None
        assert config._bitmask_view == "local"

    def test_bitmask_view_is_global_with_action_lookup(self):
        lookup = ActionLookup.from_dict({"a": 0})
        config = ShieldConfig(action_lookup=lookup)
        assert config._bitmask_view == "global"

    def test_frozen(self):
        config = ShieldConfig()
        with pytest.raises(Exception):
            config.threshold = 0.5


class TestEngineDispatch:
    def test_sparse_engine_returns_explicit_result(self, sparse_factory):
        result = sparse_factory.build(ShieldConfig(threshold=0.9))
        assert isinstance(result, ExplicitShieldResult)

    def test_dd_engine_returns_symbolic_result(self, dd_factory):
        result = dd_factory.build(ShieldConfig(threshold=0.9))
        assert isinstance(result, SymbolicShieldResult)


class TestAbsoluteComparisonMaximize:
    def test_threshold_partially_blocks_actions(self, bounded_max_factory):
        # state 10 pvals = [0.144, 0.144, 0.648]
        result = bounded_max_factory.compute_mask(ShieldConfig(threshold=0.5))
        assert result.bitmask_by_state[10] == 0b100
        assert result.is_critical(10)

    def test_threshold_one_blocks_everything_triggers_fallback(self, bounded_max_factory):
        # state 13 pvals = [0.552, 0.552, 0.936], none reach 1.0
        result = bounded_max_factory.compute_mask(ShieldConfig(threshold=1.0))
        assert result.bitmask_by_state[13] == 0
        assert result.fallback_by_state[13] == 0b100  # argmax is action index 2
        assert result.is_dangerous(13)

    def test_low_threshold_allows_all(self, bounded_max_factory):
        result = bounded_max_factory.compute_mask(ShieldConfig(threshold=0.0))
        assert result.is_safe(15)


class TestAbsoluteComparisonMinimize:
    def test_threshold_partially_blocks_actions(self, bounded_min_factory):
        # state 13 pvals = [0.0, 0.0, 0.6]
        result = bounded_min_factory.compute_mask(ShieldConfig(threshold=0.1))
        assert result.bitmask_by_state[13] == 0b011
        assert result.is_critical(13)

    def test_threshold_allows_all_when_generous(self, bounded_min_factory):
        result = bounded_min_factory.compute_mask(ShieldConfig(threshold=0.6))
        assert result.is_safe(13)

    def test_unsafe_when_only_action_exceeds_threshold(self, bounded_min_factory):
        # state 15 has a single action with pval 1.0, i.e. Pmin = 1: unsafe
        # (risk 1), which takes precedence over dangerous.
        result = bounded_min_factory.compute_mask(ShieldConfig(threshold=0.5))
        assert result.bitmask_by_state[15] == 0
        assert result.fallback_by_state[15] == 0b1
        assert result.is_unsafe(15)
        assert not result.is_dangerous(15)
        assert result.classify_state(15) == "unsafe"


class TestRelativeComparisonMaximize:
    def test_best_value_zero_and_threshold_leq_one_allows_all(self, bounded_max_factory):
        # state 0 pvals = [0.0, 0.0], p_best = 0.0: every action is allowed,
        # but the state itself is unsafe (Pmax = 0), so it is never "safe".
        result = bounded_max_factory.compute_mask(
            ShieldConfig(threshold=0.5, comparison="relative")
        )
        assert result.bitmask_by_state[0] == 0b11
        assert result.is_unsafe(0)
        assert not result.is_safe(0)

    def test_best_value_zero_and_threshold_gt_one_blocks_all(self, bounded_max_factory):
        result = bounded_max_factory.compute_mask(
            ShieldConfig(threshold=2.0, comparison="relative")
        )
        assert result.bitmask_by_state[0] == 0
        assert result.is_unsafe(0)
        assert not result.is_dangerous(0)

    def test_normal_relative_threshold(self, bounded_max_factory):
        # state 10 pvals = [0.144, 0.144, 0.648], p_best=0.648
        result = bounded_max_factory.compute_mask(
            ShieldConfig(threshold=0.9, comparison="relative")
        )
        assert result.bitmask_by_state[10] == 0b100


class TestRelativeComparisonMinimize:
    def test_best_value_one_and_threshold_geq_one_allows_all(self, bounded_min_factory):
        # state 15 pvals = [1.0], p_best = 1.0: the action is allowed, but the
        # state is unsafe (Pmin = 1).
        result = bounded_min_factory.compute_mask(
            ShieldConfig(threshold=1.0, comparison="relative")
        )
        assert result.bitmask_by_state[15] == 0b1
        assert result.is_unsafe(15)
        assert not result.is_safe(15)

    def test_best_value_one_and_threshold_lt_one_blocks_all(self, bounded_min_factory):
        result = bounded_min_factory.compute_mask(
            ShieldConfig(threshold=0.5, comparison="relative")
        )
        assert result.bitmask_by_state[15] == 0
        assert result.is_unsafe(15)
        assert not result.is_dangerous(15)

    def test_normal_relative_threshold(self, bounded_min_factory):
        # state 13 pvals = [0.0, 0.0, 0.6], p_best = 0.0 -> threshold * 0 = 0
        result = bounded_min_factory.compute_mask(
            ShieldConfig(threshold=0.5, comparison="relative")
        )
        assert result.bitmask_by_state[13] == 0b011


class TestActionLookupRemapping:
    def test_bits_keyed_by_global_index(self, sparse_factory):
        lookup = ActionLookup.from_dict(
            {"north": 0, "south": 1, "east": 2, "west": 3}
        )
        result = sparse_factory.build(
            ShieldConfig(threshold=0.0, action_lookup=lookup)
        )
        # state 0 only has 'south' and 'west' actions available locally
        allowed_global_bits = result.bitmask_by_state[0]
        assert allowed_global_bits & (1 << lookup.index("south"))
        assert allowed_global_bits & (1 << lookup.index("west"))
        assert not (allowed_global_bits & (1 << lookup.index("north")))
        assert not (allowed_global_bits & (1 << lookup.index("east")))


class TestUnknownComparisonMode:
    def test_unknown_comparison_raises_valueerror(self, sparse_factory):
        with pytest.raises(ValueError):
            sparse_factory.build(ShieldConfig(threshold=0.5, comparison="bogus"))


class TestSmgCoalitionFiltering:
    def test_coalition_states_restricted_to_ego_player(self, smg_factory):
        result = smg_factory.build(ShieldConfig(threshold=0.5))
        assert result.coalition_states == frozenset({1, 2, 3})
        assert result.is_coalition_state(0) is False
        assert result.is_coalition_state(1) is True


class TestDdEngineRestrictions:
    def test_relative_comparison_raises_valueerror(self, dd_factory):
        with pytest.raises(ValueError):
            dd_factory.build(ShieldConfig(threshold=0.5, comparison="relative"))

    def test_action_lookup_raises_valueerror(self, dd_factory):
        lookup = ActionLookup.from_dict({"north": 0})
        with pytest.raises(ValueError):
            dd_factory.build(ShieldConfig(threshold=0.5, action_lookup=lookup))

    def test_missing_choice_values_raises_runtimeerror(self, dd_factory):
        class _FakeCheckResult:
            has_choice_values = False

        # Simulate an already-available check result that (unexpectedly)
        # was not produced with choice values enabled.
        dd_factory._check_result = _FakeCheckResult()
        with pytest.raises(RuntimeError):
            dd_factory.compute_mask(ShieldConfig(threshold=0.5))
