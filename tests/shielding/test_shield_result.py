import pytest

from tempestpy.shielding.masking import ShieldConfig
from tempestpy.shielding.shield_result import ExplicitShieldResult, SymbolicShieldResult

from conftest import make_explicit_result


class TestExplicitShieldResultClassification:
    def test_safe_state(self):
        # all 2 actions allowed, no fallback
        result = make_explicit_result(bitmask_by_state=[0b11], nr_actions_by_state=[2])
        assert result.is_safe(0)
        assert not result.is_critical(0)
        assert not result.is_dangerous(0)
        assert result.classify_state(0) == "safe"

    def test_critical_state(self):
        # 1 of 2 actions allowed
        result = make_explicit_result(bitmask_by_state=[0b01], nr_actions_by_state=[2])
        assert result.is_critical(0)
        assert not result.is_safe(0)
        assert not result.is_dangerous(0)
        assert result.classify_state(0) == "critical"

    def test_dangerous_state(self):
        # no action allowed, fallback used
        result = make_explicit_result(
            bitmask_by_state=[0b00], fallback_by_state=[0b01], nr_actions_by_state=[2]
        )
        assert result.is_dangerous(0)
        assert not result.is_safe(0)
        assert not result.is_critical(0)
        assert result.classify_state(0) == "dangerous"

    def test_states_by_class(self):
        result = make_explicit_result(
            bitmask_by_state=[0b11, 0b01, 0b00],
            fallback_by_state=[0, 0, 0b01],
            nr_actions_by_state=[2, 2, 2],
        )
        classes = result.states_by_class()
        assert classes["safe"] == [0]
        assert classes["critical"] == [1]
        assert classes["dangerous"] == [2]
        assert result.safe_states() == [0]
        assert result.critical_states() == [1]
        assert result.dangerous_states() == [2]

    def test_len(self):
        result = make_explicit_result(bitmask_by_state=[0b1, 0b1, 0b1])
        assert len(result) == 3

    def test_to_bitmask_dict(self):
        result = make_explicit_result(bitmask_by_state=[0b10, 0b01])
        assert result.to_bitmask_dict() == {0: 0b10, 1: 0b01}


class TestExplicitShieldResultQueries:
    def test_query_mask_uses_bitmask_when_nonzero(self):
        result = make_explicit_result(bitmask_by_state=[0b11], fallback_by_state=[0b01])
        assert result.query_mask(0) == 0b11

    def test_query_mask_falls_back_when_bitmask_zero(self):
        result = make_explicit_result(bitmask_by_state=[0b00], fallback_by_state=[0b01])
        assert result.query_mask(0) == 0b01

    def test_is_action_allowed(self):
        result = make_explicit_result(bitmask_by_state=[0b10])
        assert result.is_action_allowed(0, 1) is True
        assert result.is_action_allowed(0, 0) is False

    def test_query_post_allows_permitted_action(self):
        result = make_explicit_result(bitmask_by_state=[0b11])
        assert result.query_post(0, 1) == 1

    def test_query_post_calls_selector_when_blocked(self):
        calls = []

        def selector(state, agent_action, **kwargs):
            calls.append((state, agent_action))
            return 7

        config = ShieldConfig(post_selector=selector)
        result = make_explicit_result(bitmask_by_state=[0b01], config=config)
        assert result.query_post(0, 1) == 7
        assert calls == [(0, 1)]

    def test_query_post_without_selector_raises_typeerror(self):
        result = make_explicit_result(bitmask_by_state=[0b01])
        with pytest.raises(TypeError):
            result.query_post(0, 1)

    def test_query_action_is_alias_for_query_post(self):
        result = make_explicit_result(bitmask_by_state=[0b11])
        assert result.query_action(0, 0) == result.query_post(0, 0)


class TestCoalitionStates:
    def test_no_coalition_states_always_true(self):
        result = make_explicit_result(bitmask_by_state=[0b1], coalition_states=None)
        assert result.is_coalition_state(0) is True
        assert result.is_coalition_state(42) is True

    def test_coalition_states_membership(self):
        result = make_explicit_result(
            bitmask_by_state=[0b1, 0b1], coalition_states=frozenset({0})
        )
        assert result.is_coalition_state(0) is True
        assert result.is_coalition_state(1) is False


class _StubAdd:
    def __init__(self, values):
        self._values = values

    def query(self, valuation):
        key = tuple(sorted(valuation.items()))
        return self._values[key]


class TestSymbolicShieldResult:
    def _make(self, bitmask_values, fallback_values=None, nr_actions=0):
        bitmask_add = _StubAdd(bitmask_values)
        fallback_add = _StubAdd(fallback_values) if fallback_values is not None else None
        return SymbolicShieldResult(
            config=ShieldConfig(),
            bitmask_add=bitmask_add,
            fallback_add=fallback_add,
            nr_actions=nr_actions,
        )

    def test_get_bitmask_queries_add(self):
        result = self._make({(("x", 1),): 0b101})
        assert result.query_mask({"x": 1}) == 0b101

    def test_fallback_defaults_to_zero_without_add(self):
        result = self._make({(("x", 1),): 0b000})
        assert result._get_fallback({"x": 1}) == 0

    def test_fallback_used_when_bitmask_zero(self):
        result = self._make(
            {(("x", 1),): 0b000}, fallback_values={(("x", 1),): 0b010}
        )
        assert result.query_mask({"x": 1}) == 0b010

    def test_nr_actions_uses_explicit_value(self):
        result = self._make({(("x", 1),): 0b011}, nr_actions=5)
        assert result._get_nr_actions({"x": 1}) == 5

    def test_nr_actions_inferred_from_bitlength(self):
        result = self._make({(("x", 1),): 0b0101}, nr_actions=0)
        assert result._get_nr_actions({"x": 1}) == (0b0101).bit_length()

    def test_to_bitmask_dict_raises_typeerror(self):
        result = self._make({(("x", 1),): 0})
        with pytest.raises(TypeError):
            result.to_bitmask_dict()

    def test_compatibility_properties(self):
        result = self._make({(("x", 1),): 0}, fallback_values={(("x", 1),): 0b001})
        assert result.bitmask_by_state is result.bitmask_add
        assert result.fallback_by_state is result.fallback_add
        assert result.fallback_by_state.query({"x": 1}) == 0b001
        assert result.nr_actions_by_state == []
        assert result.best_value_by_state is None
