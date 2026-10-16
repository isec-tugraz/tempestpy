import pytest

from tempestpy.shielding import ShieldConfig
from tempestpy.shielding import export as export_mod


@pytest.fixture
def built(bounded_max_factory):
    mask = bounded_max_factory.compute_mask(ShieldConfig(threshold=0.5))
    return bounded_max_factory, mask


class TestIdKeyedExports:
    def test_state_id_to_bitmask(self, built):
        factory, mask = built
        out = export_mod.state_id_to_bitmask(mask)
        assert out == mask.to_bitmask_dict()
        assert out[10] == 0b100

    def test_state_id_to_allowed_action_indices(self, built):
        factory, mask = built
        out = export_mod.state_id_to_allowed_action_indices(factory.model, mask)
        assert out[10] == [2]

    def test_state_id_to_allowed_action_labels(self, built):
        factory, mask = built
        out = export_mod.state_id_to_allowed_action_labels(factory.model, mask)
        state10 = list(factory.model.states)[10]
        expected_label = frozenset(state10.actions[2].labels)
        assert out[10] == [expected_label]

    def test_allowed_state_action_pairs(self, built):
        factory, mask = built
        pairs = export_mod.allowed_state_action_pairs(factory.model, mask)
        assert (10, 2) in pairs
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)


class TestValuationKeyedExports:
    def test_valuation_to_bitmask(self, built):
        factory, mask = built
        out = export_mod.valuation_to_bitmask(factory.model, mask, factory.state_lookup)
        key10 = factory.state_lookup.key_for_state_id(10)
        assert out[key10] == 0b100

    def test_valuation_to_allowed_action_labels(self, built):
        factory, mask = built
        out = export_mod.valuation_to_allowed_action_labels(
            factory.model, mask, factory.state_lookup
        )
        key10 = factory.state_lookup.key_for_state_id(10)
        state10 = list(factory.model.states)[10]
        assert out[key10] == [frozenset(state10.actions[2].labels)]

    def test_valuation_to_allowed_action_labels_with_values_mapping(self, built):
        factory, mask = built
        values = {
            (sid, i): float(v)
            for sid in range(int(factory.model.nr_states))
            for i, v in enumerate(factory.get_choice_values_for_state(sid))
        }
        out = export_mod.valuation_to_allowed_action_labels_with_values(
            factory.model, mask, factory.state_lookup, values
        )
        key10 = factory.state_lookup.key_for_state_id(10)
        state10 = list(factory.model.states)[10]
        label10 = frozenset(state10.actions[2].labels)
        assert out[key10][label10] == pytest.approx(0.648)

    def test_valuation_to_allowed_action_labels_with_values_callable(self, built):
        factory, mask = built
        out = export_mod.valuation_to_allowed_action_labels_with_values(
            factory.model,
            mask,
            factory.state_lookup,
            lambda sid, i: float(factory.get_choice_values_for_state(sid)[i]),
        )
        key10 = factory.state_lookup.key_for_state_id(10)
        state10 = list(factory.model.states)[10]
        label10 = frozenset(state10.actions[2].labels)
        assert out[key10][label10] == pytest.approx(0.648)

    def test_valuation_to_allowed_action_ids(self, built):
        factory, mask = built
        out = export_mod.valuation_to_allowed_action_ids(
            factory.model, mask, factory.state_lookup
        )
        key10 = factory.state_lookup.key_for_state_id(10)
        state10 = list(factory.model.states)[10]
        assert out[key10] == [int(state10.actions[2].id)]

    def test_valuation_to_action_label_probability_includes_blocked_actions(self, built):
        factory, mask = built
        out = export_mod.valuation_to_action_label_probability(
            factory.model,
            factory.state_lookup,
            lambda sid, i: float(factory.get_choice_values_for_state(sid)[i]),
        )
        key10 = factory.state_lookup.key_for_state_id(10)
        # all 3 actions of state 10 present, not just the allowed one
        assert len(out[key10]) == 3


class TestMaskMetadata:
    def test_mask_metadata_contains_summary(self, built):
        factory, mask = built
        meta = export_mod.mask_metadata(mask)
        assert meta["nr_states"] == len(mask.bitmask_by_state)
        assert "best_value_by_state_summary" in meta
        summary = meta["best_value_by_state_summary"]
        assert 0.0 <= summary["min"] <= summary["mean"] <= summary["max"] <= 1.0


class TestPretty:
    def test_truncates_at_max_states(self, built):
        factory, mask = built
        text = export_mod.pretty(factory.model, mask, factory.state_lookup, max_states=2)
        assert "more states" in text

    def test_invalid_action_ref_raises_valueerror(self, built):
        factory, mask = built
        with pytest.raises(ValueError):
            export_mod.pretty(
                factory.model, mask, factory.state_lookup, action_ref="bogus"
            )

    def test_show_valuations_toggle(self, built):
        factory, mask = built
        with_vals = export_mod.pretty(
            factory.model, mask, factory.state_lookup, max_states=1, show_valuations=True
        )
        without_vals = export_mod.pretty(
            factory.model, mask, factory.state_lookup, max_states=1, show_valuations=False
        )
        assert with_vals != without_vals
        assert "state 0 (" in with_vals
        assert "state 0:" in without_vals


class TestStormFormat:
    def test_storm_format_header_and_rows(self, built):
        factory, mask = built
        text = export_mod.storm_format(factory, mask)
        assert "Pre-Safety-Shield with absolute comparison" in text
        assert "model state:" in text
        # one row per model state
        assert text.count("\n") >= int(factory.model.nr_states)
