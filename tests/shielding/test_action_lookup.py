import pytest

from tempestpy.shielding.action_lookup import ActionLookup


class TestActionLookupConstruction:
    def test_from_dict_basic(self):
        lookup = ActionLookup.from_dict({"north": 0, "south": 1, "east": 2, "west": 2})
        assert lookup.nr_actions == 3
        assert lookup.label_to_index == {"north": 0, "south": 1, "east": 2, "west": 2}
        assert sorted(lookup.index_to_labels[2]) == ["east", "west"]

    def test_empty_mapping_raises(self):
        with pytest.raises(ValueError):
            ActionLookup.from_dict({})

    def test_non_contiguous_indices_raise(self):
        with pytest.raises(ValueError):
            ActionLookup.from_dict({"a": 0, "b": 2})

    def test_indices_not_starting_at_zero_raise(self):
        with pytest.raises(ValueError):
            ActionLookup.from_dict({"a": 1, "b": 2})


class TestActionLookupQueries:
    @pytest.fixture
    def lookup(self):
        return ActionLookup.from_dict({"north": 0, "south": 1, "slow_east": 2, "fast_east": 2})

    def test_index(self, lookup):
        assert lookup.index("north") == 0
        assert lookup.index("slow_east") == 2

    def test_index_missing_label_raises_keyerror(self, lookup):
        with pytest.raises(KeyError):
            lookup.index("unknown")

    def test_labels(self, lookup):
        assert lookup.labels(0) == ["north"]
        assert sorted(lookup.labels(2)) == ["fast_east", "slow_east"]

    def test_labels_out_of_range_raises_keyerror(self, lookup):
        with pytest.raises(KeyError):
            lookup.labels(99)

    def test_has_label(self, lookup):
        assert lookup.has_label("north") is True
        assert lookup.has_label("unknown") is False

    def test_has_index(self, lookup):
        assert lookup.has_index(0) is True
        assert lookup.has_index(99) is False

    def test_len(self, lookup):
        assert len(lookup) == 3

    def test_repr(self, lookup):
        text = repr(lookup)
        assert "nr_actions=3" in text
        assert "north" in text


class TestActionLookupValidateModelLabels:
    def test_validate_model_labels_passes(self, sparse_factory):
        lookup = ActionLookup.from_dict(
            {"north": 0, "south": 1, "east": 2, "west": 3}
        )
        lookup.validate_model_labels(sparse_factory.model)

    def test_validate_model_labels_missing_raises(self, sparse_factory):
        lookup = ActionLookup.from_dict({"north": 0, "south": 1})
        with pytest.raises(ValueError):
            lookup.validate_model_labels(sparse_factory.model)
