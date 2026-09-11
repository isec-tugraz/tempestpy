import pytest

from tempestpy.shielding.state_lookup import StateValuationLookup


class TestMakeKey:
    def test_make_key_orders_by_variables(self):
        key = StateValuationLookup.make_key({"y": 2, "x": 1}, variables=("x", "y"))
        assert key == (("x", 1), ("y", 2))

    def test_make_key_ignores_extra_keys(self):
        key = StateValuationLookup.make_key(
            {"x": 1, "y": 2, "z": 99}, variables=("x", "y")
        )
        assert key == (("x", 1), ("y", 2))

    def test_make_key_missing_variable_returns_none(self):
        key = StateValuationLookup.make_key({"x": 1}, variables=("x", "y"))
        assert key is None


class TestFromModel:
    @pytest.fixture
    def lookup(self, sparse_factory):
        return sparse_factory.state_lookup

    def test_variables_match_prism_program(self, lookup, sparse_factory):
        expected = tuple(
            v.name
            for v in sparse_factory.prism_program.get_variables(include_constants=False)
        )
        assert lookup.variables == expected

    def test_round_trip_state_id_and_values(self, lookup):
        for sid in range(len(lookup.state_id_to_values)):
            values = lookup.values_for_state_id(sid)
            assert lookup.get_state_id(values) == sid

    def test_has_state(self, lookup):
        values = lookup.values_for_state_id(0)
        assert lookup.has_state(values) is True
        assert lookup.has_state({"x": 999, "y": 999}) is False

    def test_get_key_matches_key_for_state_id(self, lookup):
        for sid in range(len(lookup.state_id_to_values)):
            values = lookup.values_for_state_id(sid)
            assert lookup.get_key(values) == lookup.key_for_state_id(sid)

    def test_get_state_id_missing_variable_raises_keyerror(self, lookup):
        with pytest.raises(KeyError):
            lookup.get_state_id({"x": 1})

    def test_get_state_id_unknown_valuation_raises_keyerror(self, lookup):
        with pytest.raises(KeyError):
            lookup.get_state_id({"x": 999, "y": 999})

    def test_values_for_state_id_out_of_range_raises_keyerror(self, lookup):
        with pytest.raises(KeyError):
            lookup.values_for_state_id(10 ** 6)

    def test_key_for_state_id_out_of_range_raises_keyerror(self, lookup):
        with pytest.raises(KeyError):
            lookup.key_for_state_id(10 ** 6)

    def test_no_prism_variables_raises_valueerror(self, sparse_factory):
        class _EmptyVarsProgram:
            def get_variables(self, include_constants=False):
                return ()

        with pytest.raises(ValueError):
            StateValuationLookup.from_model(sparse_factory.model, _EmptyVarsProgram())


class TestDuplicateValuationDetection:
    def test_duplicate_valuation_keys_raise_valueerror(self, monkeypatch, sparse_factory):
        duplicate_values = [{"x": 1, "y": 1}, {"x": 1, "y": 1}]

        monkeypatch.setattr(
            StateValuationLookup,
            "_extract_state_id_to_values",
            classmethod(lambda cls, model, prism_variables: (("x", "y"), duplicate_values)),
        )

        with pytest.raises(ValueError):
            StateValuationLookup.from_model(
                sparse_factory.model, sparse_factory.prism_program
            )
