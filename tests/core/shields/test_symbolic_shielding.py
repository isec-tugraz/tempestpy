from stormpy.shielding import ShieldConfig, ShieldFactory

from helpers.helper import get_example_path


def _write_known_shield_mdp(tmp_path):
    path = tmp_path / "known_symbolic_shield.nm"
    path.write_text(
        """
mdp

module known_symbolic_shield
    s : [0..4] init 0;

    [to_goal] s=0 -> (s'=3);
    [to_bad]  s=0 -> (s'=4);
    [to_mid]  s=0 -> (s'=1);

    [finish]  s=1 -> (s'=3);
    [fail]    s=1 -> (s'=4);

    [stay_goal] s=3 -> (s'=3);
    [stay_bad]  s=4 -> (s'=4);
endmodule

label "goal" = s=3;
""",
        encoding="utf-8",
    )
    return path


def _symbolic_action_labels(prism_program) -> list[str]:
    return [prism_program.get_action_name(action_index) for action_index in range(1, len(prism_program.actions))]


def _sparse_allowed_labels(factory: ShieldFactory, shield, sid: int) -> set[str]:
    mask = int(shield.bitmask_by_state[sid])
    labels: set[str] = set()
    for local_index, action in enumerate(factory.model.states[sid].actions):
        if mask & (1 << local_index):
            labels.update(action.labels)
    return labels


def _symbolic_allowed_labels(factory: ShieldFactory, shield, valuation: dict[str, int]) -> set[str]:
    mask = int(shield.bitmask_by_state.query(valuation))
    action_labels = _symbolic_action_labels(factory.prism_program)
    labels: set[str] = set()
    for bit, label in enumerate(action_labels):
        if mask & (1 << bit):
            labels.add(label)

    unknown_bits = mask >> len(action_labels)
    assert unknown_bits == 0, f"Symbolic mask contains bits without action labels: {bin(mask)}"
    return labels


class TestSymbolicShielding:
    def test_symbolic_shielding_until_known_model(self, tmp_path):
        model_path = _write_known_shield_mdp(tmp_path)
        factory = ShieldFactory(
            str(model_path),
            property='Pmax=? [true U "goal"]',
            engine="dd",
        )
        shield = factory.build(ShieldConfig(threshold=1.0))

        expected = {
            0: {"to_goal", "to_mid"},
            1: {"finish"},
            4: set(),
        }

        for state, expected_labels in expected.items():
            assert _symbolic_allowed_labels(factory, shield, {"s": state}) == expected_labels

    def test_symbolic_shielding_next_known_model(self, tmp_path):
        model_path = _write_known_shield_mdp(tmp_path)
        factory = ShieldFactory(
            str(model_path),
            property='Pmax=? [X "goal"]',
            engine="dd",
        )
        shield = factory.build(ShieldConfig(threshold=1.0))

        expected = {
            0: {"to_goal"},
            1: {"finish"},
            4: set(),
        }

        for state, expected_labels in expected.items():
            assert _symbolic_allowed_labels(factory, shield, {"s": state}) == expected_labels

    def test_symbolic_shielding_bounded_until_known_model(self, tmp_path):
        model_path = _write_known_shield_mdp(tmp_path)
        factory = ShieldFactory(
            str(model_path),
            property='Pmax=? [true U<=1 "goal"]',
            engine="dd",
        )
        shield = factory.build(ShieldConfig(threshold=1.0))

        expected = {
            0: {"to_goal"},
            1: {"finish"},
            4: set(),
        }

        for state, expected_labels in expected.items():
            assert _symbolic_allowed_labels(factory, shield, {"s": state}) == expected_labels

    def test_symbolic_shielding_until_simple_model(self):
        example_mdp = get_example_path("mdp", "maze_2.nm")
        prop = 'Pmax=? [true U "goal"]'
        config = ShieldConfig(threshold=0.5)

        factory_sparse = ShieldFactory(example_mdp, property=prop, engine="sparse")
        shield_sparse = factory_sparse.build(config)
        factory_symbolic = ShieldFactory(example_mdp, property=prop, engine="dd")
        shield_symbolic = factory_symbolic.build(config)

        valuations: list[dict[str, int]] = [
            {"s": 0},
            {"s": 1},
            {"s": 6},
            {"s": 9},
            {"s": 13},
        ]

        for valuation in valuations:
            sparse_sid = factory_sparse.get_state_id(valuation)
            sparse_labels = _sparse_allowed_labels(factory_sparse, shield_sparse, sparse_sid)
            symbolic_labels = _symbolic_allowed_labels(factory_symbolic, shield_symbolic, valuation)
            assert sparse_labels == symbolic_labels

    def test_symbolic_shielding_until_coin2(self):
        example_mdp = get_example_path("mdp", "lunar_lander_angle_shield.prism")
        prop = 'Pmax=? [G ("safe")]'
        # finished" & "all_coins_equal_1
        config = ShieldConfig(threshold=1)

        factory_symbolic = ShieldFactory(example_mdp,
                                         property=prop,
                                         engine="dd",
                                         constants={"GRAN": 20})
        shield_symbolic = factory_symbolic.build(config)
        factory_sparse = ShieldFactory(example_mdp,
                                       property=prop,
                                       engine="sparse",
                                       constants={"GRAN": 20})
        shield_sparse = factory_sparse.build(config)

        valuations = [
            factory_sparse.state_lookup.values_for_state_id(sid)
            for sid in range(min(10000, factory_sparse.model.nr_states))
        ]
        
        for valuation in valuations:
            print(f"Valuation: {valuation}")
            print("Symbolic shield: %f", bin(int(shield_symbolic.bitmask_by_state.query(valuation))))
            sparse_sid = factory_sparse.get_state_id(valuation)
            sparse_labels = _sparse_allowed_labels(factory_sparse, shield_sparse, sparse_sid)
            symbolic_labels = _symbolic_allowed_labels(factory_symbolic, shield_symbolic, valuation)
            assert sparse_labels == symbolic_labels

    def test_symbolic_shielding_next_coin2(self):
        example_mdp = get_example_path("mdp", "coin2-2.nm")
        prop = 'Pmin=? [ X "all_coins_equal_1"]'
        config = ShieldConfig(threshold=0.5)

        factory_symbolic = ShieldFactory(example_mdp,
                                         property=prop,
                                         engine="dd")
        shield_symbolic = factory_symbolic.build(config)
        factory_sparse = ShieldFactory(example_mdp,
                                       property=prop,
                                       engine="sparse",)
        shield_sparse = factory_sparse.build(config)

        valuations = [
            factory_sparse.state_lookup.values_for_state_id(sid)
            for sid in range(min(100, factory_sparse.model.nr_states))
        ]
        
        for valuation in valuations:
            print(f"Valuation: {valuation}")
            print("Symbolic shield: %f", bin(int(shield_symbolic.bitmask_by_state.query(valuation))))
            sparse_sid = factory_sparse.get_state_id(valuation)
            sparse_labels = _sparse_allowed_labels(factory_sparse, shield_sparse, sparse_sid)
            symbolic_labels = _symbolic_allowed_labels(factory_symbolic, shield_symbolic, valuation)
            assert sparse_labels == symbolic_labels

    def test_symbolic_shielding_bounded_until_coin2(self):
        example_mdp = get_example_path("mdp", "coin2-2.nm")
        prop = 'Pmin=? [ true U<=8 "all_coins_equal_1"]'
        config = ShieldConfig(threshold=0.5)

        factory_symbolic = ShieldFactory(example_mdp,
                                         property=prop,
                                         engine="dd")
        shield_symbolic = factory_symbolic.build(config)
        factory_sparse = ShieldFactory(example_mdp,
                                       property=prop,
                                       engine="sparse",)
        shield_sparse = factory_sparse.build(config)

        valuations = [
            factory_sparse.state_lookup.values_for_state_id(sid)
            for sid in range(min(100, factory_sparse.model.nr_states))
        ]
        
        for valuation in valuations:
            sparse_sid = factory_sparse.get_state_id(valuation)
            sparse_labels = _sparse_allowed_labels(factory_sparse, shield_sparse, sparse_sid)
            symbolic_labels = _symbolic_allowed_labels(factory_symbolic, shield_symbolic, valuation)
            assert sparse_labels == symbolic_labels
