import pytest
import stormpy.examples.files

from tempestpy.shielding import ShieldConfig, ShieldFactory


@pytest.fixture
def slipgrid_path():
    return stormpy.examples.files.prism_mdp_slipgrid


class TestSparseBuilderLabels:
    """The sparse model must contain what the property needs *and* everything
    declared in the program, so it stays usable beyond the one property."""

    def test_inline_expression_property_can_be_built(self, slipgrid_path):
        # "x=4 & y=4"-style expressions are not declared labels; they are only
        # built when the formula is handed to the builder.
        factory = ShieldFactory(slipgrid_path, property="Pmax=? [F x=4 & y=4]")
        mask = factory.build(ShieldConfig(threshold=0.9))
        assert len(mask) == int(factory.model.nr_states)

    def test_declared_labels_survive_when_formula_given(self, slipgrid_path):
        with_formula = ShieldFactory(slipgrid_path, property='Pmax=? [F "goal"]')
        no_formula = ShieldFactory(slipgrid_path)
        assert no_formula.model.labeling.get_labels() <= (
            with_formula.model.labeling.get_labels()
        )

    def test_declared_reward_models_survive_when_formula_given(self, slipgrid_path):
        with_formula = ShieldFactory(slipgrid_path, property='Pmax=? [F "goal"]')
        no_formula = ShieldFactory(slipgrid_path)
        assert set(no_formula.model.reward_models) <= set(
            with_formula.model.reward_models
        )
