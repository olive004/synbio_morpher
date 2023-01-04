import unittest


from tests.shared import five_circuits, mutate, simulate
from tests.shared import config as CONFIG
from src.srv.sequence_exploration.sequence_analysis import load_tabulated_info, b_tabulate_mutation_info


circuits, config, data_writer = five_circuits(CONFIG, data_writer=None)
circuits, config, data_writer = mutate(circuits, config, data_writer)
circuits, config, data_writer = simulate(circuits, config, data_writer)

info1 = b_tabulate_mutation_info(data_writer.ensemble_write_dir,
                                 data_writer=data_writer, experiment_config=config)
source_dirs = [data_writer.ensemble_write_dir]
info2 = load_tabulated_info(source_dirs)


class TestCircuit(unittest.TestCase):

    def test_interactions(self):
        pass

    def test_refcircuits(self):
        # make sure that each simulation used the correct reference circuit
        og_circuits = info1[info1["mutation_num"] == 0]

        ref_circuits = info1[info1["mutation_name"] == "ref_circuit"]

        self.assertEqual(ref_circuits["mutation_num"].unique()[0], 0,
                         f'The reference circuits should have no mutations.')
        self.assertEqual(ref_circuits["mutation_type"].unique()[0], '[]',
                         f'The reference circuits should have no mutations.')

        for diff in [
            "binding_rates_dissociation_max_interaction_diff_to_base_circuit",
            "binding_rates_dissociation_min_interaction_diff_to_base_circuit",
            "eqconstants_max_interaction_diff_to_base_circuit",
            "eqconstants_min_interaction_diff_to_base_circuit",
            "num_interacting_diff_to_base_circuit",
            "num_self_interacting_diff_to_base_circuit",
            "RMSE",
            "fold_change_diff_to_base_circuit",
            "overshoot_diff_to_base_circuit",
            "max_amount_diff_to_base_circuit",
            "min_amount_diff_to_base_circuit",
            "RMSE_diff_to_base_circuit",
            "steady_states_diff_to_base_circuit"]:
            if diff in ref_circuits.columns:
                self.assertEqual(ref_circuits[diff].unique()[0], 0,
                                f'The reference circuits should not be different to the base circuit since they are the base circuit.')
        for ratio in [
            "binding_rates_dissociation_max_interaction_ratio_from_mutation_to_base",
            "binding_rates_dissociation_min_interaction_ratio_from_mutation_to_base",
            "eqconstants_max_interaction_ratio_from_mutation_to_base",
            "eqconstants_min_interaction_ratio_from_mutation_to_base",
            "num_interacting_ratio_from_mutation_to_base",
            "num_self_interacting_ratio_from_mutation_to_base",
            "fold_change_ratio_from_mutation_to_base",
            "overshoot_ratio_from_mutation_to_base",
            "max_amount_ratio_from_mutation_to_base",
            "min_amount_ratio_from_mutation_to_base",
            "steady_states_ratio_from_mutation_to_base"]:
            if ratio in ref_circuits.columns:
                self.assertEqual(ref_circuits[ratio].unique()[0], 1,
                                f'The reference circuits should have a ratio to the base circuit since they are the base circuit.')


class TestCircuitInfo(unittest.TestCase):

    def test_tabulation(self):

        self.assertTrue((info1 == info2).all().all(),
                        f'Loaded info table incorrectly')
        self.assertEqual(info1, info2, f'Loaded info table incorrectly')


def main(args=None):
    t = TestCircuit()
    t.test_interactions()
    unittest.main()


if __name__ == '__main__':
    unittest.main()
