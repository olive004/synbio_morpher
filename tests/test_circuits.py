import unittest


from tests.shared import five_circuits, mutate, simulate # CONFIG
from tests.shared import TEST_CONFIG as CONFIG
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

        ref_circuits = info1[info1["mutation_name"] == "ref_circuit"]

        self.assertEqual(ref_circuits["mutation_num"].unique()[0], 0,
                         f'The reference circuits should have no mutations.')
        self.assertEqual(ref_circuits["mutation_type"].unique()[0], '[]',
                         f'The reference circuits should have no mutations.')

        for col in ref_circuits.columns:
            if '_diff_' in col:
                self.assertEqual(ref_circuits[col].unique()[0], 0,
                                 f'The reference circuits should not be different to the base circuit since they are the base circuit.')
            elif '_ratio_' in col:
                self.assertEqual(ref_circuits[col].unique()[0], 1,
                                 f'The reference circuits should have a ratio to the base circuit since they are the base circuit.')

        for circ in ref_circuits['circuit_name'].unique():
            curr_circuits = info1[info1['circuit_name'] == circ]
            curr_circuits 

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
