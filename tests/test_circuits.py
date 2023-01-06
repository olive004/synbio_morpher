import unittest
import numpy as np
import pandas as pd
import logging


from tests.shared import five_circuits, mutate, simulate #, CONFIG, TEST_CONFIG
from tests.shared import TEST_CONFIG as CONFIG
from src.srv.sequence_exploration.sequence_analysis import load_tabulated_info, b_tabulate_mutation_info
from src.utils.results.analytics.analytics import get_true_names_analytics, DIFF_KEY, RATIO_KEY


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
        """ Make sure that each simulation used the correct reference circuit """

        ref_circuits = info1[info1["mutation_name"] == "ref_circuit"]

        self.assertEqual(len(ref_circuits.groupby('circuit_name').agg(
            str)), len(ref_circuits['circuit_name'].unique()))

        self.assertEqual(ref_circuits["mutation_num"].unique()[0], 0,
                         f'The reference circuits should have no mutations.')
        self.assertEqual(ref_circuits["mutation_type"].unique()[0], '[]',
                         f'The reference circuits should have no mutations.')

        analytics_cols = get_true_names_analytics(candidate_cols=info1.columns)
        diffs = info1.set_index(['circuit_name', 'sample_name', 'mutation_name']).groupby('mutation_name', group_keys=False)[analytics_cols].apply(
            lambda x: x - ref_circuits.set_index(['circuit_name', 'sample_name'])[analytics_cols]).reset_index()
        ratios = info1.set_index(['circuit_name', 'sample_name', 'mutation_name']).groupby('mutation_name', group_keys=False)[analytics_cols].apply(
            lambda x: x / ref_circuits.set_index(['circuit_name', 'sample_name'])[analytics_cols]).reset_index()

        # Make sure the reference circuits have no difference to their original values
        self.assertTrue(
            all([(diff == 0) or (np.isnan(diff)) for diff in sorted(
                diffs[diffs['mutation_name'] == 'ref_circuit'][analytics_cols].agg('unique'))]),
            'The reference circuit should have a difference of 0 or nan to itself.')

        self.assertTrue(
            all([(ratio == 1) or (np.isnan(ratio)) for ratio in sorted(
                ratios[ratios['mutation_name'] == 'ref_circuit'][analytics_cols].agg('unique'))]),
            'The reference circuit should have a ratio of 1 or nan to itself.')

        for col in analytics_cols:
            if DIFF_KEY in col:
                self.assertEqual(ref_circuits[col].unique()[0], 0,
                                 f'The reference circuits should not be different to the base circuit since they are the base circuit.')
            elif RATIO_KEY in col:
                self.assertTrue((ref_circuits[col].unique()[0] == 1) or (ref_circuits[col].unique()[0] == np.inf),
                                f'The reference circuits should have a ratio of one or infinity to the base circuit since they are the base circuit.')
            elif col + DIFF_KEY in info1.columns:
                self.assertTrue(all(diffs[col] == info1[col + DIFF_KEY]),
                                'The difference between the reference circuit and its subcircuits should match between the computed and the tabulated numbers.')
            elif col + RATIO_KEY in info1.columns:
                self.assertTrue(all(ratios[col] == info1[col + RATIO_KEY]),
                                'The ratio between the reference circuit and its subcircuits should match between the computed and the tabulated numbers.')
            else:
                logging.warning(
                    f'Column {col} should be in the table or exist there as {col + DIFF_KEY} or {col + RATIO_KEY} - columns are {info1.columns}')
        
    # def test_numerical_results(self):



class TestCircuitInfo(unittest.TestCase):

    def test_tabulation(self):

        self.assertTrue((info1 == info2).all().all(),
                        f'Loaded info table incorrectly')
        self.assertTrue(all(info1 == info2), f'Loaded info table incorrectly')


def main(args=None):
    t = TestCircuit()
    t.test_interactions()
    unittest.main()


if __name__ == '__main__':
    unittest.main()
