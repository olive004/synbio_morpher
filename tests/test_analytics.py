import unittest
import numpy as np
import pandas as pd
from functools import partial
from copy import deepcopy


from tests.shared import create_test_inputs, CONFIG, TEST_CONFIG
from src.utils.results.analytics.naming import get_true_names_analytics, DIFF_KEY, RATIO_KEY
from src.utils.results.analytics.timeseries import generate_analytics
from src.utils.signal.signals_new import SignalFuncs


class TestAnalytics(unittest.TestCase):

    def test_interactions(self):

        num_species = 9
        fake_species = [str(i) for i in np.arange(num_species)]
        signal_idx = 0
        signal_onehot = 1 * \
            np.array([f == fake_species[signal_idx] for f in fake_species])

        t0 = 0
        t1 = 100
        dt = 1
        t = np.arange(t0, t1)
        target = 100
        baseline = 10

        data = np.asarray([SignalFuncs.step_function_integrated(
            t, t1/2 + i, dt=dt, target=target + i) for i in range(num_species)]) + baseline
        ref_circuit_data = np.asarray([SignalFuncs.step_function_integrated(
            t, t1/2 + i * 2, dt=dt, target=target + i * 2) for i in range(num_species)]) + baseline

        analytics = {k: np.array(v) for k, v in
                     partial(generate_analytics, time=t, labels=fake_species,
                             signal_onehot=signal_onehot, ref_circuit_data=ref_circuit_data)(data).items()
                     }

        # Signal in analytics
        self.assertEqual(
            analytics[f'precision_wrt_species-{signal_idx}'][0], 1)
        self.assertEqual(
            analytics[f'sensitivity_wrt_species-{signal_idx}'][0], 1)
        for k in analytics.keys():
            if RATIO_KEY in k and 'deriv' not in k:
                self.assertTrue((analytics[k][0].astype(np.float16)[0] == 1) | (
                    analytics[k][0].astype(np.float16)[0] == np.inf))

        # Choice analytics
        [self.assertEqual(analytics['fold_change'].astype(np.float16)[
                          i], (target + baseline + i)/baseline) for i in range(num_species)]

        for n, c in zip(['t', 'c'], [TEST_CONFIG, CONFIG]):
            circuits, config, data_writer, info = create_test_inputs(
                deepcopy(c))

            analytics_cols = get_true_names_analytics(
                candidate_cols=info.columns)

            if n == 't':
                self.assertEqual(
                    np.round(info['steady_states'].iloc[0], 4), 2.09)
                self.assertEqual(
                    np.round(info['steady_states'].iloc[4], 4), 0.5015)

            for s in info['sample_name'].unique():
                curr = info[info['sample_name'] == s]

                # Test that signal ratio is always 1 or inf
                for k in [a for a in analytics_cols if RATIO_KEY in a]:
                    if s in c['signal']['inputs']:
                        self.assertTrue(all(curr[k].astype(np.float16) == 1) | all(
                            curr[k].astype(np.float16) == np.inf))

                # Test that if the max_amount ratio was 1, that steady states did not change



            # Assert that steady states are correct
            info_ref = info[info['mutation_name'] == 'ref_circuit']
            true_steady_states = [
                2.09,
                0.5015,
                0.5015,
                2.049,
                0.4844,
                0.704,
                2.08,
                0.4956,
                0.565,
                2.21,
                0.6655,
                0.674,
                19.22,
                1.86,
                1.531
            ]
            for i in range(len(true_steady_states)):
                self.assertEqual(info_ref['steady_states'].astype(np.float16).iloc[i], np.float16(true_steady_states[i]))

            # Test that columns that should be positive are positive:
            #   fold_change	overshoot	max_amount	min_amount	RMSE	steady_states
            #   response_time_wrt_species-6	precision_wrt_species-6	sensitivity_wrt_species-6
            positive_cols = [
                'fold_change',
                'overshoot',
                'max_amount',
                'min_amount',
                'RMSE',
                'steady_states',
                'response_time_wrt_species-6',
                'precision_wrt_species-6',
                'sensitivity_wrt_species-6'
            ]

            # Plot response time
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.barplot(info, x='response_time_wrt_species-6',
                        hue='circuit_name')
            # plt.plot(info['response_time_wrt_species-6'])
            plt.savefig('test.svg')


def main():

    t = TestAnalytics()
    t.test_interactions()


if __name__ == '__main__':
    unittest.main()
