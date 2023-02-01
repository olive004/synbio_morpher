import unittest
import numpy as np
import pandas as pd
from functools import partial


from tests.shared import create_test_inputs, CONFIG, TEST_CONFIG
from src.utils.results.analytics.naming import get_true_names_analytics, DIFF_KEY, RATIO_KEY
from src.utils.results.analytics.timeseries import generate_analytics
from src.utils.signal.signals_new import SignalFuncs


class TestAnalytics(unittest.TestCase):

    def test_interactions(self):

        num_species = 9
        fake_species = [str(i) for i in np.arange(num_species)]
        signal_idx = 0
        signal_onehot = 1 * np.array([f == fake_species[signal_idx] for f in fake_species])

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
        self.assertEqual(analytics[f'precision_wrt_species-{signal_idx}'][0], 1)
        self.assertEqual(analytics[f'sensitivity_wrt_species-{signal_idx}'][0], 1)
        for k in analytics.keys():
            if RATIO_KEY in k and 'deriv' not in k:
                self.assertTrue((analytics[k][0].astype(np.float16)[0] == 1) | (analytics[k][0].astype(np.float16)[0] == np.inf))

        # Choice analytics
        [self.assertEqual(analytics['fold_change'].astype(np.float16)[i], (target + baseline + i)/baseline) for i in range(num_species)]

        for c in [TEST_CONFIG, CONFIG]:
            circuits, config, data_writer, info = create_test_inputs(c)

            analytics_cols = get_true_names_analytics(
                candidate_cols=info.columns)


def main():

    t = TestAnalytics()
    t.test_interactions()


if __name__ == '__main__':
    unittest.main()
