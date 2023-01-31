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
        signal_onehot = 1 * (fake_species == fake_species[signal_idx])

        t0 = 0
        t1 = 1000
        dt = 1
        t = np.arange(t0, t1)

        data = [SignalFuncs.step_function_integrated(
            t, t1/2 + i, target=100 + i) for i in range(num_species)]
        ref_circuit_data = [SignalFuncs.step_function_integrated(
            t, t1/2 + i * 2, target=100 + i * 2) for i in range(num_species)]

        analytics = partial(generate_analytics, time=t, labels=fake_species,
                            signal_onehot=signal_onehot, ref_circuit_data=ref_circuit_data)(data)

        

        for c in [TEST_CONFIG, CONFIG]:
            circuits, config, data_writer, info = create_test_inputs(c)

            analytics_cols = get_true_names_analytics(
                candidate_cols=info.columns)


def main():

    t = TestAnalytics()
    t.test_interactions()


if __name__ == '__main__':
    unittest.main()
