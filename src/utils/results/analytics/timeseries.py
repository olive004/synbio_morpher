import logging
import numpy as np
from typing import List


class Timeseries():
    def __init__(self, data, time=None) -> None:
        self.data = data
        if data is None:
            self.time = None
        else:
            self.time = np.arange(np.shape(self.data)[
                                  1]) if time is None else time
        self.num_dtype = np.float32

    def get_steady_state(self):
        """ Last 5% of data considered steady state """
        stability_threshold = 0.01
        final_deriv = np.average(
            self.get_derivative(self.data)[:, :-2])
        is_steady_state_reached = final_deriv < stability_threshold
        steady_states = np.expand_dims(
            self.data[:, -1], axis=1).astype(np.float32)
        return steady_states, is_steady_state_reached, final_deriv

    def fold_change(self):
        division_matrix = np.divide(
            self.data[:, -1].clip(1), self.data[:, 0].clip(1))
        if np.ndim(division_matrix) == 1:
            return np.expand_dims(division_matrix, axis=1)
        else:
            return division_matrix

    def get_derivative(self, data):
        deriv = np.gradient(data)[1]
        return deriv  # get column derivative

    def get_overshoot(self, steady_states):
        return np.expand_dims(np.max(self.data, axis=1), axis=1) - steady_states

    def calculate_precision(self, output_diff, starting_states, signal_diff, signal_start) -> np.ndarray:
        precision = np.absolute(np.divide(
            output_diff / starting_states,
            signal_diff / signal_start
        )).astype(self.num_dtype)
        return np.divide(1, precision)

    def get_precision(self, steady_states, signal_idx: int, ignore_denominator: bool = False):
        if signal_idx is None:
            return None
        starting_states = np.expand_dims(self.data[:, 0], axis=1)
        signal_start = self.data[signal_idx, 0]
        signal_end = self.data[signal_idx, -1]

        """ IF YOU IGNORE THE DENOMINATOR THE DIVIDE BY ZERO GOES AWAY AND YOU GET UNSCALED PRECISION """
        signal_diff = signal_end - signal_start
        output_diff = steady_states - starting_states

        if ignore_denominator:
            if signal_diff == 0:
                return np.zeros_like(output_diff)
            if signal_start == 0:
                signal_start = 1
            if any(starting_states == 0):
                starting_states = 1
            return self.calculate_precision(output_diff, starting_states, signal_diff, signal_start)
        return self.calculate_precision(output_diff, starting_states, signal_diff, signal_start)

    def calculate_sensitivity(self, output_diff, starting_states, signal_diff, signal_low) -> np.ndarray:
        return np.expand_dims(np.absolute(np.divide(
            output_diff / starting_states,
            signal_diff / signal_low
        )).astype(self.num_dtype), axis=1)

    def get_sensitivity(self, signal_idx: int, ignore_denominator: bool = False):
        if signal_idx is None:
            return None
        starting_states = self.data[:, 0]
        peaks = np.max(self.data, axis=1)
        signal_low = np.min(self.data[signal_idx, :])
        signal_high = np.max(self.data[signal_idx, :])

        """ IF YOU IGNORE THE DENOMINATOR THE DIVIDE BY ZERO GOES AWAY AND YOU GET UNSCALED SENSITIVITY """
        output_diff = peaks - starting_states
        signal_diff = signal_high - signal_low

        if signal_diff == 0:
            logging.warning(
                f'Signal difference was 0 from {signal_high} and {signal_low}.')
            return self.num_dtype(0)
        elif ignore_denominator and signal_low == 0 or any(starting_states == 0):
            # return np.expand_dims(np.absolute(np.divide(
            #     output_diff,
            #     signal_diff
            # )).astype(self.num_dtype), axis=1)
            return self.calculate_sensitivity(output_diff, np.ones_like(output_diff), signal_diff, np.ones_like(signal_diff))
        return self.calculate_sensitivity(output_diff, starting_states, signal_diff, signal_low)

    def get_rmse(self, ref_circuit_signal):
        if ref_circuit_signal is None:
            return np.zeros(shape=(np.shape(self.data)[0], 1))
        data = self.data - ref_circuit_signal
        rmse = np.sqrt(
            np.sum(np.divide(np.power(data, 2), len(self.data)), axis=1))
        return rmse

    def get_step_response_times(self, steady_states, deriv1, signal_idx: np.ndarray):
        time = self.time * np.ones_like(steady_states)
        margin = 0.05
        cond_out = (self.data > (steady_states + steady_states * margin)
                    ) | (self.data < (steady_states - steady_states * margin))

        # Get zero derivative within margin
        fmargin = 0.001
        fm = np.expand_dims(np.max(deriv1, axis=1) * fmargin, axis=1)
        zd = (deriv1 < fm) & (deriv1 > -fm)  # This is just dx/dt == 0

        # Start time of signal change
        idx_before_signal_change = np.max(
            np.where(zd[signal_idx] == False)[0][0] - 1, 0)
        t0 = self.time[idx_before_signal_change]

        # The time all species start to change where the derivative is not zero
        tstart_idxs = np.argmax(
            (zd == False) & (time > t0), axis=1)
        tstart = self.time[tstart_idxs]
        # Stop measuring response time where the species is within the
        # steady state margin and has a zero derivative after its start time
        idxs_first_zd_after_signal = np.argmax(
            (time * zd > np.expand_dims(tstart, axis=1)) & (cond_out == False), axis=1)
        tstop = np.where(idxs_first_zd_after_signal != 0,
                         time[np.arange(len(steady_states)), idxs_first_zd_after_signal], tstart)

        response_times = tstop - tstart
        return response_times

    def frequency(self):
        spectrum = np.fft.fft(self.data)/len(self.data)
        spectrum = spectrum[range(int(len(self.data)/2))]
        freq = np.fft.fftfreq(len(spectrum))
        return freq

    def get_analytics_types(self):
        return ['fold_change',
                'overshoot',
                'RMSE',
                'steady_states'] + self.get_signal_dependent_analytics()

    def get_signal_dependent_analytics(self):
        return ['response_time',
                'precision',
                'precision_estimate',
                'sensitivity',
                'sensitivity_estimate']

    def generate_analytics(self, labels: List[str], signal_onehot=None, ref_circuit_signal=None):
        signal_idxs = np.where(signal_onehot == 1)[0]
        signal_idxs = signal_idxs if len(signal_idxs) >= 1 else None
        analytics = {
            'first_derivative': self.get_derivative(self.data),
            'fold_change': self.fold_change(),
            'RMSE': self.get_rmse(ref_circuit_signal),
        }
        analytics['steady_states'], \
            analytics['is_steady_state_reached'], \
            analytics['final_deriv'] = self.get_steady_state()
        analytics['overshoot'] = self.get_overshoot(
            analytics['steady_states'])

        analytics['response_time'] = {}
        analytics['precision'] = {}
        analytics['precision_estimate'] = {}
        analytics['sensitivity'] = {}
        analytics['sensitivity_estimate'] = {}
        if signal_idxs is not None:
            signal_labels = list(map(labels. __getitem__, signal_idxs))
            for s, s_idx in zip(signal_labels, signal_idxs):
                # analytics['response_time'], \
                #     analytics['response_time_high'], \
                #     analytics['response_time_low'] = self.get_response_times(
                #     analytics['steady_states'], analytics['first_derivative'], signal_idxs=signal_onehot)
                analytics['precision'][s] = self.get_precision(
                    analytics['steady_states'], s_idx)
                analytics['precision_estimate'][s] = self.get_precision(
                    analytics['steady_states'], s_idx, ignore_denominator=True)
                analytics['response_time'][s] = self.get_step_response_times(
                    analytics['steady_states'], analytics['first_derivative'], signal_idx=s_idx)
                analytics['sensitivity'] = self.get_sensitivity(s_idx)
                analytics['sensitivity_estimate'] = self.get_sensitivity(
                    s_idx, ignore_denominator=True)
        else:
            analytics['response_time'] = None  # {s: None for s in labels}
            analytics['precision'] = None  # {s: None for s in labels}
            analytics['precision_estimate'] = None  # {s: None for s in labels}
            analytics['sensitivity'] = None
            analytics['sensitivity_estimate'] = None

        return analytics
