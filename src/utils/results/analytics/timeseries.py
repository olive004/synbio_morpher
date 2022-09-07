import logging
import numpy as np


class Timeseries():
    def __init__(self, data) -> None:
        self.data = data
        self.num_dtype = np.float32

        self.stability_threshold = 0.01

    def get_steady_state(self):
        """ Last 5% of data considered steady state """
        final_deriv = np.average(
            self.get_derivative()[:, :-2])
        is_steady_state_reached = final_deriv < self.stability_threshold
        steady_states = np.expand_dims(
            self.data[:, -1], axis=1).astype(np.float32)
        return steady_states, is_steady_state_reached, final_deriv

    def fold_change(self):
        division_matrix = np.divide(self.data[:, -1].clip(1), self.data[:, 0].clip(1))
        if np.ndim(division_matrix) == 1:
            return np.expand_dims(division_matrix, axis=1)
        else:
            return division_matrix

    def get_derivative(self):
        deriv = np.gradient(self.data)[1]
        return deriv  # get column derivative

    def get_overshoot(self, steady_states):
        return np.expand_dims(np.max(self.data, axis=1), axis=1) - steady_states

    def get_precision(self, steady_states, signal_idx: int):
        if signal_idx is None:
            return None
        starting_states = np.expand_dims(self.data[:, 0], axis=1)
        signal_start = self.data[signal_idx, 0]
        signal_end = self.data[signal_idx, -1]

        """ MODIFYING THIS PART A BIT - GETTING DIVIDE BY ZERO ERROR OTHERWISE"""
        precision = 1
        signal_diff = signal_end - signal_start
        output_diff = steady_states - starting_states

        # if signal_diff == 0:
        #     return self.num_dtype(0) 
        # if signal_start == 0:
        #     signal_start = 1
        # if any(starting_states == 0):
        #     starting_states = 1
        precision = np.absolute(np.divide(
            output_diff / starting_states,
            signal_diff / signal_start
        )).astype(self.num_dtype)
        return np.divide(1, precision)

    def get_sensitivity(self, signal_idx: int):
        if signal_idx is None:
            return None
        starting_states = self.data[:, 0]
        peaks = np.max(self.data, axis=1)
        signal_low = np.min(self.data[signal_idx, :])
        signal_high = np.max(self.data[signal_idx, :])

        """ MODIFYING THIS PART A BIT - GETTING DIVIDE BY ZERO ERROR OTHERWISE """
        output_diff = peaks - starting_states
        signal_diff = signal_high - signal_low

        if signal_diff == 0:
            logging.warning(f'Signal difference was 0 from {signal_high} and {signal_low}.')
            return self.num_dtype(0)
        # elif signal_low == 0 or any(starting_states == 0):
        #     return np.expand_dims(np.absolute(np.divide(
        #         output_diff,
        #         signal_diff
        #     )).astype(self.num_dtype), axis=1)
        return np.absolute(np.divide(
            output_diff / starting_states,
            signal_diff / signal_low
        )).astype(self.num_dtype)

    def get_rmse(self):
        if 'signal_diff' in self.name:
            rmse = np.sqrt(np.sum(np.divide(np.power(self.data, 2), len(self.data))))
        else:
            rmse = 'No reference given.'
        return rmse
        

    def get_response_times(self, steady_states):
        margin_high = 1.05
        margin_low = 0.95
        peak = np.max(self.data)
        has_peak = np.all(peak > steady_states)
        if has_peak:
            post_peak_data = self.data[:, np.argmax(
                self.data < np.max(self.data)):]
            response_time = np.expand_dims(np.argmax(
                post_peak_data < steady_states, axis=1).astype(self.num_dtype), axis=1)
            response_time_high = np.expand_dims(np.argmax(post_peak_data < (
                steady_states * margin_high), axis=1).astype(self.num_dtype), axis=1)
            response_time_low = np.expand_dims(np.argmax(post_peak_data < (
                steady_states * margin_low), axis=1).astype(self.num_dtype), axis=1)
        else:
            post_peak_data = self.data
            response_time = np.expand_dims(np.argmax(
                post_peak_data >= steady_states, axis=1).astype(self.num_dtype), axis=1)
            response_time_high = np.expand_dims(np.argmax(post_peak_data >= (
                steady_states * margin_high), axis=1).astype(self.num_dtype), axis=1)
            response_time_low = np.expand_dims(np.argmax(post_peak_data >= (
                steady_states * margin_low), axis=1).astype(self.num_dtype), axis=1)
        return response_time, response_time_high, response_time_low

    def get_analytics_types(self):
        return ['fold_change',
                'overshoot',
                'precision',
                'response_time',
                'response_time_high',
                'response_time_low',
                'sensitivity',
                'steady_states']

    def frequency(self):
        spectrum = np.fft.fft(self.data)/len(self.data)
        spectrum = spectrum[range(int(len(self.data)/2))]
        freq = np.fft.fftfreq(len(spectrum))
        return freq

    def generate_analytics(self, signal_idx=None):
        analytics = {
            'first_derivative': self.get_derivative(),
            'fold_change': self.fold_change(),
            'RMSE': self.get_rmse(),
            'sensitivity': self.get_sensitivity(signal_idx)
        }
        analytics['steady_states'], \
            analytics['is_steady_state_reached'], \
            analytics['final_deriv'] = self.get_steady_state()
        analytics['response_time'], \
            analytics['response_time_high'], \
            analytics['response_time_low'] = self.get_response_times(
            analytics['steady_states'])

        analytics['overshoot'] = self.get_overshoot(
            analytics['steady_states'])
        analytics['precision'] = self.get_precision(
            analytics['steady_states'], signal_idx)

        return analytics
