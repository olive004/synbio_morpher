import logging
import numpy as np


class Timeseries():
    def __init__(self, data, time=None) -> None:
        self.data = data
        if data is None:
            self.time = None
        else:
            self.time = np.arange(np.shape(self.data)[
                                  1]) if time is None else time
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
        division_matrix = np.divide(
            self.data[:, -1].clip(1), self.data[:, 0].clip(1))
        if np.ndim(division_matrix) == 1:
            return np.expand_dims(division_matrix, axis=1)
        else:
            return division_matrix

    def get_derivative(self):
        deriv = np.gradient(self.data)[1]
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

    def get_response_times(self, steady_states, first_derivative, signal_idx):
        if signal_idx is None:
            return np.zeros(np.shape(steady_states)[0])

        def get_response_time_thresholded(threshold):
            """ The response time is calculated as the time from the last point
            where the signal changed to the point where the output steadied. This 
            is the same as the last stationary point of the output minus the 
            last stationary point of the signal derivative. """
            zero_deriv = np.logical_and(first_derivative <= 0 +
                                        threshold, first_derivative >= 0-threshold).astype(int)
            logging.info(np.gradient(zero_deriv, axis=1) != 0, axis=1)
            logging.info(np.sum(np.gradient(zero_deriv, axis=1) != 0, axis=1))
            logging.info(np.sum(np.gradient(zero_deriv, axis=1) != 0, axis=1))
            if np.any(np.sum(np.gradient(zero_deriv, axis=1) != 0, axis=1) == 1):
                return [None] * np.shape(first_derivative)[0]
            last_deriv_change_points = [np.where(np.gradient(zero_deriv, axis=1)[
                                                 i] != 0)[0][-1] for i in range(np.shape(first_derivative)[0])]
            return [self.time[last_deriv_change_points[signal_idx]] -
                    self.time[last_deriv_change_points[i]] for i in range(np.shape(steady_states)[0])]
        margin = 0.05
        response_time = get_response_time_thresholded(threshold=0.001)
        response_time_high = get_response_time_thresholded(
            threshold=first_derivative[np.where(steady_states <= margin*steady_states+steady_states)][-1])
        response_time_low = get_response_time_thresholded(
            threshold=first_derivative[np.where(steady_states >= -margin*steady_states+steady_states)][-1])

        return response_time, response_time_high, response_time_low

    # def get_response_times_og(self, final_steady_states):
    #     margin_high = 1.05
    #     margin_low = 0.95
    #     peak = np.max(self.data)
    #     has_peak = np.all(peak > final_steady_states)
    #     if has_peak:
    #         post_peak_data = self.data[:, np.argmax(
    #             self.data < peak):]
    #         response_time = np.expand_dims(np.argmax(
    #             post_peak_data < final_steady_states, axis=1).astype(self.num_dtype), axis=1)
    #         response_time_high = np.expand_dims(np.argmax(post_peak_data < (
    #             final_steady_states * margin_high), axis=1).astype(self.num_dtype), axis=1)
    #         response_time_low = np.expand_dims(np.argmax(post_peak_data < (
    #             final_steady_states * margin_low), axis=1).astype(self.num_dtype), axis=1)
    #     else:
    #         post_peak_data = self.data
    #         response_time = np.expand_dims(np.argmax(
    #             post_peak_data >= final_steady_states, axis=1).astype(self.num_dtype), axis=1)
    #         response_time_high = np.expand_dims(np.argmax(post_peak_data >= (
    #             final_steady_states * margin_high), axis=1).astype(self.num_dtype), axis=1)
    #         response_time_low = np.expand_dims(np.argmax(post_peak_data >= (
    #             final_steady_states * margin_low), axis=1).astype(self.num_dtype), axis=1)
    #     return response_time, response_time_high, response_time_low

    # def get_response_times_reverse(self, final_steady_states):
    #     margin_high = 1.05
    #     margin_low = 0.95
    #     dip = np.min(self.data)
    #     has_dip = np.all(dip < final_steady_states)
    #     if has_dip:
    #         post_dip_data = self.data[:, np.argmax(
    #             self.data > dip):]
    #         response_time = np.expand_dims(np.argmax(
    #             post_dip_data > final_steady_states, axis=1).astype(self.num_dtype), axis=1)
    #         response_time_high = np.expand_dims(np.argmax(post_dip_data > (
    #             final_steady_states * margin_high), axis=1).astype(self.num_dtype), axis=1)
    #         response_time_low = np.expand_dims(np.argmax(post_dip_data > (
    #             final_steady_states * margin_low), axis=1).astype(self.num_dtype), axis=1)
    #     else:
    #         post_dip_data = self.data
    #         response_time = np.expand_dims(np.argmax(
    #             post_dip_data >= final_steady_states, axis=1).astype(self.num_dtype), axis=1)
    #         response_time_high = np.expand_dims(np.argmax(post_dip_data >= (
    #             final_steady_states * margin_high), axis=1).astype(self.num_dtype), axis=1)
    #         response_time_low = np.expand_dims(np.argmax(post_dip_data >= (
    #             final_steady_states * margin_low), axis=1).astype(self.num_dtype), axis=1)
    #     return response_time, response_time_high, response_time_low

    def get_analytics_types(self):
        return ['fold_change',
                'overshoot',
                'precision',
                'precision_estimate',
                'response_time',
                'response_time_high',
                'response_time_low',
                'RMSE',
                'sensitivity',
                'sensitivity_estimate',
                'steady_states']

    def frequency(self):
        spectrum = np.fft.fft(self.data)/len(self.data)
        spectrum = spectrum[range(int(len(self.data)/2))]
        freq = np.fft.fftfreq(len(spectrum))
        return freq

    def generate_analytics(self, signal_idx=None, ref_circuit_signal=None):
        analytics = {
            'first_derivative': self.get_derivative(),
            'fold_change': self.fold_change(),
            'RMSE': self.get_rmse(ref_circuit_signal),
            'sensitivity': self.get_sensitivity(signal_idx),
            'sensitivity_estimate': self.get_sensitivity(signal_idx, ignore_denominator=True)
        }
        analytics['steady_states'], \
            analytics['is_steady_state_reached'], \
            analytics['final_deriv'] = self.get_steady_state()
        analytics['response_time'], \
            analytics['response_time_high'], \
            analytics['response_time_low'] = self.get_response_times(
            analytics['steady_states'], analytics['first_derivative'], signal_idx=signal_idx)

        analytics['overshoot'] = self.get_overshoot(
            analytics['steady_states'])
        analytics['precision'] = self.get_precision(
            analytics['steady_states'], signal_idx)
        analytics['precision_estimate'] = self.get_precision(
            analytics['steady_states'], signal_idx, ignore_denominator=True)

        return analytics
