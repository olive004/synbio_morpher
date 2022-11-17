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

    def get_response_times(self, steady_states, first_deriv, signal_idx: np.ndarray):
        """ """
        time = self.time * np.ones_like(steady_states)
        margin = 0.05
        sm = steady_states * margin
        fm = np.max(first_deriv[signal_idx]) * 0.001
        cond_out = (steady_states > (steady_states + sm)) & (steady_states < (steady_states - sm))

        zd = (first_deriv[signal_idx] < (first_deriv[signal_idx] + fm)) & (first_deriv[signal_idx] > (first_deriv[signal_idx] - fm))
        second_derivative = self.get_derivative(first_deriv)
        t0 = time[np.where(zd)[0]]

        def find_start_time(fd, signal_idx):
            fd[signal_idx, :] == 0

        def find_final_time(fd, sd):
            cond_out_fd = fd == 0 & cond_out
            if any(time[zd] > t0):
                t_out = time[np.logical_and(cond_out_fd, time > t0)]
                ty_f = time[np.logical_and(fd == 0, time > t0)][0]
                ty = max(ty_f, t_out[0])
            else:
                zd2 = sd == 0
                ty = time[np.logical_and(zd2, time > t0)]
            return ty

        t_signal = find_final_time(first_deriv[signal_idx, :], second_derivative[signal_idx, :])
        ty = find_final_time(first_deriv, second_derivative)
        return t_signal - ty

    # def get_response_times(self, steady_states, first_derivative, signal_idxs: np.ndarray):
    #     species_num = np.shape(steady_states)[0]
    #     if signal_idxs is None:
    #         return [None] * 3

    #     def get_response_time_thresholded(threshold, signal_idx):
    #         """ The response time is calculated as the time from the last point
    #         where the signal changed to the point where the output steadied. This 
    #         is the same as the last stationary point of the output minus the 
    #         last stationary point of the signal derivative. """
    #         zero_deriv = np.logical_and(first_derivative <= 0 +
    #                                     threshold, first_derivative >= 0-threshold).astype(int)
    #         signal_mask = np.ones(species_num)
    #         signal_mask[signal_idx] = 0
    #         if np.all(np.sum(np.gradient(zero_deriv, axis=1) != 0, axis=1) * signal_mask == 0):
    #             # If no species responded to the signal
    #             return [None] * species_num
    #         all_deriv_change_points = [np.where(np.gradient(zero_deriv, axis=1)[
    #             i] != 0)[0] for i in range(species_num)]
    #         response_times = [None] * species_num
    #         if len(all_deriv_change_points[signal_idx]) != 0:
    #             for i in range(species_num):
    #                 if len(all_deriv_change_points[i]) != 0 and i != signal_idx:
    #                     response_times[i] = self.time[all_deriv_change_points[signal_idx][-1]] - \
    #                         self.time[all_deriv_change_points[i][-1]]
    #         return response_times
    #     margin = 0.05
    #     clean_derivative = first_derivative[np.where(first_derivative < 1e20)]
    #     threshold = max(
    #         0.001 * np.max(clean_derivative), 
    #     0.001) # avoid inf
    #     response_time = get_response_time_thresholded(
    #         threshold=threshold, signal_idx=signal_idxs[0])
    #     response_time_high = get_response_time_thresholded(
    #         threshold=first_derivative[np.where(
    #             clean_derivative <= margin*clean_derivative+clean_derivative)][-1],
    #         signal_idx=signal_idxs[0])
    #     response_time_low = get_response_time_thresholded(
    #         threshold=clean_derivative[np.where(
    #             clean_derivative >= -margin*clean_derivative+clean_derivative)][-1],
    #         signal_idx=signal_idxs[0])

    #     return response_time, response_time_high, response_time_low

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

    def generate_analytics(self, labels: List[str], signal_onehot=None, ref_circuit_signal=None):
        signal_idxs = np.where(signal_onehot == 1)[0]
        signal_idxs = signal_idxs if len(signal_idxs) >= 1 else None
        analytics = {
            'first_derivative': self.get_derivative(self.data),
            'fold_change': self.fold_change(),
            'RMSE': self.get_rmse(ref_circuit_signal),
            'sensitivity': self.get_sensitivity(signal_idxs),
            'sensitivity_estimate': self.get_sensitivity(signal_idxs, ignore_denominator=True)
        }
        analytics['steady_states'], \
            analytics['is_steady_state_reached'], \
            analytics['final_deriv'] = self.get_steady_state()
        analytics['overshoot'] = self.get_overshoot(
            analytics['steady_states'])
        
        analytics['response_time'] = {}
        analytics['precision'] = {}
        analytics['precision_estimate'] = {}
        if signal_idxs is not None:
            signal_labels = list(map(labels. __getitem__, signal_idxs))
            for s, s_idx in zip(signal_labels, signal_idxs):
                # analytics['response_time'], \
                #     analytics['response_time_high'], \
                #     analytics['response_time_low'] = self.get_response_times(
                #     analytics['steady_states'], analytics['first_derivative'], signal_idxs=signal_onehot)

                analytics['response_time'][s] = self.get_response_times(
                    analytics['steady_states'], analytics['first_derivative'], signal_idx=s_idx)
                analytics['precision'][s] = self.get_precision(
                    analytics['steady_states'], s_idx)
                analytics['precision_estimate'][s] = self.get_precision(
                    analytics['steady_states'], s_idx, ignore_denominator=True)
        else:
            analytics['response_time'] = {s: None for s in labels}
            analytics['precision'] = {s: None for s in labels}
            analytics['precision_estimate'] = {s: None for s in labels}

        return analytics
