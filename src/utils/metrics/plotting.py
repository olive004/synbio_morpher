from copy import deepcopy
import numpy as np


class Timeseries():
    def __init__(self, data) -> None:
        self.data = data

        self.stability_threshold = 0.01

    def stability(self):
        """ Last 5% of data considered steady state """
        final_deriv = np.average(
            self.get_derivative()[:-int(len(self.data)*0.05)])
        is_steady_state_reached = final_deriv < self.stability_threshold
        return (is_steady_state_reached, final_deriv)

    def fold_change(self):
        division_vector = deepcopy(self.data[:, -1]).clip(1)
        division_matrix = np.divide(division_vector, division_vector.T)
        import logging
        logging.info(np.shape(self.data))
        return division_matrix

    def get_derivative(self):
        return np.gradient(self.data)[0]  # get column derivative

    def frequency(self):
        spectrum = np.fft.fft(self.data)/len(self.data)
        spectrum = spectrum[range(int(len(self.data)/2))]
        freq = np.fft.fftfreq(len(spectrum))
        return freq

    def generate_analytics(self):
        analytics = {
            'first_derivative': self.get_derivative(),
            'fold_change': self.fold_change(),
            'steady_state': self.stability()
        }
        return analytics
