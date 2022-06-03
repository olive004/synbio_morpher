import numpy as np


class Timeseries():
    def __init__(self, data) -> None:
        self.data = data

        self.stability_threshold = 0.01

    def get_steady_state(self):
        """ Last 5% of data considered steady state """
        final_deriv = np.average(
            self.get_derivative()[:, :-2])
        is_steady_state_reached = final_deriv < self.stability_threshold
        steady_states = np.expand_dims(self.data[:, -1], axis=1)
        return {
            "is_steady_state_reached": is_steady_state_reached,
            "steady_states": steady_states,
            "final_deriv": final_deriv
        }

    def fold_change(self):
        division_vector = self.data[:, -1].clip(1)
        division_matrix = np.divide(division_vector, division_vector.T)
        return division_matrix

    def get_derivative(self):
        deriv = np.gradient(self.data)[1]
        return deriv  # get column derivative 
    
    def get_response_time(self, steady_states):
        self.data

    def frequency(self):
        spectrum = np.fft.fft(self.data)/len(self.data)
        spectrum = spectrum[range(int(len(self.data)/2))]
        freq = np.fft.fftfreq(len(spectrum))
        return freq

    def generate_analytics(self):
        analytics = {
            'first_derivative': self.get_derivative(),
            'fold_change': self.fold_change(),
            'steady_state': self.get_steady_state(),
        }
        analytics['response_time'] = self.get_response_time(analytics['steady_state']['steady_states'])
        return analytics
