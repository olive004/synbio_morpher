from functools import partial
import numpy as np


class Signal():
    def __init__(self, identities_idx, signal=None,
                 total_time=None,
                 magnitude=1) -> None:
        self.identities_idx = identities_idx
        self.abstract_signal = signal if signal is not None else [0, 1]
        self.total_time = total_time if total_time else len(
            self.abstract_signal)
        self.magnitude = magnitude
        self.time_steps = total_time / len(self.abstract_signal)
        self.time_dilation_func = partial(np.repeat, repeats=self.time_steps)

    @property
    def abstract_signal(self):
        return self._abstract_signal

    @abstract_signal.setter
    def abstract_signal(self, value):
        self._abstract_signal = value

    @property
    def real_signal(self):
        return self.time_dilation_func(self.abstract_signal) * self.magnitude

    def follow(self):
        pass


class AdaptationTarget(Signal):
    def __init__(self, identities_idx, signal=None, total_time=None, magnitude=1) -> None:
        super().__init__(identities_idx, signal, total_time, magnitude)

    @property
    def real_signal(self):
        return self.adapt_pulse_double_exp(len(self.time_steps), height=self.magnitude)

    def adapt_pulse_double_exp(time_points: int, height) -> np.array:
        ''' Double exponential response curve
        From Shen et al. 2021 https://doi.org/10.1038/s41467-021-23420-5 
        https://github.com/sjx93/rnn_for_gene_network_2020/blob/main/fig1-2_adaptation/target_response_curves.py '''
        # inputs: scalars
        # output: np.array of shape=time_points
        # double exponential response curve
        xs0 = np.linspace(0.0, (time_points-1), time_points)
        y = height*2*(np.exp(-xs0/6) - np.exp(-xs0/3))
        return y


class OscillatingSignal(Signal):
    def __init__(self, identities_idx, signal=None, total_time=None, magnitude=1) -> None:
        super().__init__(identities_idx, signal, total_time, magnitude)
