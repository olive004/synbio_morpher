from functools import partial
import numpy as np


class Signal():
    def __init__(self, in_magnitude, total_time, idx_identity, signal=None) -> None:
        self.total_time = total_time
        self.idx_identity = idx_identity
        self.abstract_signal = signal if signal is not None else [1, 0, 0, 1, 0]
        self.magnitude = in_magnitude
        time_steps = total_time / len(self.abstract_signal)
        self.time_dilation_func = partial(np.repeat, repeats=time_steps)

    @property
    def abstract_signal(self):
        return self._abstract_signal

    @abstract_signal.setter
    def abstract_signal(self, value):
        self._abstract_signal = value

    @property
    def real_signal(self):
        return self.time_dilation_func(self.abstract_signal) * self.magnitude
