
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
from functools import partial
import logging
from typing import List
import numpy as np
import jax.numpy as jnp


def step_function(t, total_time, step_num, dt, target):
    return (jnp.sin(t) + 1)* jnp.where((10 < t) & (t < 130), target/24, 0)


class Signal():
    def __init__(self, identities_idx, total_time,
                 signal=None,
                 time_interval=1,
                 magnitude=1) -> None:
        self.identities_idx = identities_idx
        self.abstract_signal = signal if signal is not None else [0, 1]
        self.total_time = total_time
        self.magnitude = magnitude
        self.time_interval = time_interval
        self.time_steps = self.get_time_steps()

    def get_time_steps(self):
        return self.total_time / self.time_interval

    @property
    def abstract_signal(self):
        return self._abstract_signal

    @abstract_signal.setter
    def abstract_signal(self, value):
        self._abstract_signal = value

    @property
    def time_dilation_func(self):
        return partial(
            np.repeat, repeats=self.time_steps / len(self.abstract_signal))

    def update_time_interval(self, new_time_interval):
        self.time_interval = new_time_interval
        self.time_steps = self.get_time_steps()

    @property
    def real_signal(self) -> np.ndarray:
        """ Signal is 1-d np matrix """
        signal = self.time_dilation_func(self.abstract_signal) * self.magnitude
        if len(signal) < self.time_steps:
            signal = np.concatenate((signal, np.repeat(
                signal[-1], self.time_steps - len(signal))))
        assert len(
            signal) == self.time_steps, f'The signal length {len(signal)} does not equal its intended length {self.time_steps}'
        return signal

    @property
    def summarized_signal(self) -> List[list]:
        """ Return signal as a tuple of the signal scaled to the magnitude, 
        the starting time, and the ending time point of the signal. """
        summ_signal = []
        time_span = self.total_time / len(self.abstract_signal)
        for i, signal in enumerate(self.abstract_signal):
            summ_signal.append(
                (signal*self.magnitude, time_span*i, time_span*(i+1))
            )
        return summ_signal

    @property
    def time(self):
        return np.arange(0, self.total_time, self.time_interval)

    def show(self, out_path='input_signal_plot'):
        from synbio_morpher.utils.results.visualisation import VisODE
        VisODE().plot(data=self.time, y=self.real_signal, out_path=out_path)


class AdaptationTarget(Signal):
    def __init__(self, identities_idx, signal_type='double_exp',
                 signal=None, total_time=None, time_interval=1, magnitude=1) -> None:
        super().__init__(identities_idx, signal, total_time, time_interval, magnitude)

        self.signal_type = signal_type

    @property
    def real_signal(self):
        return self.adapt_pulse(self.time_steps, height=self.magnitude, pulse_type=self.signal_type)

    def adapt_pulse(self, time_points, height, pulse_type):
        ''' Adaptation response curves
        From Shen et al. 2021 https://doi.org/10.1038/s41467-021-23420-5 
        https://github.com/sjx93/rnn_for_gene_network_2020/blob/main/fig1-2_adaptation/target_response_curves.py '''

        def adapt_pulse_double_exp(height) -> np.array:
            # inputs: scalars
            # output: np.array of shape=time_points
            # double exponential response curve
            xs0 = self.time
            y = height*4*(np.exp(-xs0/6) - np.exp(-xs0/3))
            return y

        def adapt_pulse_triangular(height) -> np.array:
            # triangular pulse for the adaptation task
            xs0 = self.time
            y = height * 2 * np.maximum(np.minimum(0.1*xs0, 1-0.1*xs0), 0)
            return y

        if pulse_type == 'double_exp':
            return adapt_pulse_double_exp(height)
        elif pulse_type == 'triangular':
            return adapt_pulse_triangular(height)
        else:
            raise ValueError(f'Could not recognise pulse type {pulse_type}')


class OscillatingSignal(Signal):
    
    numtype = np.float64
    
    def __init__(self, identities_idx, positions, heights, durations,
                 signal=None, total_time=None, magnitude=1) -> None:
        super().__init__(identities_idx, signal, total_time, magnitude)
        self.positions = positions
        self.heights = heights
        self.durations = durations

    @property
    def real_signal(self):
        return self.spikes(positions=self.positions,
                           heights=self.heights,
                           durations=self.durations)

    def spikes(self, positions: list, heights: list, durations: list) -> np.array:
        # inputs: np.array of position,height,duration for each triangular pulse
        # pulse positions arranged in ascending order

        time_points = self.total_time

        num_peaks = positions.shape[0]
        xs0 = self.time
        xs = np.tile(np.expand_dims(xs0, 0), [num_peaks, 1])
        y0 = np.expand_dims((positions/durations + 1)*2*heights, 1) - \
            2*np.expand_dims(heights/durations, 1)*xs
        y0_ = np.expand_dims(heights, 1)-np.abs(y0-np.expand_dims(heights, 1))
        position_next = np.concatenate(
            [positions[1:], time_points+np.ones(1)], axis=0)  # [num_peaks]
        mask = self.numtype((xs >= np.expand_dims(positions, 1)) *
                          (xs < np.expand_dims(position_next, 1)) *
                          (xs < np.expand_dims(positions+durations, 1)))
        y1 = y0_*mask  # [num_peaks, time_points]
        y = self.numtype(np.sum(y1, axis=0))
        return y
