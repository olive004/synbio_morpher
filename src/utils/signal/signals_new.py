from functools import partial
import logging
from typing import List
import numpy as np
import jax
import jax.numpy as jnp


# Various signal functions
class SignalFuncs():
    def step_function(t, total_time, step_num, dt, target):
        return (jnp.sin(t) + 1)* jnp.where((10 < t) & (t < 130), target/24, 0)

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
        mask = np.float32((xs >= np.expand_dims(positions, 1)) *
                            (xs < np.expand_dims(position_next, 1)) *
                            (xs < np.expand_dims(positions+durations, 1)))
        y1 = y0_*mask  # [num_peaks, time_points]
        y = np.float32(np.sum(y1, axis=0))
        return y


class Signal():
    def __init__(self, 
                 onehot: np.ndarray,
                 signal_cfg: dict=None,
                 time_interval=1) -> None:
        self.onehot = onehot
        self.time_interval = time_interval
        self.func = self.choose_func(signal_cfg)

    def choose_func(signal_cfg: dict):
        return partial(
            SignalFuncs.__getattribute__(signal_cfg['function_name']),
            **signal_cfg['function_kwargs']
        )

    def get_time_steps(self, total_time):
        return total_time / self.time_interval

    def update_time_interval(self, new_time_interval):
        self.time_interval = new_time_interval

    @property
    def real_signal(self, total_time) -> np.ndarray:
        """ Signal is 1-d np matrix """
        return jax.lax.vmap(self.func)(np.arange(0, total_time, self.time_interval))

    def show(self, total_time, out_path='input_signal_plot'):
        from src.utils.results.visualisation import VisODE
        VisODE().plot(data=np.arange(0, total_time, self.time_interval), 
            y=self.real_signal, out_path=out_path)
