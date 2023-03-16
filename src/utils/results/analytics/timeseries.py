import logging
import numpy as np
import jax.numpy as jnp
from typing import List, Tuple
from src.utils.misc.type_handling import merge_dicts
from src.utils.results.analytics.naming import DIFF_KEY, RATIO_KEY


def get_derivative(data):
    deriv = jnp.gradient(data)[1]
    return deriv  # get column derivative


def get_fold_change(starting_states, steady_states):
    denom = jnp.where(starting_states != 0,
                      starting_states, -1)
    fold_change = jnp.where(denom != -1,
                            steady_states / denom, np.nan)
    return fold_change


def get_overshoot(steady_states, peaks):
    return jnp.absolute(peaks - steady_states)


def calculate_precision(output_diff, starting_states, signal_diff, signal_0) -> jnp.ndarray:
    denom = jnp.where(signal_0 != 0, signal_diff / signal_0, 1)
    numer = jnp.where((starting_states != 0).astype(int),
                      output_diff / starting_states, 1)
    precision = jnp.absolute(jnp.divide(
        numer, denom))
    return jnp.divide(1, precision)


def get_precision(signal_idx: int, starting_states, steady_states):
    if signal_idx is None:
        return None
    signal_0 = starting_states[signal_idx]
    signal_1 = steady_states[signal_idx]

    signal_diff = signal_1 - signal_0
    output_diff = steady_states - starting_states

    return calculate_precision(output_diff, starting_states, signal_diff, signal_0)


def calculate_sensitivity(output_diff, starting_states, signal_diff, signal_0) -> jnp.ndarray:
    denom = jnp.where(signal_0 != 0, signal_diff / signal_0, np.inf)
    numer = jnp.where((starting_states != 0).astype(int),
                      output_diff / starting_states, np.inf)
    return jnp.absolute(jnp.divide(
        numer, denom))


def get_sensitivity(signal_idx: int, starting_states, peaks):
    if signal_idx is None:
        return None
    signal_1 = peaks[signal_idx]
    signal_0 = starting_states[signal_idx]

    output_diff = peaks - starting_states
    signal_diff = signal_1 - signal_0

    return calculate_sensitivity(output_diff, starting_states, signal_diff, signal_0)


def get_rmse(data, ref_circuit_data):
    if ref_circuit_data is None:
        return jnp.zeros(shape=(jnp.shape(data)[0], 1))
    rmse = jnp.sqrt(
        jnp.sum(jnp.divide(jnp.power(data - ref_circuit_data, 2), jnp.shape(data)[1]), axis=1))
    return jnp.expand_dims(rmse, axis=1)


def get_step_response_times(data, t, steady_states, signal_time: int):
    """ Assumes that data starts pre-perturbation, but after an initial steady state 
    has been reached. """

    margin = 0.05
    is_steadystate = ~((data > (steady_states + steady_states * margin)
                        ) | (data < (steady_states - steady_states * margin)))
    argmax_workaround = jnp.ones_like(
        steady_states) * jnp.arange(len(t)) == jnp.expand_dims(np.argmax(is_steadystate, axis=1), axis=1)
    tstop = jnp.max(t * argmax_workaround, axis=1)
    response_times = jnp.where(
        (tstop != t[-1]) & (tstop >= signal_time),
        tstop - signal_time,
        np.inf
    )

    if response_times.ndim == 1:
        return jnp.expand_dims(response_times, axis=1)
    return response_times


def frequency(data):
    spectrum = jnp.fft.fft(data)/len(data)
    spectrum = spectrum[range(int(len(data)/2))]
    freq = jnp.fft.fftfreq(len(spectrum))
    return freq


def generate_base_analytics(data: jnp.ndarray, time: jnp.ndarray, labels: List[str],
                            signal_idxs: jnp.ndarray, signal_time,
                            ref_circuit_data: jnp.ndarray) -> dict:
    """ Assuming [species, time] for data """
    if data is None:
        return {}
    analytics = {
        'first_derivative': get_derivative(data),
        'initial_steady_states': jnp.expand_dims(data[:, 0], axis=1),
        'max_amount': jnp.expand_dims(jnp.max(data, axis=-1), axis=1),
        'min_amount': jnp.expand_dims(jnp.min(data, axis=-1), axis=1),
        'RMSE': get_rmse(data, ref_circuit_data),
        'steady_states': jnp.expand_dims(data[:, -1], axis=1)
    }
    analytics['final_deriv'] = jnp.expand_dims(
        analytics['first_derivative'][:, -1], axis=1)
    analytics['fold_change'] = get_fold_change(
        starting_states=analytics['initial_steady_states'],
        steady_states=analytics['steady_states']
    )

    peaks = jnp.where(analytics['initial_steady_states'] !=
                      analytics['max_amount'], analytics['max_amount'], analytics['min_amount'])
    analytics['overshoot'] = get_overshoot(
        steady_states=analytics['steady_states'],
        peaks=peaks
    )

    if signal_idxs is not None:
        signal_labels = list(map(labels.__getitem__, signal_idxs))
        for s, s_idx in zip(signal_labels, signal_idxs):

            analytics[f'precision_wrt_species-{s_idx}'] = get_precision(
                signal_idx=s_idx,
                starting_states=analytics['initial_steady_states'],
                steady_states=analytics['steady_states']
            )
            analytics[f'response_time_wrt_species-{s_idx}'] = get_step_response_times(
                data=data, t=time, steady_states=analytics['steady_states'],
                signal_time=signal_time)
            analytics[f'sensitivity_wrt_species-{s_idx}'] = get_sensitivity(
                signal_idx=s_idx, peaks=peaks, starting_states=analytics['initial_steady_states']
            )
    return analytics


def generate_differences_ratios(analytics: dict, ref_analytics) -> Tuple[dict]:
    differences = {}
    ratios = {}
    for k in ref_analytics.keys():
        differences[k +
                    DIFF_KEY] = jnp.subtract(analytics[k], ref_analytics[k])
        ratios[k + RATIO_KEY] = jnp.where(ref_analytics[k] !=
                                          0, jnp.divide(analytics[k], ref_analytics[k]), np.inf)
    return differences, ratios


def generate_analytics(data, time, labels: List[str], ref_circuit_data=None,
                       signal_idxs=None, signal_time=None):
    if data.shape[0] != len(labels):
        species_axis = data.shape.index(len(labels))
        data = np.swapaxes(data, 0, species_axis)
    analytics = generate_base_analytics(data=data, time=time, labels=labels,
                                        signal_idxs=signal_idxs, signal_time=signal_time,
                                        ref_circuit_data=ref_circuit_data)

    # Differences & ratios
    ref_analytics = generate_base_analytics(data=ref_circuit_data, time=time, labels=labels,
                                            signal_idxs=signal_idxs, signal_time=signal_time,
                                            ref_circuit_data=ref_circuit_data)
    differences, ratios = generate_differences_ratios(analytics, ref_analytics)
    return merge_dicts(analytics, differences, ratios)


class Timeseries():
    def __init__(self, data, time=None) -> None:
        self.data = data
        if data is None:
            self.time = None
        else:
            self.time = np.arange(np.shape(self.data)[
                                  1]) if time is None else time
