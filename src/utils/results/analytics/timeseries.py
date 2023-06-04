import logging
import numpy as np
import jax.numpy as jnp
from typing import List, Tuple
from src.utils.misc.type_handling import merge_dicts
from src.utils.results.analytics.naming import DIFF_KEY, RATIO_KEY


TIMEAXIS = 1


def get_peaks(initial_steady_states, final_steady_states, maxa, mina):
    return jnp.where(
        initial_steady_states < final_steady_states,
        maxa - mina + initial_steady_states,
        mina - maxa + initial_steady_states
    )


def get_derivative(data):
    if data.shape[TIMEAXIS] <= 1:
        return np.ones_like(data) * np.inf
    deriv = jnp.gradient(data)[1]
    return deriv


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


def get_precision_simp(starting_states, steady_states, signal_factor):
    numer = jnp.where((starting_states != 0).astype(int),
                      (steady_states - starting_states) / starting_states, 1)

    precision = jnp.absolute(jnp.divide(
        numer, signal_factor))
    return jnp.divide(1, precision)


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


def get_sensitivity_simp(starting_states, peaks, signal_factor):
    numer = jnp.where((starting_states != 0).astype(int),
                      (peaks - starting_states) / starting_states, np.inf)
    return jnp.absolute(jnp.divide(
        numer, signal_factor))


def get_rmse(data, ref_circuit_data):
    if ref_circuit_data is None:
        return jnp.zeros(shape=(jnp.shape(data)[0], 1))
    t_max = min([jnp.shape(data)[-1], jnp.shape(ref_circuit_data)[-1]]) 
    rmse = jnp.sqrt(
        jnp.sum(jnp.divide(jnp.power(data[:, :t_max] - ref_circuit_data[:, :t_max], 2), jnp.shape(data)[1]), axis=1))
    return jnp.expand_dims(rmse, axis=1)


def get_step_response_times(data, t, steady_states, deriv, signal_time: int):
    """ Assumes that data starts pre-perturbation, but after an initial steady state 
    has been reached. """

    t_expanded = t * jnp.ones_like(steady_states)
    margin_stst = 0.001
    is_data_outside_stst = (data > (steady_states + steady_states * margin_stst)
                            ) | (data < (steady_states - steady_states * margin_stst))

    # 1. Get zero derivative within margin
    fmargin = 0.01
    fm = jnp.expand_dims(jnp.max(deriv, axis=1) * fmargin, axis=1)
    zd = (deriv <= fm) & (deriv >= -fm)  # This is just dx/dt == 0

    # 2. Find start time of signal change
    t0 = signal_time

    # 3. Get the time all species first start to change where the derivative is not zero
    #    If tstart is equal to 0, it means the species did not change after the signal
    tstart = t * ((zd == False) & (t_expanded >= t0)).astype(int)
    tstart = jnp.where(tstart == 0, np.inf, tstart)
    tstart = jnp.min(tstart, axis = 1)

    # 4. Stop measuring response time where the species is within the
    # steady state margin and has a zero derivative after its start time
    idxs_first_zd_after_signal = jnp.argmax(
        (t_expanded >= jnp.expand_dims(tstart, axis=1)) * (is_data_outside_stst == False), axis=1)
        # ((t_expanded * zd) >= jnp.expand_dims(tstart, axis=1)) * (is_data_outside_stst == False), axis=1)

    argmax_workaround = jnp.ones_like(
        steady_states) * jnp.arange(len(t)) == jnp.expand_dims(idxs_first_zd_after_signal - 1, axis=1)
    tstop = jnp.where(jnp.max(t_expanded * argmax_workaround, axis=1) != 0,
                        jnp.max(t_expanded * argmax_workaround, axis=1), tstart)

    response_times = jnp.where(
        tstart != 0,
        tstop - t0,
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
                            signal_onehot: jnp.ndarray, signal_time,
                            ref_circuit_data: jnp.ndarray) -> dict:
    """ Assuming [species, time] for data """
    signal_idxs = None if signal_onehot is None else [int(np.where(signal_onehot == 1)[0])]
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
    analytics['fold_change'] = get_fold_change(
        starting_states=analytics['initial_steady_states'],
        steady_states=analytics['steady_states']
    )

    peaks = get_peaks(analytics['initial_steady_states'], analytics['steady_states'],
                      analytics['max_amount'], analytics['min_amount']) 

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
            # t axis : 1
            t_end = np.min([len(time), data.shape[1]])
            analytics[f'response_time_wrt_species-{s_idx}'] = get_step_response_times(
                data=data[:, :t_end], t=time[:t_end], steady_states=analytics['steady_states'][:, :t_end],
                deriv=analytics['first_derivative'][:, :t_end], signal_time=signal_time)
            analytics[f'sensitivity_wrt_species-{s_idx}'] = get_sensitivity(
                signal_idx=s_idx, peaks=peaks, starting_states=analytics['initial_steady_states']
            )
    return analytics


def generate_differences_ratios(analytics: dict, ref_analytics) -> Tuple[dict]:
    t_axis = 1
    differences = {}
    ratios = {}
    for k in ref_analytics.keys():
        t_end = np.min([ref_analytics[k].shape[t_axis], analytics[k].shape[t_axis]])
        differences[k +
                    DIFF_KEY] = jnp.subtract(analytics[k][:, :t_end], ref_analytics[k][:, :t_end])
        ratios[k + RATIO_KEY] = jnp.where(ref_analytics[k][:, :t_end] !=
                                          0, jnp.divide(analytics[k][:, :t_end], ref_analytics[k][:, :t_end]), np.inf)
    return differences, ratios


def generate_analytics(data, time, labels: List[str], ref_circuit_data=None,
                       signal_onehot=None, signal_time=None):
    if data.shape[0] != len(labels):
        species_axis = data.shape.index(len(labels))
        data = np.swapaxes(data, 0, species_axis)
    analytics = generate_base_analytics(data=data, time=time, labels=labels,
                                        signal_onehot=signal_onehot, signal_time=signal_time,
                                        ref_circuit_data=ref_circuit_data)

    # Differences & ratios
    ref_analytics = generate_base_analytics(data=ref_circuit_data, time=time, labels=labels,
                                            signal_onehot=signal_onehot, signal_time=signal_time,
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
