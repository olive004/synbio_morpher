import logging
import numpy as np
import jax.numpy as jnp
from typing import List, Tuple
from src.utils.misc.type_handling import merge_dicts


NUM_DTYPE = np.float32
DIFF_KEY = '_diff_to_base_circuit'
RATIO_KEY = '_ratio_from_mutation_to_base'


def get_derivative(data):
    deriv = jnp.gradient(data)[1]
    return deriv  # get column derivative


def get_steady_state(data):
    """ Last 5% of data considered steady state """
    steady_state_threshold = 0.01
    final_deriv = jnp.average(
        get_derivative(data)[:, -3:], axis=1)
    steady_states = jnp.expand_dims(
        data[:, -1], axis=1).astype(jnp.float32)
    return steady_states, final_deriv


def fold_change(data):
    # fold_change = jnp.where(data[:, 0] != 0,
    #     data[:, -1] / data[:, 0], np.inf)
    denom = jnp.where(data[:, 0] != 0,
                      data[:, 0], -1)
    fold_change = jnp.where(denom != -1,
                            data[:, -1] / denom, np.inf)
    if len(np.shape(fold_change)) > 1:
        fold_change = jnp.expand_dims(fold_change, axis=1)
    return fold_change


def get_overshoot(data, steady_states):
    return (jnp.expand_dims(jnp.max(data, axis=1), axis=1) - steady_states)


def calculate_precision(output_diff, starting_states, signal_diff, signal_start) -> jnp.ndarray:
    denom = jnp.where(signal_start != 0, signal_diff / signal_start, 1)
    numer = jnp.where((starting_states != 0).astype(int),
                      output_diff / starting_states, 1)
    precision = jnp.absolute(jnp.divide(
        numer, denom))
    return jnp.divide(1, precision)


def get_precision(data, steady_states, signal_idx: int):
    if signal_idx is None:
        return None
    starting_states = jnp.expand_dims(data[:, 0], axis=1)
    signal_start = data[signal_idx, 0]
    signal_end = data[signal_idx, -1]

    """ IF YOU IGNORE THE DENOMINATOR THE DIVIDE BY ZERO GOES AWAY AND YOU GET UNSCALED PRECISION """
    signal_diff = signal_end - signal_start
    output_diff = steady_states - starting_states

    return calculate_precision(output_diff, starting_states, signal_diff, signal_start)


def calculate_sensitivity(output_diff, starting_states, signal_diff, signal_low) -> jnp.ndarray:
    denom = jnp.where(signal_low != 0, signal_diff / signal_low, np.inf)
    numer = jnp.where((starting_states != 0).astype(int),
                      output_diff / starting_states, np.inf)
    return jnp.expand_dims(jnp.absolute(jnp.divide(
        numer, denom)), axis=1)


def get_sensitivity(data, signal_idx: int):
    if signal_idx is None:
        return None
    starting_states = data[:, 0]
    peaks = jnp.max(data, axis=1)
    signal_low = jnp.min(data[signal_idx, :])
    signal_high = jnp.max(data[signal_idx, :])

    """ IF YOU IGNORE THE DENOMINATOR THE DIVIDE BY ZERO GOES AWAY AND YOU GET UNSCALED SENSITIVITY """
    output_diff = peaks - starting_states
    signal_diff = signal_high - signal_low

    return calculate_sensitivity(output_diff, starting_states, signal_diff, signal_low)


def get_rmse(data, ref_circuit_data):
    if ref_circuit_data is None:
        return jnp.zeros(shape=(jnp.shape(data)[0], 1))
    rmse = jnp.sqrt(
        jnp.sum(jnp.divide(jnp.power(data - ref_circuit_data, 2), jnp.shape(data)[1]), axis=1))
    return jnp.expand_dims(rmse, axis=1)


def get_step_response_times(data, t, steady_states, deriv1, signal_idx: jnp.ndarray):
    time = t * jnp.ones_like(steady_states)
    margin = 0.05
    cond_out = (data > (steady_states + steady_states * margin)
                ) | (data < (steady_states - steady_states * margin))

    # Get zero derivative within margin
    fmargin = 0.001
    fm = jnp.expand_dims(jnp.max(deriv1, axis=1) * fmargin, axis=1)
    zd = (deriv1 < fm) & (deriv1 > -fm)  # This is just dx/dt == 0

    # Start time of signal change
    t0 = jnp.max(t * (zd[signal_idx] == False).astype(int))

    # The time all species start to change where the derivative is not zero
    tstart = jnp.max(t * ((zd == False) & (time > t0)).astype(int), axis=1)

    # Stop measuring response time where the species is within the
    # steady state margin and has a zero derivative after its start time
    idxs_first_zd_after_signal = jnp.argmax(
        (time * zd > jnp.expand_dims(tstart, axis=1)) & (cond_out == False), axis=1)

    argmax_workaround = jnp.ones_like(
        steady_states) * jnp.arange(len(t)) == jnp.expand_dims(idxs_first_zd_after_signal, axis=1)
    tstop = jnp.where(jnp.max(time * argmax_workaround, axis=1) != 0,
                      jnp.max(time * argmax_workaround, axis=1), tstart)

    response_times = tstop - tstart

    if response_times.ndim == 1:
        return jnp.expand_dims(response_times, axis=1)
    return response_times


def frequency(data):
    spectrum = jnp.fft.fft(data)/len(data)
    spectrum = spectrum[range(int(len(data)/2))]
    freq = jnp.fft.fftfreq(len(spectrum))
    return freq


def get_analytics_types_base() -> List[str]:
    return ['fold_change',
            'overshoot',
            'max_amount',
            'min_amount',
            'RMSE',
            'steady_states']


def get_signal_dependent_analytics() -> List[str]:
    return ['response_time',
            'precision',
            'precision_estimate',
            'sensitivity',
            'sensitivity_estimate']


def get_analytics_types_all() -> List[str]:
    return get_analytics_types() + get_analytics_types_diffs() + get_analytics_types_ratios()


def get_analytics_types() -> List[str]:
    """ The naming here has to be unique and not include small 
    raw values like diff, ratio, max, min. """
    return get_analytics_types_base() + get_signal_dependent_analytics()


def get_diffs(analytics_func=get_analytics_types) -> List[str]:
    return [a + DIFF_KEY for a in analytics_func()]


def get_ratios(analytics_func=get_analytics_types) -> List[str]:
    return [a + RATIO_KEY for a in analytics_func()]


def get_signal_dependent_analytics_all() -> List[str]:
    return get_signal_dependent_analytics() + \
        get_diffs(get_signal_dependent_analytics) + \
        get_ratios(get_signal_dependent_analytics)


def get_base_analytics_all() -> List[str]:
    return get_analytics_types_base() + \
        get_diffs(get_analytics_types_base) + \
        get_ratios(get_analytics_types_base)


def get_analytics_types_diffs() -> List[str]:
    return get_diffs(get_analytics_types)


def get_analytics_types_ratios() -> List[str]:
    return get_ratios(get_analytics_types)


def get_true_names_analytics(candidate_cols: List[str]) -> List[str]:
    true_names = []
    analytics_sig = get_signal_dependent_analytics()
    analytics_base = get_base_analytics_all()

    for c in candidate_cols:
        for s in analytics_sig:
            if s in c:
                if (c.replace(s, '')).startswith('_wrt'):
                    # true_name = c.replace(DIFF_KEY, '').replace(RATIO_KEY, '')
                    true_names.append(c)
        for b in analytics_base:
            if b == c:
                true_names.append(c)
    return true_names


def generate_base_analytics(data: jnp.ndarray, time: jnp.ndarray, labels: List[str], signal_idxs: jnp.ndarray, ref_circuit_data: jnp.ndarray) -> dict:
    if data is None:
        return {}
    analytics = {
        'first_derivative': get_derivative(data),
        'fold_change': fold_change(data),
        'RMSE': get_rmse(data, ref_circuit_data),
        'max_amount': jnp.expand_dims(jnp.max(data, axis=1), axis=1),
        'min_amount': jnp.expand_dims(jnp.min(data, axis=1), axis=1)
    }
    analytics['steady_states'], \
        analytics['final_deriv'] = get_steady_state(data)
    analytics['overshoot'] = get_overshoot(data,
                                           analytics['steady_states'])

    if signal_idxs is not None:
        signal_labels = list(map(labels.__getitem__, signal_idxs))
        for s, s_idx in zip(signal_labels, signal_idxs):
            analytics[f'precision_wrt_species-{s_idx}'] = get_precision(
                data, analytics['steady_states'], s_idx)
            analytics[f'precision_estimate_wrt_species-{s_idx}'] = get_precision(
                data, analytics['steady_states'], s_idx)
            analytics[f'response_time_wrt_species-{s_idx}'] = get_step_response_times(
                data, time, analytics['steady_states'], analytics['first_derivative'], signal_idx=s_idx)
            analytics[f'sensitivity_wrt_species-{s_idx}'] = get_sensitivity(
                data, s_idx)
            analytics[f'sensitivity_estimate_wrt_species-{s_idx}'] = get_sensitivity(
                data, s_idx)
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


# def generate_means(analytics: dict):


def generate_analytics(data, time, labels: List[str], ref_circuit_data=None, signal_onehot=None):
    signal_idxs = jnp.where(signal_onehot == 1)[0]
    signal_idxs = signal_idxs if len(signal_idxs) >= 1 else None
    analytics = generate_base_analytics(data=data, time=time, labels=labels,
                                        signal_idxs=signal_idxs, ref_circuit_data=ref_circuit_data)

    # Differences & ratios
    ref_analytics = generate_base_analytics(data=ref_circuit_data, time=time, labels=labels,
                                            signal_idxs=signal_idxs, ref_circuit_data=ref_circuit_data)
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
        self.num_dtype = np.float32
