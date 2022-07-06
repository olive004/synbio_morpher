

import logging
import numpy as np

from src.utils.misc.numerical import calculate_num_decimals


def parameter_range_creation(range_min, range_max, range_step, is_logscale=False) -> np.ndarray:
    """ Rounding numbers created with arange to nearest decimal of the range_step 
    to avoid numerical errors downstream """
    if not is_logscale:
        parameter_range = np.arange(range_min, range_max, range_step).astype(np.float64)
        return np.around(parameter_range, np.power(10, calculate_num_decimals(range_step)-1))
    else:
        num_parameters = int(np.ceil((range_max - range_min) / range_step))
        log_scale = np.logspace(range_min, range_max, num=num_parameters)
        return np.interp(log_scale, (log_scale.min(), log_scale.max()), (range_min, range_max)).astype(np.float64)


def create_parameter_range(range_configs: dict) -> np.ndarray:
    min_key = [k for k in range_configs.keys() if 'min' in k or 'start' in k][0]
    max_key = [k for k in range_configs.keys() if 'max' in k or 'end' in k][0]
    step_key = [k for k in range_configs.keys() if 'step' in k][0]
    is_logscale = range_configs.get('log_scale', False)
    return parameter_range_creation(*[
        range_configs[min_key], range_configs[max_key], range_configs[step_key]
    ], is_logscale=is_logscale)
