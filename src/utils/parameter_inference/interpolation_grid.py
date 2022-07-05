

import logging
import numpy as np


def parameter_range_creation(range_min, range_max, range_step, is_logscale=False) -> np.ndarray:
    if not is_logscale:
        return np.arange(range_min, range_max, range_step)
    else:
        num_parameters = int(np.ceil((range_max - range_min) / range_step))
        log_scale = np.logspace(range_min, range_max, num=num_parameters)
        return np.interp(log_scale, (log_scale.min(), log_scale.max()), (range_min, range_max))


def create_parameter_range(range_configs: dict) -> np.ndarray:
    logging.info(range_configs)

    min_key = [k for k in range_configs.keys() if 'min' in k or 'start' in k][0]
    max_key = [k for k in range_configs.keys() if 'max' in k or 'end' in k][0]
    step_key = [k for k in range_configs.keys() if 'step' in k][0]
    is_logscale = range_configs.get('log_scale', False)
    return parameter_range_creation(*[
        range_configs[min_key], range_configs[max_key], range_configs[step_key]
    ], is_logscale=is_logscale)
