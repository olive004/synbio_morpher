
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
import argparse
import os
from typing import Dict, List

from synbio_morpher.srv.parameter_prediction.simulator_loading import find_simulator_loader
from synbio_morpher.srv.io.manage.sys_interface import PACKAGE_DIR
from synbio_morpher.utils.misc.io import get_pathnames
from synbio_morpher.utils.misc.string_handling import remove_special_py_functions
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
from synbio_morpher.srv.io.manage.sys_interface import make_filename_safely
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict


def get_configs(config_file, config_filepath):
    config_filepath = make_filename_safely(config_filepath)
    if config_file is None and config_filepath:
        config_file = load_json_as_dict(config_filepath)
    elif config_file and config_filepath:
        raise ValueError(
            'Both a config and a config filepath were defined - only use one config option.')
    elif config_file is None and config_filepath is None:
        raise ValueError('Config file or path needed as input to function.')
    return config_file


def create_argparse_from_dict(dict_args: Dict):
    parser = argparse.ArgumentParser()
    args_namespace, left_argv = parser.parse_known_args(args=dict_args)
    args_namespace = update_namespace_with_dict(args_namespace, dict_args)
    return args_namespace, left_argv


def get_simulator_names() -> List:
    simulator_dir = os.path.join(PACKAGE_DIR, "src", "srv", "parameter_prediction")
    simulators = remove_special_py_functions(os.listdir(simulator_dir))
    from synbio_morpher.srv.parameter_prediction.simulator_loading import extra_simulators
    simulators = simulators + extra_simulators
    return simulators


def handle_simulator_cfgs(simulator, simulator_cfg_path):
    simulator_cfg = load_json_as_dict(simulator_cfg_path)
    cfg_protocol = find_simulator_loader(simulator)
    if any([os.sep in v for v in simulator_cfg.values() if type(v) == str]):
        for k, v in simulator_cfg.items():
            if type(v) == str:
                simulator_cfg[k] = os.path.join(PACKAGE_DIR, v) if ('src' in v) and (os.sep in v) and (PACKAGE_DIR not in v) else v
    return cfg_protocol(simulator_cfg)


def parse_cfg_args(config: dict = None, default_args: Dict = None) -> Dict:

    if default_args is None:
        default_args = retrieve_default_args()
    simulator_kwargs = load_simulator_kwargs(default_args, config)
    config['interaction_simulator']['simulator_kwargs'] = simulator_kwargs
    config['interaction_simulator']['molecular_params'] = config['molecular_params']
    config['interaction_simulator']['compute_by_filename'] = config['interaction_simulator'].get('compute_by_filename', False)
    config['simulation_steady_state'] = default_args['simulation_steady_state']

    return config


def load_simulator_kwargs(default_args: dict, config_args: str = None) -> Dict:
    target_simulator_name = config_args.get(
        'interaction_simulator', {}).get('name')
    simulator_kwargs = None
    for simulator_name in get_simulator_names():
        kwarg_condition = simulator_name in default_args
        if not target_simulator_name is None:
            kwarg_condition = kwarg_condition and simulator_name == target_simulator_name
        if kwarg_condition:
            simulator_kwargs = handle_simulator_cfgs(
                simulator_name, default_args[simulator_name])
            break
    return simulator_kwargs


def retrieve_default_args() -> Dict:
    fn = get_pathnames(file_key='default_args', search_dir=os.path.join(
        PACKAGE_DIR, 'src', 'utils', 'common', 'configs', 'simulators'), first_only=True)
    default_args = load_json_as_dict(fn)
    return default_args


def update_namespace_with_dict(namespace_args, updater_dict: Dict):
    vars(namespace_args).update(updater_dict)
    return namespace_args
