import argparse
import json
import os
from typing import Dict, List

from src.srv.parameter_prediction.simulator_loading import find_simulator_loader
from src.utils.misc.string_handling import remove_special_py_functions
from src.utils.data.data_format_tools.common import merge_dicts, load_json_as_dict


# ROOT_DIR = os.environ['ROOT_DIR']


def create_argparse_from_dict(dict_args: Dict):
    parser = argparse.ArgumentParser()
    args_namespace, left_argv = parser.parse_known_args(args=dict_args)
    args_namespace = update_namespace_with_dict(args_namespace, dict_args)
    return args_namespace, left_argv


def get_simulator_names() -> List:
    simulator_dir = os.path.join("src", "srv", "parameter_prediction")
    simulators = remove_special_py_functions(os.listdir(simulator_dir))
    from src.srv.parameter_prediction.simulator_loading import extra_simulators
    simulators = simulators + extra_simulators
    return simulators


def handle_simulator_cfgs(simulator, simulator_cfg_path):
    simulator_cfg = load_json_as_dict(simulator_cfg_path)
    cfg_protocol = find_simulator_loader(simulator)
    return cfg_protocol(simulator_cfg)


def parse_cfg_args(config_file: dict = None, dict_args: Dict = None) -> Dict:

    if dict_args is None:
        dict_args = retrieve_default_args()
    dict_args = load_simulator_cfgs(dict_args)
    dict_args = merge_dicts(dict_args, config_file)

    return dict_args


def load_simulator_cfgs(dict_args) -> Dict:
    for simulator_name in get_simulator_names():
        if simulator_name in dict_args:
            simulator_cfg = handle_simulator_cfgs(
                simulator_name, dict_args[simulator_name])
            dict_args[simulator_name] = simulator_cfg
    return dict_args


def retrieve_default_args() -> Dict:
    fn = os.path.join('scripts', 'common', 'default_args.json')
    default_args = json.load(open(fn))
    return default_args


def update_namespace_with_dict(args, updater_dict: Dict):
    vars(args).update(updater_dict)
    return args
