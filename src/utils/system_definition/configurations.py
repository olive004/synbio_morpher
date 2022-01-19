import argparse
import json
import logging
import os
from typing import Dict, Iterable, List

from src.utils.parameter_prediction.simulator_loading import find_simulator_loader
from src.utils.misc.string_handling import remove_special_py_functions


def create_argparse_from_dict(dict_args: Dict):
    parser = argparse.ArgumentParser()
    args_namespace, left_argv = parser.parse_known_args(args=dict_args)
    args_namespace = update_namespace_with_dict(args_namespace, dict_args)
    return args_namespace, left_argv


def get_simulator_names() -> List:
    simulator_dir = os.path.join("src", "utils", "parameter_prediction")
    simulators = remove_special_py_functions(os.listdir(simulator_dir))
    from src.utils.parameter_prediction.simulator_loading import extra_simulators
    simulators = simulators + extra_simulators
    return simulators


def handle_simulator_cfgs(simulator, simulator_cfg_path):
    simulator_cfg = load_json_as_dict(simulator_cfg_path)
    cfg_protocol = find_simulator_loader(simulator)
    return cfg_protocol(simulator_cfg)


def parse_cfg_args(config_file: str = None, dict_args: Dict = None) -> Dict:

    if dict_args is None:
        dict_args = retrieve_default_args()

    dict_args = load_simulator_cfgs(dict_args)
    dict_args = merge_dicts(dict_args, config_file)

    return dict_args


# def expand_json_cfgs(paths_dict):
#     grouped_cfgs = {}
#     for group, path in paths_dict.items():
#         try:
#             grouped_cfgs[group] = json.load(open(path))
#         except FileNotFoundError:
#             logging.error(f'JSON path {path} not found')
#     return grouped_cfgs


def load_json_as_dict(json_pathname):
    try:
        jdict = json.load(open(json_pathname))
    except FileNotFoundError:
        logging.error(f'JSON path {json_pathname} not found')
        SystemExit
    return jdict


def load_simulator_cfgs(dict_args) -> Dict:
    for simulator_name in get_simulator_names():
        if simulator_name in dict_args:
            simulator_cfg = handle_simulator_cfgs(simulator_name, dict_args[simulator_name])
            dict_args[simulator_name] = simulator_cfg
    return dict_args


def merge_json_into_dict(old_dict: Dict, json_files: Iterable):
    for jfile in json_files:
        json_dict = json.load(open(jfile))
        old_dict = merge_dicts(old_dict, json_dict)
    return old_dict


def merge_dicts(*dict_likes):
    all_dicts = {}
    for dict_like in dict_likes:
        if type(dict_like) == dict:
            all_dicts = {**all_dicts, **dict_like}
    return all_dicts


def retrieve_default_args() -> Dict:
    fn = "scripts/common/default_args.json"
    default_args = json.load(open(fn))
    return default_args


def update_namespace_with_dict(args, updater_dict: Dict):
    vars(args).update(updater_dict)
    return args
