import argparse
import json
import logging
import os
from typing import Dict, Iterable


def get_simulator_names():
    simulator_dir = os.join("src", "utils", "parameter_prediction")
    simulators = {f for f in os.walk(simulator_dir)}
    return simulators


def parse_args(config_file: str = None, dict_args: Dict = None) -> Dict:

    if dict_args is None:
        dict_args = retrieve_default_args()

    pooled_cfgs = merge_dicts(load_simulator_cfgs(dict_args), config_file)
    dict_args = merge_json_into_dict(dict_args, simulator_cfgs.values())
    parser = argparse.ArgumentParser()
    args_namespace = parser.parse_args()
    update_namespace_with_dict(args_namespace, dict_args)
    return args_namespace


def load_simulator_cfgs(dict_args) -> Dict:
    simulators = get_simulator_names()
    config_path = {}
    for simulator_name in simulators:
        if simulator_name in dict_args:
            config_path[simulator_name] = dict_args[simulator_name]
            try:
                jf = json.load(open(config_path[simulator_name]))
            except FileNotFoundError:
                logging.error(f'Path to simulator {simulator_name} not found')
    return config_path


def merge_json_into_dict(old_dict: Dict, json_files: Iterable):
    for jfile in json_files:
        json_dict = json.load(open(jfile))
        old_dict = old_dict | json_dict
    return old_dict


def merge_dicts(*dict_likes):
    all_dicts = {}
    for dict_like in dict_likes:
        if type(dict_like) == dict:
            all_dicts = all_dicts | dict_like
    return all_dicts


def retrieve_default_args() -> Dict:
    fn = "scripts/common/default_args.json"
    default_args = json.load(open(fn))
    return default_args


def update_namespace_with_dict(args, updater_dict: Dict):
    vars(args).update(updater_dict)
    return args
