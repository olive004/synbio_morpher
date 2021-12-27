import argparse
import json
import logging
import os
from typing import Dict


def get_simulator_names():
    logging.debug(os.getcwd())
    simulator_dir = os.join(os.getcwd(), "src", "utils", "parameter_prediction")
    simulators = {f for f in os.walk(simulator_dir)}
    return simulators


def parse_args(config_file: str = None, dict_args: Dict = None) -> Dict:

    if dict_args is None:
        dict_args = retrieve_default_args()

    simulator_cfgs = load_simulator_cfgs(dict_args)
    simulator_cfgs = simulator_cfgs | config_file if config_file is not None else simulator_cfgs
    dict_args = merge_dict_with_json(dict_args)
    for sim, config_file in simulator_cfgs:
        configs = json.load(open(config_file))
        dict_args = dict_args | configs
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


def merge_dict_with_json(old_dict: Dict, json_files):
    for jfile in json_files:
        json_dict = json.load(open(jfile))
        old_dict = old_dict | json_dict
    return old_dict


def retrieve_default_args() -> Dict:
    fn = "scripts/common/default_args.json"
    default_args = json.load(open(fn))
    return default_args


def update_namespace_with_dict(args, updater_dict: Dict):
    vars(args).update(updater_dict)
    return args
