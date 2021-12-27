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


def parse_args(config_file: str = None, all_args: Dict = None) -> Dict:

    if all_args is None:
        all_args = retrieve_default_args()

    if config_file is not None:
        configs = json.load(open(config_file))

        all_args = all_args | configs
    parser = argparse.ArgumentParser()
    args_namespace = parser.parse_args()
    update_namespace_with_dict(args_namespace, all_args)
    return args_namespace


def parse_simulation_args(dict_args):
    simulators = get_simulator_names()
    config_path = {}
    for simulator_name in simulators:
        if simulator_name in dict_args:
            config_path[simulator_name] = dict_args[simulator_name]
            try:
                jf = json.load(config_path[simulator_name])
            except FileNotFoundError:
                logging.error(f'Path to simulator {simulator_name} not found')
    return config_path


def retrieve_default_args() -> Dict:
    fn = "scripts/common/default_args.json"
    default_args = json.load(open(fn))
    return default_args


def update_namespace_with_dict(args, updater_dict: Dict):
    vars(args).update(updater_dict)
    return args