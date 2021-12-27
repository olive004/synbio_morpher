import json
import argparse
from typing import Dict


def parse_args(config_file: str = None, all_args: Dict = None) -> Dict:

    if all_args is None:
        all_args = retrieve_default_args()

    if config_file is not None:
        configs = json.load(open(config_file))

        all_args = all_args | configs
    parser = argparse.ArgumentParser()
    update_namespace_with_dict(parser.parse_args(), all_args)
    return all_args


def retrieve_default_args() -> Dict:

    fn = "scripts/common/default_args.json"
    default_args = json.load(open(fn))
    return default_args


def update_namespace_with_dict(args: Namespace, updater_dict: Dict):
    for k in updater_dict:
        vars(args).update(updater_dict[k])