import json
from typing import Dict


def parse_kwargs(config_file: str = None) -> Dict:

    all_args = retrieve_default_args()

    if config_file is not None:
        with open(config_file) as json_file:
            configs = json.load(json_file)

        all_args = all_args | configs
    return all_args


def retrieve_default_args() -> Dict:
    
    fn = "scripts/common/default_args.json"
    with open(fn) as json_file:
        default_args = json.load(json_file)
    return {}