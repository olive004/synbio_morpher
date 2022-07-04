

from copy import deepcopy
from ctypes import Union
import glob
import logging
import os
from tracemalloc import start
from typing import Tuple
import pandas as pd

from src.srv.io.loaders.misc import load_csv
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.errors import ConfigError
from src.utils.misc.io import get_pathnames, get_subdirectories


def get_purposes(script_dir=None):
    script_dir = 'scripts' if script_dir is None else script_dir
    return get_subdirectories(script_dir, only_basedir=True)


# retrieve_search_dir
def get_search_dir(config_search_key: str, config_file: dict, modify_config_for_posterity=True) -> Tuple[dict, str]:
    search_config = config_file.get(config_search_key, {})
    if not search_config:
        raise KeyError(
            f'Could not find {config_search_key} in config keys: {config_file.keys()}.')
    update = search_config.get(
        "is_source_dir_incomplete", None)
    if update:
        search_dir = os.path.join(search_config.get("source_dir"),
                                  get_recent_experiment_folder(search_config.get(
                                      "source_dir")), search_config.get("purpose_of_ensembled_source_dir"))
        if not os.path.isdir(search_dir):
            raise ConfigError(f'Could not find directory {search_dir}')
        if modify_config_for_posterity:
            config_file[config_search_key]['source_dir_actually_used_if_incomplete'] = search_dir
        return config_file, search_dir
    else:
        search_dir = search_config.get('source_dir')
        return config_file, search_dir


def get_root_experiment_folder(miscpath):
    split_path = miscpath.split(os.sep)
    purposes = [p for p in split_path if p in get_purposes()]
    if len(purposes) == 1:
        target_top_dir = os.path.join(
            *split_path[:split_path.index(purposes[0])+1])
        experiment_folder = deepcopy(miscpath)
        while not os.path.dirname(experiment_folder) == target_top_dir:
            experiment_folder = os.path.dirname(experiment_folder)
    elif len(purposes) == 2:
        experiment_folder = os.path.join(
            *split_path[:split_path.index(purposes[1])+1])
    else:
        if len(os.path.split(miscpath)) == 1:
            raise ValueError(
                f'Root experiment folder not found recursively in base {miscpath}')
        experiment_folder = get_root_experiment_folder(
            os.path.dirname(miscpath))
    return experiment_folder


def get_subprocesses_dirnames(source_dir):
    experiment_folder = get_root_experiment_folder(source_dir)
    return get_pathnames(experiment_folder, file_key='subprocess', conditional='directories')


def load_experiment_output_summary(experiment_folder) -> pd.DataFrame:
    summary_path = os.path.join(experiment_folder, 'output_summary.csv')
    return load_csv(summary_path)


def load_experiment_report(experiment_folder: str) -> dict:
    experiment_folder = get_root_experiment_folder(experiment_folder)
    report_path = os.path.join(experiment_folder, 'experiment.json')
    return load_json_as_dict(report_path)


def load_experiment_config_original(starting_experiment_folder: str, target_purpose: str) -> dict:
    original_config = load_experiment_config(starting_experiment_folder)
    while not original_config['experiment']['purpose'] == target_purpose:
        try:
            original_config, original_source_dir = get_search_dir(
                original_config)
        except ConfigError:
            raise ConfigError('Could not find the original configuration file used '
                              f'for purpose {target_purpose} when starting from '
                              f'experiment folder {starting_experiment_folder}.')
        original_config = load_experiment_config(
            original_source_dir)
    if not original_config['experiment']['purpose'] == 'parameter_based_simulation':
        logging.warning(f'Loaded wrong config from {original_source_dir} with purpose '
                        f'{original_config["experiment"]["purpose"]}')
    return original_config


def load_experiment_config(experiment_folder: str) -> dict:
    if experiment_folder is None:
        raise ValueError('If trying to load something from the experiment config, please supply '
                         f'a valid directory for the source experiment instead of {experiment_folder}')
    experiment_report = load_experiment_report(experiment_folder)
    return load_json_as_dict(experiment_report.get('config_filepath'))


def get_recent_experiment_folder(purpose_folder: str) -> str:
    return sorted(os.listdir(purpose_folder))[-1]


def get_path_from_output_summary(name, output_summary: pd.DataFrame = None, experiment_folder: str = None):
    if output_summary is None:
        assert experiment_folder, f'No experiment path given, cannot find experiment summary.'
        output_summary = load_experiment_output_summary(experiment_folder)
    pathname = output_summary.loc[output_summary['out_name']
                                  == name]['out_path'].values[0]
    return pathname
