

from copy import deepcopy
from ctypes import Union
import glob
import logging
import os
from tracemalloc import start
from typing import Tuple
import numpy as np
import pandas as pd

from src.srv.io.loaders.misc import load_csv
from src.utils.data.data_format_tools.common import load_json_as_dict, make_iterable_like
from src.utils.misc.errors import ConfigError
from src.utils.misc.io import get_pathnames, get_subdirectories


def get_purposes(script_dir=None):
    script_dir = 'scripts' if script_dir is None else script_dir
    return get_subdirectories(script_dir, only_basedir=True)


def get_purpose_from_pathname(pathname):
    split_path = pathname.split(os.sep)
    if 'data' in split_path:
        top_dir = 'data'
    elif 'script' in split_path:
        top_dir = 'script'
    else:
        logging.warning(f'The supplied pathname {pathname} most likely does not '
                        'point to a purpose.')
        top_dir = None
    try:
        purpose_idx = split_path.index(top_dir) + 1
        return split_path[purpose_idx]
    except ValueError:
        raise ValueError(f'Could not find purpose for script path {pathname}.')


def get_purpose_from_cfg(cfg, experiment_folder):
    if cfg.get('experiment') is None:
        try:
            purpose = cfg['purpose']
        except KeyError:
            purpose = get_most_likely_purpose(experiment_folder)
    else:
        purpose = cfg['experiment']['purpose']
    return purpose


def get_most_likely_purpose(starting_experiment_dir):
    experiment_folder = get_root_experiment_folder(starting_experiment_dir)
    return get_purpose_from_pathname(experiment_folder)


# retrieve_search_dir
def get_search_dir(config_file: dict, config_search_key: str = None,
                   modify_config_for_posterity: bool = True) -> Tuple[dict, str]:
    def find_config_search_key(config_search_key):
        if config_search_key is None:
            config_search_key = [k for k, v in config_file.items() if type(
                v) == dict and 'source_dir' in v.keys()]
            if len(config_search_key) > 1:
                logging.warning(f'Detected multiple candidate search directories as keys: {config_search_key}'
                                f'\nFor config file {config_file}')
            elif not config_search_key:
                raise KeyError(
                    f'Could not find search directory with "source_dir" for config {config_file}.')
            else:
                config_search_key = config_search_key[0]
        return config_search_key
    config_search_key = find_config_search_key(config_search_key)
    search_config = config_file.get(config_search_key, {})
    if type(search_config) == str:
        return config_file, search_dir
    if not search_config:
        raise KeyError(
            f'Could not find {config_search_key} in config keys: {config_file.keys()}.')
    update = search_config.get(
        "is_source_dir_incomplete", None)
    if update:
        search_dir = os.path.join(search_config.get("source_dir"),
                                  get_recent_experiment_folder(search_config.get(
                                      "source_dir")), search_config.get("purpose_to_get_source_dir_from"))
        if not os.path.isdir(search_dir):
            raise ConfigError(f'Could not find directory {search_dir}')
        if modify_config_for_posterity:
            config_file[config_search_key]['source_dir_actually_used_POSTERITY'] = search_dir
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


def load_result_report(local_experiment_folder: str, result_type: str = 'signal'):
    def process_pre_result_report(jdict):
        iterable_like = make_iterable_like(jdict)
        for k, v in iterable_like:
            if type(v) == list:
                jdict[k] = process_pre_result_report(v)
            if type(v) == str:
                jdict[k] = np.float32(v)
        return jdict
    report_path = get_pathnames(local_experiment_folder, file_key=['report', result_type],
                                first_only=True)
    return process_pre_result_report(load_json_as_dict(report_path))


def load_experiment_report(experiment_folder: str) -> dict:
    experiment_folder = get_root_experiment_folder(experiment_folder)
    report_path = os.path.join(experiment_folder, 'experiment.json')
    return load_json_as_dict(report_path)


def load_experiment_config_original(starting_experiment_folder: str, target_purpose: str) -> dict:
    """ Load the experiment config from a previous experiment that led
    to the current (starting) experiment folder"""
    original_config = load_experiment_config(starting_experiment_folder)
    current_purpose = get_purpose_from_cfg(
        original_config, starting_experiment_folder)
    while not current_purpose == target_purpose:
        try:
            original_config, original_source_dir = get_search_dir(
                config_file=original_config)
        except ConfigError:
            raise ConfigError('Could not find the original configuration file used '
                              f'for purpose {target_purpose} when starting from '
                              f'experiment folder {starting_experiment_folder}.')
        original_config = load_experiment_config(
            original_source_dir)
        current_purpose = get_purpose_from_cfg(
            original_config, starting_experiment_folder)
    if not current_purpose == target_purpose:
        logging.warning(f'Loaded wrong config from {original_source_dir} with purpose '
                        f'{current_purpose}')
    return original_config


def load_experiment_config(experiment_folder: str) -> dict:
    if experiment_folder is None:
        raise ValueError('If trying to load something from the experiment config, please supply '
                         f'a valid directory for the source experiment instead of {experiment_folder}')
    experiment_report = load_experiment_report(experiment_folder)
    experiment_config = experiment_report.get('config_params')
    logging.info(experiment_config)
    return experiment_config


def get_recent_experiment_folder(purpose_folder: str) -> str:
    return sorted(os.listdir(purpose_folder))[-1]


def get_path_from_output_summary(name, output_summary: pd.DataFrame = None, experiment_folder: str = None):
    if output_summary is None:
        assert experiment_folder, f'No experiment path given, cannot find experiment summary.'
        output_summary = load_experiment_output_summary(experiment_folder)
    pathname = output_summary.loc[output_summary['out_name']
                                  == name]['out_path'].values[0]
    return pathname
