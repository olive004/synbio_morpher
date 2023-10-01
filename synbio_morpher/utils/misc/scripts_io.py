
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    


from copy import deepcopy
import logging
import os
from typing import Tuple
import numpy as np
import pandas as pd

from synbio_morpher.srv.io.manage.sys_interface import SCRIPT_DIR, DATA_DIR, PACKAGE_DIR, PACKAGE_NAME
from synbio_morpher.srv.io.loaders.misc import load_csv
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict, make_iterable_like
from synbio_morpher.utils.misc.errors import ConfigError, ExperimentError
from synbio_morpher.utils.misc.io import get_pathnames, get_subdirectories
from synbio_morpher.utils.misc.type_handling import flatten_listlike


def get_purposes(script_dir=None):
    script_module = __import__('.'.join([PACKAGE_NAME, SCRIPT_DIR]), fromlist=[''])
    script_dir = script_module.__path__[0] if script_dir is None else script_dir
    return get_subdirectories(script_dir, only_basedir=True)


def get_purpose_from_pathname(pathname):
    split_path = pathname.split(os.sep)
    if DATA_DIR in split_path:
        top_dir = DATA_DIR
    elif SCRIPT_DIR in split_path:
        top_dir = SCRIPT_DIR
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


def find_config_searchdir_key(config: dict, config_searchdir_key: str, source_dir_key: str):
    if config_searchdir_key is None:
        if source_dir_key in list(config.keys()):
            return source_dir_key
        config_searchdir_key = [k for k, v in config.items() if type(
            v) == dict and source_dir_key in v.keys()]
        if len(config_searchdir_key) > 1:
            logging.warning(f'Detected multiple candidate search directories as keys: {config_searchdir_key}'
                            f'\nFor config file {config}')
        elif not config_searchdir_key:
            raise KeyError(
                f'Could not find search directory with {source_dir_key} for config {config}.')
        else:
            config_searchdir_key = config_searchdir_key[0]
    else:
        if config_searchdir_key in list(config.values()):
            return config_searchdir_key
    return config_searchdir_key


def get_search_dir(config_file: dict, config_searchdir_key: str = None,
                   modify_config_for_posterity: bool = True) -> Tuple[dict, str]:
    """ When a specific data folder is to be loaded, this can be specified
    explicitly in the config, even if the exact data folder is not known at 
    the start of runtime. This function helps retrieve the data folder in both cases. """
    source_dir_key = 'source_dir'
    config_searchdir_key = find_config_searchdir_key(
        config_file, config_searchdir_key, source_dir_key)
    sourcing_config = config_file.get(config_searchdir_key, {})
    if type(sourcing_config) == str:
        return config_file, sourcing_config
    if not sourcing_config:
        raise KeyError(
            f'Could not find {config_searchdir_key} in config keys: {config_file.keys()}.')
    update = sourcing_config.get(
        "is_source_dir_incomplete", None)
    if update:
        if sourcing_config.get("purpose_to_get_source_dir_from") is None:
            logging.warning('The source directory is incomplete = True, but no source dir + purpose pair was supplied to find the correct directory.')
        if sourcing_config.get('source_dir_actually_used_POSTERITY') is not None:
            return config_file, sourcing_config['source_dir_actually_used_POSTERITY']

        source_dir = os.path.join(sourcing_config.get(source_dir_key),
                                get_recent_experiment_folder(sourcing_config.get(
                                    source_dir_key)), sourcing_config.get("purpose_to_get_source_dir_from"))
            
        if not os.path.isdir(source_dir):
            raise ConfigError(
                f'Could not find directory {source_dir} (maybe it is not the most recent experiment directory anymore)')
        if modify_config_for_posterity:
            config_file[config_searchdir_key]['source_dir_actually_used_POSTERITY'] = source_dir
        return config_file, source_dir
    else:
        source_dir = sourcing_config[source_dir_key]
        return config_file, source_dir


def get_root_experiment_folder(miscpath: str):
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
        if len(os.path.split(miscpath)) == 1 or len(miscpath) == 0:
            raise ExperimentError(
                f'Root experiment folder not found recursively in base {miscpath}')
        experiment_folder = get_root_experiment_folder(
            os.path.dirname(miscpath))
    return experiment_folder


def get_subprocesses_dirnames(source_dir):
    experiment_folder = get_root_experiment_folder(source_dir)
    return get_pathnames(experiment_folder, file_key='subprocess', conditional='directories')


def load_experiment_output_summary(experiment_folder) -> pd.DataFrame:
    if not os.path.isdir(experiment_folder):
        experiment_folder = os.path.join(PACKAGE_DIR, experiment_folder) if PACKAGE_DIR not in experiment_folder else experiment_folder
        if not os.path.isdir(experiment_folder):
            raise ValueError(f'Invalid experiment folder {experiment_folder} loaded. Check for spelling errors in pathnames supplied in config.')
    summary_path = os.path.join(experiment_folder, 'output_summary.csv')
    if os.path.isfile(summary_path):
        output_summary = load_csv(summary_path)
    else:
        output_summary = make_output_summary(experiment_folder)
    return output_summary


def load_result_report(local_experiment_folder: str, result_type: str = 'signal', index=slice(None)):
    def process_pre_result_report(jdict):
        iterable_like = make_iterable_like(jdict)
        for k, v in iterable_like:
            if type(v) == list:
                jdict[k] = np.asarray(flatten_listlike(v, safe=True))[index]
                # jdict[k] = process_pre_result_report(v)
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
        if type(original_source_dir) == list and len(original_source_dir) == 1:
            logging.warning(
                f'Expected string for {original_source_dir} but got list')
            original_source_dir = original_source_dir[0]
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
        raise ExperimentError('If trying to load a file from the experiment config, please supply '
                              f'a valid directory for the source experiment instead of {experiment_folder}')
    experiment_report = load_experiment_report(experiment_folder)
    try:
        experiment_config = experiment_report['config_params']
    except KeyError:
        raise KeyError(
            f'Could not retrieve original configs from experiment, instead got {experiment_report}')
    return experiment_config


def make_output_summary(experiment_folder: str) -> pd.DataFrame:
    output_summary_all = pd.DataFrame
    for file in os.path.join(experiment_folder):
        output_summary = {}
        if os.path.isfile(file):
            output_summary['out_name'] = os.path.basename(file)
            output_summary['out_path'] = os.path.join(experiment_folder, file)
            output_summary['filename_addon'] = os.path.splitext(file)[1]
            output_summary['out_type'] = os.path.splitext(file)[1]
            output_summary['name'] = os.path.basename(file)
            purposes = [p for p in experiment_folder.split(os.path.sep) if p in get_purposes()]
            output_summary['subdir'] = purposes[-1] if purposes else None
        output_summary_all = pd.concat([output_summary_all, pd.DataFrame.from_dict(output_summary)])
    return output_summary_all


def get_recent_experiment_folder(purpose_folder: str) -> str:
    return sorted(os.listdir(purpose_folder))[-1]


def get_path_from_output_summary(name, output_summary: pd.DataFrame = None, experiment_folder: str = None):
    if output_summary is None:
        assert experiment_folder, f'No experiment path given, cannot find experiment summary.'
        output_summary = load_experiment_output_summary(experiment_folder)
    pathname = output_summary.loc[output_summary['out_name']
                                  == name]['out_path'].values[0]
    return pathname
