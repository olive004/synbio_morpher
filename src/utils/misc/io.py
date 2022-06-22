

from copy import deepcopy
import glob
import logging
import os
import pandas as pd

from src.srv.io.loaders.misc import load_csv
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.string_handling import remove_file_extension


def isolate_filename(filepath: str):
    if type(filepath) == str:
        return os.path.splitext(os.path.basename(filepath))[0]
    return None


def get_subdirectories(parent_dir, only_basedir=False):
    # return [name for name in os.listdir(parent_dir)
    #         if os.path.isdir(os.path.join(parent_dir, name))]
    subdirectories = [f.path for f in os.scandir(parent_dir) if f.is_dir()]
    if only_basedir:
        return [os.path.basename(s) for s in subdirectories]
    return subdirectories


def create_location(pathname):
    if not os.path.isdir(pathname):
        os.umask(0)
        os.makedirs(pathname, mode=0o777)


def get_pathnames(search_dir, file_key='', first_only=False, allow_empty=False, optional_subdir=''):
    if type(file_key) == list:
        all_path_names = []
        for fk in file_key:
            all_path_names.append(
                set(sorted(glob.glob(os.path.join(search_dir, '*' + fk + '*'))))
            )
        path_names = list(all_path_names[0].intersection(*all_path_names[1:]))
    elif not file_key:
        path_names = sorted([os.path.join(search_dir, f) for f in os.listdir(
            search_dir) if os.path.isfile(os.path.join(search_dir, f))])
    else:
        path_names = sorted([f for f in glob.glob(os.path.join(
            search_dir, '*' + file_key + '*')) if os.path.isfile(f)])
    if first_only and path_names:
        path_names = path_names[0]
    if not path_names and optional_subdir:
        path_names = get_pathnames(os.path.join(search_dir, optional_subdir), file_key=file_key,
                                   first_only=first_only, allow_empty=allow_empty)
    if not path_names and not allow_empty:
        raise ValueError(
            f'Could not find file matching "{file_key}" in {search_dir}.')
    return path_names


def get_purposes(script_dir=None):
    script_dir = 'scripts' if script_dir is None else script_dir
    return get_subdirectories(script_dir, only_basedir=True)


def get_search_dir(search_config_key: str, config_file: dict):
    search_config = config_file.get(search_config_key, {})
    update = search_config.get(
        "is_source_dir_incomplete", None)
    if update:
        search_dir = os.path.join(search_config.get("source_dir"),
                                  get_recent_experiment_folder(search_config.get(
                                      "source_dir")), search_config.get("purpose_of_ensembled_source_dir"))
        assert os.path.isdir(
            search_dir), f'Could not find directory {search_dir}'
        config_file[search_config_key]['source_dir_actually_used_if_incomplete'] = search_dir
        return config_file, search_dir
    elif update == None:
        raise KeyError(
            f'Could not find {search_config_key} in config keys: {config_file.keys()}.')
    else:
        search_dir = search_config.get('search_dir')
        return config_file, search_dir


def get_root_experiment_folder(miscpath):
    split_path = miscpath.split(os.sep)
    purposes = [p for p in split_path if p in get_purposes()]
    if len(purposes) == 1:
        target_top_dir = os.path.join(*split_path[:split_path.index(purposes[0])+1])
        experiment_folder = deepcopy(miscpath)
        while not os.path.dirname(experiment_folder) == target_top_dir:
            experiment_folder = os.path.dirname(experiment_folder)
    elif len(purposes) == 2:
        experiment_folder = os.path.join(*split_path[:split_path.index(purposes[1])+1])
    else:
        if len(os.path.split(miscpath)) == 1:
            raise ValueError(
                f'Root experiment folder not found recursively in base {miscpath}')
        experiment_folder = get_root_experiment_folder(os.path.dirname(miscpath))
    return experiment_folder


def load_experiment_output_summary(experiment_folder) -> pd.DataFrame:
    summary_path = os.path.join(experiment_folder, 'output_summary.csv')
    return load_csv(summary_path)


def load_experiment_report(experiment_folder: str) -> dict:
    experiment_folder = get_root_experiment_folder(experiment_folder)
    report_path = os.path.join(experiment_folder, 'experiment.json')
    return load_json_as_dict(report_path)


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


def convert_pathname_to_module(filepath: str):
    filepath = remove_file_extension(filepath)
    return os.path.normpath(filepath).replace(os.sep, '.')


def import_module_from_path(module_name: str, filepath: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    return importlib.util.module_from_spec(spec)
