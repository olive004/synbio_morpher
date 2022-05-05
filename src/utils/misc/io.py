

import glob
import logging
import os
import pandas as pd

from src.srv.io.loaders.misc import load_csv
from src.utils.data.data_format_tools.common import load_json_as_dict


def isolate_filename(filepath: str):
    return os.path.splitext(os.path.basename(filepath))[0]


def get_subdirectories(parent_dir):
    # return [name for name in os.listdir(parent_dir)
    #         if os.path.isdir(os.path.join(parent_dir, name))]
    return [f.path for f in os.scandir(parent_dir) if f.is_dir()]



def create_location(pathname):
    if not os.path.isdir(pathname):
        os.umask(0)
        os.makedirs(pathname, mode=0o777)


def get_pathnames(file_key, search_dir, first_only=False):
    path_names = glob.glob(os.path.join(search_dir, '*' + file_key + '*'))
    if first_only and path_names:
        path_names = path_names[0]
    if not path_names:
        raise ValueError(
            f'Could not find file matching "{file_key}" in {search_dir}.')
    return path_names


def load_experiment_output_summary(experiment_folder) -> pd.DataFrame:
    summary_path = os.path.join(experiment_folder, 'output_summary.csv')
    return load_csv(summary_path)


def load_experiment_report(experiment_folder):
    report_path = os.path.join(experiment_folder, 'experiment.json')
    return load_json_as_dict(report_path )


def get_path_from_exp_summary(name, experiment_summary: pd.DataFrame = None, experiment_folder: str = None):
    if experiment_summary is None:
        assert experiment_folder, f'No experiment path given, cannot find experiment summary.'
        experiment_summary = load_experiment_output_summary(experiment_folder)
    pathname = experiment_summary.loc[experiment_summary['out_name'] == name]['out_path'].values[0]
    return pathname
