


import logging
import os

import numpy as np
from src.srv.io.loaders.data_loader import DataLoader
from src.srv.io.results.result_writer import ResultWriter
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.scripts_io import get_search_dir, get_subprocesses_dirnames, load_experiment_config


def main(config=None, data_writer=None):

    if config is None:
        config = os.path.join(
            'scripts', 'stitch_parameter_grid', 'configs', 'base_config.json')
    config_file = load_json_as_dict(config)
    if data_writer is None:
        data_writer = ResultWriter(purpose=config_file[
            'experiment'].get('purpose', 'stitch_parameter_grid'))

    # Load in parameter grids

    config_file, source_dir = get_search_dir('source_parameter_dir', config_file=config_file)
    experiment_config = load_experiment_config(source_dir)
    experiment_settings = experiment_config['experiment']
    logging.info(experiment_config)
    num_subprocesses = 1
    if experiment_settings['parallelise']:
        num_subprocesses = experiment_settings['num_subprocesses']
    
    # If there was multithreading, load each parameter_grid one by one from subfolders
    parameter_grids = []
    for subprocess in range(num_subprocesses):
        subproccess_dir = get_subprocesses_dirnames(source_dir)
        parameter_grids.append(DataLoader().load_data(source_dir))

    logging.info(parameter_grids)

    # stitch them together 
        # Find the starting and ending indices
        
    matrix_size = np.size(parameter_grids[0])

            # If there was multithreading, indices will be different


    # Write full matrices