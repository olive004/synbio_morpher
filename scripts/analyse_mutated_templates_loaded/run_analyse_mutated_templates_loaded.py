from functools import partial
import logging
import os

import numpy as np
import pandas as pd

from fire import Fire
from src.utils.misc.io import get_pathnames_from_mult_dirs
from src.utils.misc.scripts_io import load_experiment_config, load_experiment_config_original

from src.utils.results.experiments import Experiment, Protocol
from src.utils.results.result_writer import ResultWriter
from src.utils.results.visualisation import visualise_data
from src.srv.parameter_prediction.simulator import SIMULATOR_UNITS, RawSimulationHandling
from src.srv.sequence_exploration.sequence_analysis import tabulate_mutation_info
from src.utils.data.data_format_tools.common import load_csv_mult, load_json_as_dict


def main(config=None, data_writer=None):
    # Set configs
    if config is None:
        config = os.path.join(
            "scripts", "analyse_mutated_templates_loaded", "configs", "base_config.json")
    config_file = load_json_as_dict(config)

    # Start_experiment
    if data_writer is None:
        data_writer = ResultWriter(purpose=config_file.get('experiment', {}).get('purpose', 'analyse_mutated_templates_loaded'))

    source_dirs = config_file.get('source_dirs', [])

    protocols = [
        Protocol(
            partial(
                get_pathnames_from_mult_dirs,
                search_dirs=source_dirs,
                file_key='tabulated_mutation_info.csv',
                first_only=True),
            req_output=True,
            name='get_pathnames_from_mult_dirs'
        ),
        Protocol(
            load_csv_mult,
            req_input=True,
            req_output=True,
            name='load_csv'
        ), 
        Protocol(
            pd.concat,
            req_input=True,
            req_output=True,
            name='concatenate_csvs'
        )
        # ),
        # Protocol(
        #     visualise_data
        # )
    ]

    source_dir = source_dirs[0]
    source_config = load_experiment_config_original(source_dir, 'analyse_mutated_templates')
    # source_config = load_experiment_config(source_dir)

    if config_file.get('only_visualise_circuits', False):
        exclude_rows_via_cols = ['mutation_name']
    else:
        exclude_rows_via_cols = []

    num_mutations = source_config['mutations']['mutation_nums_within_sequence']
    plot_grammar = 's' if num_mutations > 1 else ''

    binding_rates_threshold_upper = np.power(10, 6)
    binding_rates_threshold_upper_text = f', with cutoff at {binding_rates_threshold_upper}' if binding_rates_threshold_upper else ''
    
    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
