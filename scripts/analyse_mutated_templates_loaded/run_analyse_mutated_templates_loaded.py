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
    source_dir = source_dirs[0]
    source_config = load_experiment_config_original(source_dir, 'mutation_effect_on_interactions_signal')
    
    # binding_rates_threshold_upper = np.power(10, 6)
    binding_rates_threshold_upper = None
    binding_rates_threshold_upper_text = f', with cutoff at {binding_rates_threshold_upper}' if binding_rates_threshold_upper else ''

    if config_file.get('only_visualise_circuits', False):
        exclude_rows_via_cols = ['mutation_name']
    else:
        exclude_rows_via_cols = []

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
            partial(
                load_csv_mult,
                
            ),
            req_input=True,
            req_output=True,
            name='load_csv'
        ), 
        Protocol(
            pd.concat,
            req_input=True,
            req_output=True,
            name='concatenate_csvs'
        ),
        # precision log
        Protocol(
            partial(
                visualise_data,
                data_writer=data_writer, cols_x=['precision_diff_to_base_circuit'],
                plot_type='histplot',
                out_name='precision_diff_log',
                exclude_rows_nonempty_in_cols=exclude_rows_via_cols,
                exclude_rows_zero_in_cols=['mutation_num'],
                hue=['mutation_num'],
                log_axis=(True, False),
                use_sns=True,
                expand_coldata_using_col_x=True,
                column_name_for_expanding_labels='sample_names',
                idx_for_expanding_labels=0,
                title=f'Precision difference between circuit\nand mutated counterparts',
                xlabel='Precision difference'
            ),
            req_input=True,
            name='visualise_interactions_difference',
            skip=config_file.get('only_visualise_circuits', False)
        ),
        # Binding rates min int's mutations
        Protocol(
            partial(visualise_data, data_writer=data_writer, cols_x=['binding_rates_min_interaction'],
                    plot_type='histplot',
                    out_name='binding_rates_min_freqs_mutations_logs',
                    exclude_rows_nonempty_in_cols=exclude_rows_via_cols,
                    threshold_value_max=binding_rates_threshold_upper,
                    misc_histplot_kwargs={
                        "hue": ['mutation_num'],
                        "multiple": "stacked"},
                    log_axis=(True, False),
                    use_sns=True,
                    title=f'Minimum ' + r'$k_d$' + ' strength',
                    xlabel='Dissociation rate ' + r'$k_d$' + ' (' +
                    f'{SIMULATOR_UNITS[source_config["interaction_simulator"]["name"]]["rate"]})' +
                    f'{binding_rates_threshold_upper_text}'),
            req_input=True,
            name='visualise_mutated_interactions'
        ),
    ]

    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)