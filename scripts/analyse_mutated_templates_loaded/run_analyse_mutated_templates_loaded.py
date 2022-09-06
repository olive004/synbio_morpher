from functools import partial
import logging
import os

import numpy as np
import pandas as pd

from fire import Fire
from src.utils.misc.io import get_pathnames_from_mult_dirs
from src.utils.misc.scripts_io import load_experiment_config, load_experiment_config_original
from src.utils.misc.string_handling import prettify_keys_for_label
from src.utils.results.analytics.timeseries import Timeseries

from src.utils.results.experiments import Experiment, Protocol
from src.utils.results.result_writer import ResultWriter
from src.utils.results.visualisation import visualise_data
from src.srv.parameter_prediction.simulator import SIMULATOR_UNITS, RawSimulationHandling
from src.srv.sequence_exploration.sequence_analysis import get_mutation_info_columns, tabulate_mutation_info
from src.utils.data.data_format_tools.common import load_json_as_dict, load_json_mult


def main(config=None, data_writer=None):
    # Set configs
    if config is None:
        config = os.path.join(
            "scripts", "analyse_mutated_templates_loaded", "configs", "base_config.json")
    config_file = load_json_as_dict(config)

    # Start_experiment
    if data_writer is None:
        data_writer = ResultWriter(purpose=config_file.get(
            'experiment', {}).get('purpose', 'analyse_mutated_templates_loaded'))

    source_dirs = config_file.get('source_dirs', [])
    source_dir = source_dirs[0]
    source_config = load_experiment_config_original(
        source_dir, 'mutation_effect_on_interactions_signal')

    # binding_rates_threshold_upper = np.power(10, 6)
    binding_rates_threshold_upper = None
    binding_rates_threshold_upper_text = f', with cutoff at {binding_rates_threshold_upper}' if binding_rates_threshold_upper else ''

    def readout(v):
        logging.info(v)
        return v

    protocols = [
        Protocol(
            partial(
                get_pathnames_from_mult_dirs,
                search_dirs=source_dirs,
                file_key='tabulated_mutation_info.json',
                first_only=True),
            req_output=True,
            name='get_pathnames_from_mult_dirs'
        ),
        Protocol(
            partial(
                load_json_mult,
                as_type=pd.DataFrame
            ),
            req_input=True,
            req_output=True,
            name='load_json'
        ),
        Protocol(
            partial(
                pd.concat,
                axis=0,
                ignore_index=True),
            req_input=True,
            req_output=True,
            name='concatenate_dfs'
        ),
        # Binding rates min int's mutations
        Protocol(
            partial(
                visualise_data, data_writer=data_writer, cols_x=[
                    'binding_rates_min_interaction'],
                plot_type='histplot',
                out_name='binding_rates_min_freqs_mutations_logs',
                threshold_value_max=binding_rates_threshold_upper,
                exclude_rows_zero_in_cols=['mutation_num'],
                misc_histplot_kwargs={
                    "hue": 'mutation_num',
                    "multiple": "dodge",
                    "element": "poly"},
                log_axis=(True, False),
                use_sns=True,
                title=f'Minimum ' + r'$k_d$' + ' strength',
                xlabel='Dissociation rate ' + r'$k_d$' + ' (' +
                fr'{SIMULATOR_UNITS[source_config["interaction_simulator"]["name"]]["rate"]})' +
                f'{binding_rates_threshold_upper_text}'),
            req_input=True,
            name='visualise_mutated_interactions'
        ),
    ]

    # Visualisations

    # Binding rates
    rate_unit = r'{}'.format(SIMULATOR_UNITS[source_config["interaction_simulator"]["name"]]["rate"])
    protocols.append(Protocol(
        partial(
            visualise_data, data_writer=data_writer, cols_x=[
                'binding_rates_min_interaction'],
            plot_type='histplot',
            out_name='binding_rates_min_freqs_mutations_logs',
            threshold_value_max=binding_rates_threshold_upper,
            exclude_rows_zero_in_cols=['mutation_num'],
            misc_histplot_kwargs={
                "hue": 'mutation_num',
                "multiple": "dodge",
                "element": "poly"},
            log_axis=(True, False),
            use_sns=True,
            title=f'Minimum ' + r'$k_d$' + ' strength',
            xlabel='Dissociation rate ' + r'$k_d$' + ' (' +
            f'{rate_unit})' +
            f'{binding_rates_threshold_upper_text}'),
        req_input=True,
        name='visualise_mutated_interactions'
    ))
    # Plot mean of interacting numbers
    protocols.append(Protocol(
        partial(
            visualise_data, data_writer=data_writer, 
            cols_x=['mutation_num'],
            cols_y=['eqconstants_num_interacting'],
            plot_type='bar_plot',
            out_name='num_interacting_barplot',
            threshold_value_max=binding_rates_threshold_upper,
            exclude_rows_zero_in_cols=['mutation_num'],
            use_sns=True,
            ci="sd",
            title=f'Minimum ' + r'$k_d$' + ' strength',
            ylabel='Number of interactions',
            xlabel='Number of mutations'),
        req_input=True,
        name='visualise_mutated_interactions'
    ))

    # Analytics visualisation
    analytics_types = ['fold_change',
                       'overshoot',
                       'precision',
                       'response_time',
                       'response_time_high',
                       'response_time_low',
                       'sensitivity',
                       'steady_states']  # Timeseries(data=None).get_analytics_types()
    # Log graphs with mutation number hue
    # Difference
    for filltype in ['dodge', 'fill']:
        for cols_x, title, xlabel in [
                [
                    f'{analytics_type}_diff_to_base_circuit',
                    f'{prettify_keys_for_label(analytics_type)} difference between circuit\nand mutated counterparts',
                    f'{prettify_keys_for_label(analytics_type)} difference'
                ] for analytics_type in analytics_types]:

            protocols.append(Protocol(
                partial(
                    visualise_data,
                    data_writer=data_writer, cols_x=[cols_x],
                    plot_type='histplot',
                    out_name=f'{cols_x}_log_{filltype}',
                    exclude_rows_zero_in_cols=['mutation_num'],
                    misc_histplot_kwargs={
                        "multiple": filltype,
                        "hue": 'mutation_num',
                        "element": "step"
                    },
                    log_axis=(True, False),
                    use_sns=True,
                    expand_coldata_using_col_x=True,
                    column_name_for_expanding_labels='sample_names',
                    idx_for_expanding_labels=0,
                    title=title,
                    xlabel=xlabel
                ),
                req_input=True,
                name='visualise_interactions_difference',
                skip=config_file.get('only_visualise_circuits', False)
            ))

    # Log graphs with mutation number hue
    # Ratios
    for filltype in ['dodge', 'fill']:
        for cols_x, title, xlabel in [
                [
                    f'{analytics_type}_ratio_from_mutation_to_base',
                    f'{prettify_keys_for_label(analytics_type)} ratio from mutated\nto original circuit',
                    f'{prettify_keys_for_label(analytics_type)} ratio'
                ] for analytics_type in analytics_types]:

            protocols.append(Protocol(
                partial(
                    visualise_data,
                    data_writer=data_writer, cols_x=[cols_x],
                    plot_type='histplot',
                    out_name=f'{cols_x}_log_{filltype}',
                    exclude_rows_zero_in_cols=['mutation_num'],
                    misc_histplot_kwargs={
                        "multiple": filltype,
                        "hue": 'mutation_num',
                        "element": "step"
                    },
                    log_axis=(True, False),
                    use_sns=True,
                    expand_coldata_using_col_x=True,
                    column_name_for_expanding_labels='sample_names',
                    idx_for_expanding_labels=0,
                    title=title,
                    xlabel=xlabel
                ),
                req_input=True,
                name='visualise_interactions_difference',
                skip=config_file.get('only_visualise_circuits', False)
            ))

    for col_x, col_y in [precision]

    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
