from functools import partial
import logging
import os

import pandas as pd

from fire import Fire
from src.utils.misc.io import get_pathnames_from_mult_dirs
from src.utils.misc.scripts_io import load_experiment_config_original
from src.utils.misc.string_handling import prettify_keys_for_label

from src.utils.results.experiments import Experiment, Protocol
from src.utils.results.result_writer import ResultWriter
from src.utils.results.visualisation import visualise_data
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
        )
    ]

    # Visualisations

    # Analytics visualisation
    analytics_types = ['fold_change',
                       'overshoot',
                       'precision',
                       'response_time',
                       'response_time_high',
                       'response_time_low',
                       'sensitivity',
                       'steady_states']  # Timeseries(data=None).get_analytics_types()

    # Scatter plots
    # Difference
    for cols_x, cols_y, title, xlabel, ylabel in [
            [
                'mutation_positions',
                f'{analytics_type}_diff_to_base_circuit',
                f'Position vs. {prettify_keys_for_label(analytics_type)} difference between circuit\nand mutated counterparts',
                f'{prettify_keys_for_label("mutation_position")}',
                f'{prettify_keys_for_label(analytics_type)} difference'
            ] for analytics_type in analytics_types]:
        protocols.append(Protocol(
            partial(
                visualise_data,
                data_writer=data_writer,
                cols_x=[cols_x], cols_y=[cols_y],
                plot_type='scatter_plot',
                out_name=f'{cols_x}_{cols_y}',
                exclude_rows_zero_in_cols=['mutation_num'],
                log_axis=(False, False),
                use_sns=True,
                hue='mutation_num',
                expand_xcoldata_using_col=True,
                expand_ycoldata_using_col=True,
                column_name_for_expanding_xcoldata=None,
                column_name_for_expanding_ycoldata='sample_names',
                idx_for_expanding_xcoldata=1,
                idx_for_expanding_ycoldata=0,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel
            ),
            req_input=True,
            name='visualise_interactions_difference',
            skip=config_file.get('only_visualise_circuits', False)
        )
        )

    # Ratio
    for cols_x, cols_y, title, xlabel, ylabel in [
            [
                'mutation_positions',
                f'{analytics_type}_ratio_from_mutation_to_base',
                f'Position vs. {prettify_keys_for_label(analytics_type)} ratio from mutated\nto original circuit',
                f'{prettify_keys_for_label("mutation_position")}',
                f'{prettify_keys_for_label(analytics_type)} ratio'
            ] for analytics_type in analytics_types]:
        protocols.append(Protocol(
            partial(
                visualise_data,
                data_writer=data_writer,
                cols_x=[cols_x], cols_y=[cols_y],
                plot_type='scatter_plot',
                out_name=f'{cols_x}_{cols_y}',
                exclude_rows_zero_in_cols=['mutation_num'],
                log_axis=(False, False),
                use_sns=True,
                hue='mutation_num',
                expand_xcoldata_using_col=True,
                expand_ycoldata_using_col=True,
                # column_name_for_expanding_xcoldata='mutation_num',
                column_name_for_expanding_xcoldata=None,
                column_name_for_expanding_ycoldata='sample_names',
                idx_for_expanding_xcoldata=1,
                idx_for_expanding_ycoldata=0,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel
            ),
            req_input=True,
            name='visualise_interactions_difference',
            skip=config_file.get('only_visualise_circuits', False)
        )
        )

    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
