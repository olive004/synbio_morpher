from functools import partial
import logging
import operator
import os

import numpy as np
import pandas as pd

from fire import Fire
from src.srv.io.manage.script_manager import script_preamble
from src.utils.misc.io import get_pathnames_from_mult_dirs
from src.utils.misc.numerical import cast_astype
from src.utils.misc.scripts_io import load_experiment_config_original
from src.utils.misc.string_handling import prettify_keys_for_label
from src.utils.results.analytics.timeseries import Timeseries

from src.utils.results.experiments import Experiment, Protocol
from src.utils.results.result_writer import ResultWriter
from src.utils.results.visualisation import visualise_data
from src.utils.data.data_format_tools.common import load_json_as_dict, load_json_mult


def main(config=None, data_writer=None):
    # Set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
            # "scripts", "analyse_mutated_templates_loaded", "configs", "base_config_test_2.json"))
            "scripts", "analyse_mutated_templates_loaded", "configs", "analyse_large.json"))
            # "scripts", "analyse_mutated_templates_loaded", "configs", "analyse_large_highmag.json"))
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
    analytics_types = Timeseries(data=None).get_analytics_types()

    # Bar plots

    def plot_mutation_attr(mutation_attr: str, remove_outliers: bool, log_axis: tuple, plot_type):
        outlier_std_threshold = 3
        outlier_text = f', outliers removed (>{outlier_std_threshold} standard deviations)' if remove_outliers else ''

        log_text = '_log' if any(log_axis) else ''
        outlier_save_text = '_nooutliers' if remove_outliers else ''
        # Difference
        for cols_x, cols_y, title, xlabel, ylabel in [
                [
                    mutation_attr,
                    f'{analytics_type}_diff_to_base_circuit',
                    f'{prettify_keys_for_label(mutation_attr)} vs. {prettify_keys_for_label(analytics_type)} difference between circuit\nand mutated counterparts',
                    f'{prettify_keys_for_label(mutation_attr)}',
                    f'{prettify_keys_for_label(analytics_type)} difference{outlier_text}'
                ] for analytics_type in analytics_types]:
            protocols.append(Protocol(
                partial(
                    visualise_data,
                    data_writer=data_writer,
                    cols_x=[cols_x], cols_y=[cols_y],
                    plot_type=plot_type,
                    out_name=f'{cols_x}_{cols_y}{log_text}{outlier_save_text}',
                    exclude_rows_zero_in_cols=['mutation_num'],
                    postprocessor_func_x=partial(cast_astype, 
                        dtypes=[int, float]),
                    selection_conditions=[(
                        mutation_attr, operator.ne, ''
                    )],
                    log_axis=log_axis,
                    use_sns=True,
                    hue='mutation_num',
                    expand_xcoldata_using_col=True,
                    # expand_ycoldata_using_col=True,
                    column_name_for_expanding_xcoldata=None,
                    # column_name_for_expanding_ycoldata='sample_names',
                    idx_for_expanding_xcoldata=1,
                    # idx_for_expanding_ycoldata=0,
                    remove_outliers_y=remove_outliers,
                    outlier_std_threshold_y=outlier_std_threshold,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel
                ),
                req_input=True,
                name='visualise_interactions_difference',
                skip=config_file.get('only_visualise_circuits', False)
            )
            )

        # Ratios
        for cols_x, cols_y, title, xlabel, ylabel in [
                [
                    mutation_attr,
                    f'{analytics_type}_ratio_from_mutation_to_base',
                    f'{prettify_keys_for_label(mutation_attr)} vs. {prettify_keys_for_label(analytics_type)} ratio from mutated\nto original circuit',
                    f'{prettify_keys_for_label(mutation_attr)}',
                    f'{prettify_keys_for_label(analytics_type)} ratio{outlier_text}'
                ] for analytics_type in analytics_types]:
            protocols.append(Protocol(
                partial(
                    visualise_data,
                    data_writer=data_writer,
                    cols_x=[cols_x], cols_y=[cols_y],
                    plot_type=plot_type,
                    out_name=f'{cols_x}_{cols_y}{log_text}{outlier_save_text}',
                    exclude_rows_zero_in_cols=['mutation_num'],
                    postprocessor_func_x=partial(cast_astype, 
                        dtypes=[int, float]),
                    selection_conditions=[(
                        mutation_attr, operator.ne, ''
                    )],
                    log_axis=log_axis,
                    use_sns=True,
                    hue='mutation_num',
                    expand_xcoldata_using_col=True,
                    # expand_ycoldata_using_col=True,
                    column_name_for_expanding_xcoldata=None,
                    # column_name_for_expanding_ycoldata='sample_names',
                    idx_for_expanding_xcoldata=1,
                    # idx_for_expanding_ycoldata=0,
                    remove_outliers_y=remove_outliers,
                    outlier_std_threshold_y=outlier_std_threshold,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel
                ),
                req_input=True,
                name='visualise_interactions_difference',
                skip=config_file.get('only_visualise_circuits', False)
            )
            )

    for remove_outliers in [False, True]:
        for log_axis in [(False, False), (False, True)]:
            plot_mutation_attr('mutation_type', remove_outliers=remove_outliers, log_axis=log_axis,
                               plot_type='bar_plot')
            plot_mutation_attr('mutation_positions', remove_outliers=remove_outliers, log_axis=log_axis,
                               plot_type='line_plot')

    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)