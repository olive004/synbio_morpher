from functools import partial
import logging
import os

import numpy as np
import pandas as pd

from fire import Fire
from src.srv.io.manage.script_manager import script_preamble
from src.utils.misc.scripts_io import get_search_dir
from src.utils.misc.string_handling import prettify_keys_for_label
from src.utils.results.analytics.timeseries import Timeseries

from src.utils.results.experiments import Experiment, Protocol
from src.utils.results.result_writer import ResultWriter
from src.srv.parameter_prediction.simulator import RawSimulationHandling
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.io import get_pathnames_from_mult_dirs

from src.utils.results.visualisation import visualise_data
from src.utils.data.data_format_tools.common import load_json_as_dict, load_json_mult


def main(config=None, data_writer=None):
    # Set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        "scripts", "analyse_mutated_templates_loaded", "configs", "base_config_test_2.json"))
    config_file = load_json_as_dict(config)

    # Start_experiment
    if data_writer is None:
        data_writer = ResultWriter(
            purpose=config_file.get('experiment', {}).get('purpose'))

    config_file, source_dirs = get_search_dir(
        config_searchdir_key='source_dirs', config_file=config_file)
    if type(source_dirs) != list:
        source_dirs = [source_dirs]
    # source_dir = source_dirs[0]
    # source_config = load_experiment_config_original(
    #     source_dir, 'mutation_effect_on_interactions_signal')

    if config_file.get('preprocessing_func') == 'rate_to_energy':
        preprocessing_func = RawSimulationHandling().rate_to_energy,
    else:
        preprocessing_func = None

    if config_file.get('only_visualise_circuits', False):
        exclude_rows_via_cols = ['mutation_name']
    else:
        exclude_rows_via_cols = []

    num_mutations = 1
    plot_grammar = 's' if num_mutations > 1 else ''

    # binding_rates_threshold_upper = np.power(10, 6)
    binding_rates_threshold_upper = None
    binding_rates_threshold_upper_text = f', with cutoff at {binding_rates_threshold_upper}' if binding_rates_threshold_upper else ''

    # rate_unit = fr'${SIMULATOR_UNITS[source_config["interaction_simulator"]["name"]]["rate"]}$'
    rate_unit = r'$s^{-1}$'

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
        # Binding rates max int's og
        Protocol(
            partial(visualise_data, data_writer=data_writer, cols_x=['binding_rates_max_interaction'],
                    plot_type='histplot',
                    out_name='binding_rates_max_freqs',
                    preprocessor_func_x=preprocessing_func,
                    exclude_rows_nonempty_in_cols=['mutation_name'],
                    threshold_value_max=binding_rates_threshold_upper,
                    log_axis=(False, False),
                    use_sns=True,
                    title='Maximum ' + r'$k_d$' + ' strength, unmutated circuits',
                    xlabel='Dissociation rate ' + r'$k_d$' + ' (' +
                    f'{rate_unit})' +
                    f'{binding_rates_threshold_upper_text}'),
            req_input=True,
            name='visualise_circuit_interactions'
        ),
        # Binding rates max int's mutations
        Protocol(
            partial(visualise_data, data_writer=data_writer, cols_x=['binding_rates_max_interaction'],
                    plot_type='histplot',
                    out_name='binding_rates_max_freqs_mutations',
                    preprocessor_func_x=preprocessing_func,
                    exclude_rows_nonempty_in_cols=exclude_rows_via_cols,
                    threshold_value_max=binding_rates_threshold_upper,
                    log_axis=(False, False),
                    use_sns=True,
                    title=f'Maximum ' + r'$k_d$' + \
                    f' strength, {num_mutations} mutation{plot_grammar}',
                    xlabel='Dissociation rate' + r'$k_d$' + '(' +
                    f'{rate_unit})' +
                    f'{binding_rates_threshold_upper_text}'),
            req_input=True,
            name='visualise_mutated_interactions'
        ),
        # Binding rates max int's diff
        Protocol(
            partial(visualise_data, data_writer=data_writer, cols_x=['binding_rates_max_interaction_diff_to_base_circuit'],
                    plot_type='histplot',
                    out_name='binding_rates_max_freqs_diffs',
                    preprocessor_func_x=preprocessing_func,
                    exclude_rows_nonempty_in_cols=exclude_rows_via_cols,
                    log_axis=(False, False),
                    use_sns=True,
                    title=f'Difference between circuit\nand mutated (maximum ' + \
                    r'$k_d$' + f'), {num_mutations} mutation{plot_grammar}',
                    xlabel='Difference in ' + r'$k_d$' + ' (' +
                    f'{rate_unit})' +
                    f'{binding_rates_threshold_upper_text}'),
            req_input=True,
            name='visualise_interactions_difference',
            skip=config_file.get('only_visualise_circuits', False)
        ),

        # Binding rates min int's og
        Protocol(
            partial(visualise_data, data_writer=data_writer, cols_x=['binding_rates_min_interaction'],
                    plot_type='histplot',
                    out_name='binding_rates_min_freqs',
                    preprocessor_func_x=preprocessing_func,
                    exclude_rows_nonempty_in_cols=['mutation_name'],
                    threshold_value_max=binding_rates_threshold_upper,
                    log_axis=(False, False),
                    use_sns=True,
                    title='Minimum ' + r'$k_d$' + ' strength, unmutated circuits',
                    xlabel='Dissociation rate ' + r'$k_d$' + ' (' +
                    f'{rate_unit})' +
                    f'{binding_rates_threshold_upper_text}'),
            req_input=True,
            name='visualise_circuit_interactions'
        )]

    # Interaction histplots og
    protocols.append(
        Protocol(
            partial(visualise_data, data_writer=data_writer, cols_x=['binding_rates_min_interaction'],
                    plot_type='histplot',
                    out_name='binding_rates_min_interaction_og',
                    preprocessor_func_x=preprocessing_func,
                    exclude_rows_nonempty_in_cols=['mutation_name'],
                    threshold_value_max=binding_rates_threshold_upper,
                    log_axis=(False, False),
                    use_sns=True,
                    title='Maximum ' + r'$k_d$' + ' strength, unmutated circuits',
                    xlabel='Dissociation rate ' + r'$k_d$' + ' (' +
                    f'{rate_unit})' +
                    f'{binding_rates_threshold_upper_text}'),
            req_input=True,
            name='visualise_mutated_interactions'
        )
    )
    protocols.append(
        Protocol(
            partial(visualise_data, data_writer=data_writer, cols_x=['eqconstants_max_interaction'],
                    plot_type='histplot',
                    out_name='eqconstants_max_interaction_og',
                    preprocessor_func_x=preprocessing_func,
                    exclude_rows_nonempty_in_cols=['mutation_name'],
                    threshold_value_max=binding_rates_threshold_upper,
                    log_axis=(False, False),
                    use_sns=True,
                    title=f'Maximum equilibrium constant, {num_mutations} mutation{plot_grammar}',
                    xlabel='Equilibrium constant'),
            req_input=True,
            name='visualise_mutated_interactions'
        )
    )

    # Interaction histplots
    interaction_types = ['binding_rates_max_interaction',
                         'binding_rates_min_interaction',
                         'eqconstants_max_interaction']
    visualisation_types = [
        '', '_ratio_from_mutation_to_base', '_diff_to_base_circuit']
    # With mutated circuits
    for log_opt in [(False, False)]:
        log_text = '_log' if log_opt[0] or log_opt[1] else ''

        titles = [
            f'Maximum ' + r'$k_d$' +
            f' strength, {num_mutations} mutation{plot_grammar}',
            f'Difference between circuit\nand mutated (maximum ' +
            r'$k_d$' + f'), {num_mutations} mutation{plot_grammar}',
            f'Ratio between mutated and \noriginal circuit (maximum ' +
            r'$k_d$' + f'), {num_mutations} mutation{plot_grammar}',
            f'Minimum ' + r'$k_d$' +
            f' strength, {num_mutations} mutation{plot_grammar}',
            f'Difference between circuit\nand mutated (minimum ' +
            r'$k_d$' + f'), {num_mutations} mutation{plot_grammar}',
            f'Ratio between mutated and \noriginal circuit (minimum ' +
            r'$k_d$' + f'), {num_mutations} mutation{plot_grammar}',
            f'Maximum equilibrium constant, {num_mutations} mutation{plot_grammar}',
            f'Difference between circuit\nand mutated equilibrium constant, {num_mutations} mutation{plot_grammar}',
            f'Ratio between mutated and original circuit\nequilibrium constant, {num_mutations} mutation{plot_grammar}'
        ]
        title_count = 0
        for interaction_type in interaction_types:
            for visualisation_type in visualisation_types:
                if interaction_type == 'eqconstants_max_interaction':
                    xlabel = 'Equilibrium constant'
                else:
                    xlabel = 'Dissociation rate ' + r'$k_d$' + ' (' \
                        f'{rate_unit})' \
                        f'{binding_rates_threshold_upper_text}'

                cols_x = f'{interaction_type}{visualisation_type}'
                out_name = f'{cols_x}{log_text}'
                protocols.append(
                    Protocol(
                        partial(visualise_data, data_writer=data_writer, cols_x=[cols_x],
                                plot_type='histplot',
                                out_name=out_name,
                                preprocessor_func_x=preprocessing_func,
                                exclude_rows_nonempty_in_cols=exclude_rows_via_cols,
                                threshold_value_max=binding_rates_threshold_upper,
                                log_axis=log_opt,
                                use_sns=True,
                                title=titles[title_count],
                                xlabel=xlabel),
                        req_input=True,
                        name='visualise_mutated_interactions'
                    )
                )
                title_count += 1

        # Without mutated circuits
        titles = [
            'Maximum ' + r'$k_d$' + ' strength, unmutated circuits',
            'Minimum ' + r'$k_d$' + ' strength, unmutated circuits',
            'Maximum equilibrium constant, unmutated circuits',
        ]
        for cols_x, title in zip(interaction_types, titles):
            if interaction_type == 'eqconstants_max_interaction':
                xlabel = 'Equilibrium constant'
            else:
                xlabel = 'Dissociation rate ' + r'$k_d$' + ' (' \
                    f'{rate_unit})' \
                    f'{binding_rates_threshold_upper_text}'

            protocols.append(
                Protocol(
                    partial(visualise_data, data_writer=data_writer, cols_x=[cols_x],
                            plot_type='histplot',
                            out_name=f'{cols_x}_unmutated',
                            preprocessor_func_x=preprocessing_func,
                            exclude_rows_nonempty_in_cols=['mutation_name'],
                            threshold_value_max=binding_rates_threshold_upper,
                            log_axis=log_opt,
                            use_sns=True,
                            title=title,
                            xlabel=xlabel),
                    req_input=True,
                    name='visualise_mutated_interactions'
                )
            )

    # Analytics histplots
    analytics_types = Timeseries(data=None).get_analytics_types()
    for log_opt in [(False, False)]:
        for analytics_type, cols_x, title, xlabel in [
                [
                    analytics_type,
                    f'{analytics_type}_diff_to_base_circuit',
                    f'{prettify_keys_for_label(analytics_type)} difference between circuit\nand mutated counterparts, {num_mutations} mutation{plot_grammar}',
                    f'{prettify_keys_for_label(analytics_type)} difference'
                ] for analytics_type in analytics_types]:

            log_text = '_log' if log_opt[0] or log_opt[1] else ''
            protocols.append(
                Protocol(
                    partial(
                        visualise_data,
                        data_writer=data_writer, cols_x=[cols_x],
                        plot_type='histplot',
                        out_name=f'{analytics_type}{log_text}',
                        preprocessor_func_x=preprocessing_func,
                        exclude_rows_nonempty_in_cols=exclude_rows_via_cols,
                        log_axis=log_opt,
                        use_sns=True,
                        title=title,
                        xlabel=xlabel),
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
