import operator
from functools import partial
import logging
import os

import pandas as pd

from fire import Fire
from src.srv.io.manage.script_manager import script_preamble
from src.utils.misc.scripts_io import get_search_dir, load_experiment_config_original
from src.utils.misc.string_handling import prettify_keys_for_label
from src.utils.results.analytics.naming import get_true_names_analytics, get_signal_dependent_analytics_all, DIFF_KEY, RATIO_KEY

from src.utils.results.experiments import Experiment, Protocol
from src.utils.results.result_writer import ResultWriter
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.io import get_pathnames_from_mult_dirs

from src.utils.results.visualisation import visualise_data
from src.utils.data.data_format_tools.common import load_json_as_dict, load_csv_mult


def main(config=None, data_writer=None):
    # Set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        "scripts", "analyse_mutated_templates_loaded", "configs", "base_config.json"))
    config_file = load_json_as_dict(config)

    config_file, source_dirs = get_search_dir(
        config_searchdir_key='source_dirs', config_file=config_file)
    if type(source_dirs) != list:
        source_dirs = [source_dirs]

    if config_file.get('only_visualise_circuits', False):
        exclude_rows_via_cols = ['mutation_name']
    else:
        exclude_rows_via_cols = []

    # source_config = load_experiment_config_original(
    #     source_dirs[0], target_purpose='mutation_effect_on_interactions_signal')
    logging.warning('The source configuration is being chosen as that from the first source directory supplied: ' +
                    f'{source_dirs[0]} \nIf multiple directories were used to generate mutations, modify the script.')
    # num_mutations = source_config['mutations']['mutation_nums_within_sequence']
    # num_mutations = [num_mutations] if type(
    #     num_mutations) != list else num_mutations
    # plot_grammar = 's' if type(
    #     num_mutations) == list or num_mutations > 1 else ''
    # plot_grammar = 's' if num_mutations > 1 else ''

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
                file_key='tabulated_mutation_info.csv',
                first_only=True),
            req_output=True,
            name='get_pathnames_from_mult_dirs'
        ),
        Protocol(
            partial(
                load_csv_mult
                # as_type=pd.DataFrame
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

    # Binding rates interactions og min and max
    def interaction_vis(data: pd.DataFrame, data_writer):
        num_mutations = list(data['mutation_num'].unique()) + ['all']
        log_opts = [(False, False), (True, False)]
        for log_opt in log_opts:
            log_text = '_log' if any(log_opt) else ''
            cols_xs = ['eqconstants_max_interaction',
                       'eqconstants_min_interaction']
            out_names = ['eqconstants_max_freqs',
                         'eqconstants_min_freqs']
            titles = ['Maximum equilibrium constant, unmutated circuits',
                      'Minimum equilibrium constant, unmutated circuits']
            for cols_x, out_name, title in zip(cols_xs, out_names, titles):
                visualise_data(
                    og_data=data, data_writer=data_writer, cols_x=[cols_x],
                    plot_type='histplot',
                    out_name=out_name + log_text,
                    exclude_rows_nonempty_in_cols=[
                        'mutation_name'],
                    selection_conditions=[
                        ('mutation_num', operator.ne, 0)],
                    log_axis=log_opt,
                    use_sns=True,
                    title=title,
                    xlabel='Equilibrium constant (unitless)')

            for m in num_mutations:
                # Eqconstants max, min mutations
                if m == 'all':
                    plot_grammar_m = 's'
                    hue = 'mutation_num'
                    selection_conditions = None
                else:
                    plot_grammar_m = 's' if m > 1 else ''
                    hue = None
                    selection_conditions = [('mutation_num', operator.eq, m)]
                all_cols_x = ['eqconstants_max_interaction',
                              'eqconstants_max_interaction_diff_to_base_circuit',
                              'eqconstants_min_interaction',
                              'eqconstants_min_interaction_diff_to_base_circuit']
                titles = [f'Maximum equilibrium constant, {m} mutation{plot_grammar_m}',
                          f'Difference between circuit\nand mutated (maximum equilibrium constant), {m} mutation{plot_grammar_m}',
                          f'Minimum equilibrium constant, {m} mutation{plot_grammar_m}',
                          f'Difference between circuit\nand mutated (minimum equilibrium constant), {m} mutation{plot_grammar_m}']
                out_names = [f'eqconstants_max_freqs_mutations_m{m}',
                             f'eqconstants_max_freqs_diffs_m{m}',
                             f'eqconstants_min_freqs_mutations_m{m}',
                             f'eqconstants_min_freqs_diffs_m{m}']
                for cols_x, title, out_name in zip(all_cols_x, titles, out_names):
                    visualise_data(
                        og_data=data, data_writer=data_writer, cols_x=[cols_x],
                        plot_type='histplot',
                        out_name=out_name + log_text,
                        exclude_rows_nonempty_in_cols=exclude_rows_via_cols,
                        selection_conditions=selection_conditions,
                        # threshold_value_max=binding_rates_threshold_upper,
                        log_axis=log_opt,
                        use_sns=True,
                        hue=hue,
                        title=title,
                        xlabel='Equilibrium constant (unitless)')

        # Interaction histplots
        # The choice for these is in the csv generated by the summarise_simulation script
        interaction_types_chosen = ['binding_rates_dissociation_max_interaction',
                                    'binding_rates_dissociation_min_interaction',
                                    'eqconstants_max_interaction',
                                    'eqconstants_min_interaction']
        visualisation_types = [
            '', '_ratio_from_mutation_to_base', '_diff_to_base_circuit']
        # With mutated circuits
        for log_opt in [(False, False)]:
            log_text = '_log' if any(log_opt) else ''
            for m in num_mutations:

                if m == 'all':
                    plot_grammar_m = 's'
                    hue = 'mutation_num'
                    selection_conditions = None
                else:
                    plot_grammar_m = 's' if m > 1 else ''
                    hue = None
                    selection_conditions = [('mutation_num', operator.eq, m)]
                titles = [
                    # Binding rates dissociation
                    f'Maximum ' + r'$k_d$' + f', {m} mutation{plot_grammar_m}',
                    f'Ratio between mutated and \noriginal circuit (maximum ' +
                    r'$k_d$' + f'), {m} mutation{plot_grammar_m}',
                    f'Difference between circuit\nand mutated (maximum ' +
                    r'$k_d$' + f'), {m} mutation{plot_grammar_m}',
                    f'Minimum ' + r'$k_d$' + f', {m} mutation{plot_grammar_m}',
                    f'Ratio between mutated and \noriginal circuit (minimum ' +
                    r'$k_d$' + f'), {m} mutation{plot_grammar_m}',
                    f'Difference between circuit\nand mutated (minimum ' +
                    r'$k_d$' + f'), {m} mutation{plot_grammar_m}',
                    # eqconstant
                    f'Maximum equilibrium constant, {m} mutation{plot_grammar_m}',
                    f'Ratio between mutated and original circuit\nequilibrium constant (max), {m} mutation{plot_grammar_m}',
                    f'Difference between reference and mutated\ncircuits, equilibrium constant (max), {m} mutation{plot_grammar_m}',
                    f'Minimum equilibrium constant, {m} mutation{plot_grammar_m}',
                    f'Ratio between mutated and original circuit\nequilibrium constant (min), {m} mutation{plot_grammar_m}',
                    f'Difference between reference and mutated\ncircuits, equilibrium constant (min), {m} mutation{plot_grammar_m}'
                ]
                title_count = 0
                for interaction_type in interaction_types_chosen:
                    for visualisation_type in visualisation_types:
                        if 'eqconstants' in interaction_type:
                            xlabel = 'Equilibrium constant'
                        else:
                            xlabel = 'Dissociation rate ' + r'$k_d$' + ' (' \
                                f'{rate_unit})' \
                                f'{binding_rates_threshold_upper_text}'
                        if 'Difference' in titles[title_count]:
                            xlabel = r'$\Delta$ ' + xlabel

                        cols_x = f'{interaction_type}{visualisation_type}'
                        if cols_x == 'RMSE_ratio_from_mutation_to_base':
                            continue
                        out_name = f'{cols_x}{log_text}_m{m}'
                        visualise_data(
                            og_data=data, data_writer=data_writer, cols_x=[
                                cols_x],
                            plot_type='histplot',
                            out_name=out_name,
                            exclude_rows_nonempty_in_cols=exclude_rows_via_cols,
                            # threshold_value_max=binding_rates_threshold_upper,
                            selection_conditions=selection_conditions,
                            log_axis=log_opt,
                            use_sns=True,
                            hue=hue,
                            title=titles[title_count],
                            xlabel=xlabel)
                        title_count += 1

        # Analytics histplots
        analytics_types = get_true_names_analytics(data)
        for log_opt in [(False, False)]:
            log_text = '_log' if any(log_opt) else ''
            for m in num_mutations:
                if m == 'all':
                    plot_grammar_m = 's'
                    hue = 'mutation_num'
                    selection_conditions = None
                else:
                    plot_grammar_m = 's' if m > 1 else ''
                    hue = None
                    selection_conditions = [('mutation_num', operator.eq, m)]
                # for v, vt in zip(visualisation_types, visualisation_type_titles):
                for analytics_type, cols_x, title, xlabel in [
                        [
                            analytics_type,
                            f'{analytics_type}',
                            f'{prettify_keys_for_label(analytics_type)}: circuit\nand mutated counterparts, {m} mutation{plot_grammar_m}',
                            f'{prettify_keys_for_label(analytics_type)}'
                        ] for analytics_type in analytics_types]:

                    visualise_data(
                        og_data=data,
                        data_writer=data_writer, cols_x=[cols_x],
                        plot_type='histplot',
                        out_name=f'{cols_x}{log_text}_m{m}',
                        exclude_rows_nonempty_in_cols=exclude_rows_via_cols,
                        selection_conditions=selection_conditions,
                        log_axis=log_opt,
                        use_sns=True,
                        hue=hue,
                        title=title,
                        xlabel=xlabel)

    protocols.append(
        Protocol(
            partial(
                interaction_vis,
                data_writer=data_writer
            ),
            req_input=True,
            req_output=True,
            name='visualise_interactions_analytics'
        )
    )

    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
