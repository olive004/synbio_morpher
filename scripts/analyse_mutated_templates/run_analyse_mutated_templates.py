from functools import partial
import logging
import os

import numpy as np

from fire import Fire
from src.srv.io.manage.script_manager import script_preamble
from src.utils.misc.scripts_io import load_experiment_config
from src.utils.misc.string_handling import prettify_keys_for_label
from src.utils.results.analytics.naming import get_analytics_types

from src.utils.results.experiments import Experiment, Protocol
from src.utils.results.result_writer import ResultWriter
from src.utils.results.visualisation import visualise_data
from src.srv.parameter_prediction.simulator import SIMULATOR_UNITS, RawSimulationHandling
from src.srv.sequence_exploration.sequence_analysis import tabulate_mutation_info
from src.utils.data.data_format_tools.common import load_json_as_dict


def main(config=None, data_writer=None):
    # Set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        # "scripts", "analyse_mutated_templates", "configs", "logscale", "analyse_templates.json"))
        # "scripts", "analyse_mutated_templates", "configs", "analyse_mutated_templates_1.json"))
        # "scripts", "analyse_mutated_templates", "configs", "analyse_mutated_templates_2.json"))
        # "scripts", "analyse_mutated_templates", "configs", "analyse_mutated_templates_10.json"))
        # "scripts", "analyse_mutated_templates", "configs", "analyse_mutated_templates_20.json"))
        # "scripts", "analyse_mutated_templates", "configs", "analyse_mutated_templates_1_highmag.json"))
        # "scripts", "analyse_mutated_templates", "configs", "analyse_mutated_templates_2_highmag.json"))
        # "scripts", "analyse_mutated_templates", "configs", "analyse_mutated_templates_10_highmag.json"))
        "scripts", "analyse_mutated_templates", "configs", "analyse_mutated_templates_20_highmag.json"))
    # "scripts", "analyse_mutated_templates", "configs", "base_config_testing.json"))
    # "scripts", "analyse_mutated_templates", "configs", "base_config.json"))
    config_file = load_json_as_dict(config)

    # Start_experiment
    if data_writer is None:
        data_writer = ResultWriter(
            purpose=config_file.get('experiment', {}).get('purpose'))

    source_dir = config_file.get('source_dir')
    source_config = load_experiment_config(source_dir)

    if config_file.get('only_visualise_circuits', False):
        exclude_rows_via_cols = ['mutation_name']
    else:
        exclude_rows_via_cols = []

    num_mutations = source_config['mutations']['mutation_nums_within_sequence']
    plot_grammar = 's' if type(num_mutations) == list or num_mutations > 1 else ''
    logging.warning('Not implemented: number of mutations varying in config')

    binding_rates_threshold_upper = np.power(10, 6)
    binding_rates_threshold_upper_text = f', with cutoff at {binding_rates_threshold_upper}' if binding_rates_threshold_upper else ''

    rate_unit = SIMULATOR_UNITS[source_config["interaction_simulator"]
                                ["name"]]["rate"]

    protocols = [
        Protocol(
            partial(tabulate_mutation_info, source_dir=source_dir,
                    data_writer=data_writer),
            req_output=True,
            name='tabulate_mutation_info'
        ),
        # Binding rates max int's og
        Protocol(
            partial(visualise_data, data_writer=data_writer, cols_x=['binding_rates_max_interaction'],
                    plot_type='histplot',
                    out_name='binding_rates_max_freqs',
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
        ), ]

    # Interaction histplots og
    protocols.append(
        Protocol(
            partial(visualise_data, data_writer=data_writer, cols_x=['binding_rates_min_interaction'],
                    plot_type='histplot',
                    out_name='binding_rates_min_interaction_og',
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
                    exclude_rows_nonempty_in_cols=['mutation_name'],
                    threshold_value_max=binding_rates_threshold_upper,
                    log_axis=(False, False),
                    use_sns=True,
                    title=f'Maximum equilibrium constant, {num_mutations} mutation{plot_grammar}',
                    xlabel='Equilibrium constant',
            req_input=True,
            name='visualise_mutated_interactions'
        ))
    )

    # Interaction histplots
    interaction_types = ['binding_rates_min_interaction',
                         'eqconstants_max_interaction']
    visualisation_types = [
        '', '_ratio_from_mutation_to_base', '_diff_to_base_circuit']
    titles = [
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
    # With mutated circuits
    for log_opt in [(False, False), (True, False)]:
        title_count = 0
        for interaction_type in interaction_types:
            for visualisation_type in visualisation_types:

                cols_x = f'{interaction_type}{visualisation_type}'
                out_name = f'{cols_x}_log' if any(log_opt) else f'{cols_x}'
                protocols.append(
                    Protocol(
                        partial(visualise_data, data_writer=data_writer, cols_x=[cols_x],
                                plot_type='histplot',
                                out_name=out_name,
                                exclude_rows_nonempty_in_cols=exclude_rows_via_cols,
                                threshold_value_max=binding_rates_threshold_upper,
                                log_axis=log_opt,
                                use_sns=True,
                                title=titles[title_count],
                                xlabel='Dissociation rate ' + r'$k_d$' + ' (' +
                                f'{rate_unit})' +
                                f'{binding_rates_threshold_upper_text}'),
                        req_input=True,
                        name='visualise_mutated_interactions'
                    )
                )
                title_count += 1
        # Without mutated circuits
        protocols.append(
            Protocol(
                partial(visualise_data, data_writer=data_writer, cols_x=[cols_x],
                        plot_type='histplot',
                        out_name=out_name,
                        exclude_rows_nonempty_in_cols=exclude_rows_via_cols,
                        threshold_value_max=binding_rates_threshold_upper,
                        log_axis=log_opt,
                        use_sns=True,
                        title=titles[title_count],
                        xlabel='Dissociation rate ' + r'$k_d$' + ' (' +
                        f'{rate_unit})' +
                        f'{binding_rates_threshold_upper_text}'),
                req_input=True,
                name='visualise_mutated_interactions'
            )
        )

    # Analytics histplots
    analytics_types = get_analytics_types()
    for log_opt in [(False, False), (True, False)]:
        for analytics_type, cols_x, title, xlabel in [
                [
                    analytics_type,
                    f'{analytics_type}_diff_to_base_circuit',
                    f'{prettify_keys_for_label(analytics_type)} difference between circuit\nand mutated counterparts, {num_mutations} mutation{plot_grammar}',
                    f'{prettify_keys_for_label(analytics_type)} difference'
                ] for analytics_type in analytics_types]:

            log_text = '_log' if any(log_opt) else ''
            protocols.append(
                Protocol(partial(
                    visualise_data,
                    data_writer=data_writer, cols_x=[cols_x],
                    plot_type='histplot',
                    out_name=f'{analytics_type}{log_text}',
                    exclude_rows_nonempty_in_cols=exclude_rows_via_cols,
                    log_axis=log_opt,
                    use_sns=True,
                    title=title,
                    xlabel=xlabel,
                    req_input=True,
                    name='visualise_interactions_difference',
                    skip=config_file.get('only_visualise_circuits', False)
                ))
            )

    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
