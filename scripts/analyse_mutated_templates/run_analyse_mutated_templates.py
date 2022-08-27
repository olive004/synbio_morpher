from functools import partial
import logging
import os

import numpy as np

from fire import Fire
from src.utils.misc.scripts_io import load_experiment_config, load_experiment_config_original

from src.utils.results.experiments import Experiment, Protocol
from src.utils.results.result_writer import ResultWriter
from src.utils.results.visualisation import visualise_data
from src.srv.parameter_prediction.simulator import SIMULATOR_UNITS, RawSimulationHandling
from src.srv.sequence_exploration.sequence_analysis import tabulate_mutation_info
from src.utils.data.data_format_tools.common import load_json_as_dict


def main(config=None, data_writer=None):
    # Set configs
    if config is None:
        config = os.path.join(
            # "scripts", "analyse_mutated_templates", "configs", "logscale", "analyse_templates.json")
            # "scripts", "analyse_mutated_templates", "configs", "logscale", "analyse_mutated_templates_1.json")
            # "scripts", "analyse_mutated_templates", "configs", "logscale", "analyse_mutated_templates_2.json")
            "scripts", "analyse_mutated_templates", "configs", "base_config.json")
    config_file = load_json_as_dict(config)

    # Start_experiment
    if data_writer is None:
        data_writer = ResultWriter(purpose='analyse_mutated_templates')

    source_dir = config_file.get('source_dir')
    source_config = load_experiment_config(source_dir)
    if config_file.get('preprocessing_func') == 'rate_to_energy':
        preprocessing_func = RawSimulationHandling().rate_to_energy,
    else:
        preprocessing_func = None

    if config_file.get('only_visualise_circuits', False):
        exclude_rows_via_cols = ['mutation_name']
    else:
        exclude_rows_via_cols = []

    num_mutations = source_config['mutations']['mutation_nums_within_sequence']
    plot_grammar = 's' if num_mutations > 1 else ''

    binding_rates_threshold_upper = np.power(10,6)
    binding_rates_threshold_upper_text = f', with cutoff at {binding_rates_threshold_upper}' if binding_rates_threshold_upper else ''
    protocols = [
        Protocol(
            partial(tabulate_mutation_info, source_dir=source_dir,
                    data_writer=data_writer),
            req_output=True,
            name='tabulate_mutation_info'
        ),
        # Binding rates max int's og
        Protocol(
            partial(visualise_data, data_writer=data_writer, cols=['binding_rates_max_interaction'],
                    plot_type='histplot',
                    out_name='binding_rates_max_freqs',
                    preprocessor_func=preprocessing_func,
                    exclude_rows_nonempty_in_cols=['mutation_name'],
                    threshold_value_max=binding_rates_threshold_upper,
                    log_axis=config_file.get('log_scale', (False, False)),
                    use_sns=True,
                    title='Maximum k_d strength, unmutated circuits',
                    xlabel='Dissociation rate k_d (' +
                    f'{SIMULATOR_UNITS[source_config["interaction_simulator"]["name"]]["rate"]})' +
                    f'{binding_rates_threshold_upper_text}'),
            req_input=True,
            name='visualise_circuit_interactions'
        ),
        # Binding rates max int's mutations
        Protocol(
            partial(visualise_data, data_writer=data_writer, cols=['binding_rates_max_interaction'],
                    plot_type='histplot',
                    out_name='binding_rates_max_freqs_mutations',
                    preprocessor_func=preprocessing_func,
                    exclude_rows_nonempty_in_cols=exclude_rows_via_cols,
                    threshold_value_max=binding_rates_threshold_upper,
                    log_axis=config_file.get('log_scale', (False, False)),
                    use_sns=True,
                    title=f'Maximum k_d strength, {num_mutations} mutation{plot_grammar}',
                    xlabel='Dissociation rate k_d (' +
                    f'{SIMULATOR_UNITS[source_config["interaction_simulator"]["name"]]["rate"]})' +
                    f'{binding_rates_threshold_upper_text}'),
            req_input=True,
            name='visualise_mutated_interactions'
        ),
        # Binding rates max int's diff
        Protocol(
            partial(visualise_data, data_writer=data_writer, cols=['binding_rates_max_interaction_diff_to_base_circuit'],
                    plot_type='histplot',
                    out_name='binding_rates_max_freqs_diffs',
                    preprocessor_func=preprocessing_func,
                    exclude_rows_nonempty_in_cols=exclude_rows_via_cols,
                    log_axis=config_file.get('log_scale', (False, False)),
                    use_sns=True,
                    title=f'Difference between circuit\nand mutated (minimum k_d), {num_mutations} mutation{plot_grammar}',
                    xlabel='Difference in k_d (' +
                    f'{SIMULATOR_UNITS[source_config["interaction_simulator"]["name"]]["rate"]})' +
                    f'{binding_rates_threshold_upper_text}'),
            req_input=True,
            name='visualise_interactions_difference',
            skip=config_file.get('only_visualise_circuits', False)
        ),
        # Binding rates min int's og
        Protocol(
            partial(visualise_data, data_writer=data_writer, cols=['binding_rates_min_interaction'],
                    plot_type='histplot',
                    out_name='binding_rates_min_freqs',
                    preprocessor_func=preprocessing_func,
                    exclude_rows_nonempty_in_cols=['mutation_name'],
                    threshold_value_max=binding_rates_threshold_upper,
                    log_axis=config_file.get('log_scale', (False, False)),
                    use_sns=True,
                    title='Minimum k_d strength, unmutated circuits',
                    # xlabel='bbb'),
                    xlabel='Dissociation rate k_d (' +
                    f'{SIMULATOR_UNITS[source_config["interaction_simulator"]["name"]]["rate"]})' +
                    f'{binding_rates_threshold_upper_text}'),
            req_input=True,
            name='visualise_circuit_interactions'
        ),
        # Binding rates min int's mutations
        Protocol(
            partial(visualise_data, data_writer=data_writer, cols=['binding_rates_min_interaction'],
                    plot_type='histplot',
                    out_name='binding_rates_min_freqs_mutations',
                    preprocessor_func=preprocessing_func,
                    exclude_rows_nonempty_in_cols=exclude_rows_via_cols,
                    threshold_value_max=binding_rates_threshold_upper,
                    log_axis=config_file.get('log_scale', (False, False)),
                    use_sns=True,
                    title=f'Minimum k_d strength, {num_mutations} mutation{plot_grammar}',
                    xlabel='Dissociation rate k_d (' +
                    f'{SIMULATOR_UNITS[source_config["interaction_simulator"]["name"]]["rate"]})' +
                    f'{binding_rates_threshold_upper_text}'),
            req_input=True,
            name='visualise_mutated_interactions'
        ),
        # Binding rates min int's diff
        Protocol(
            partial(visualise_data, data_writer=data_writer, cols=['binding_rates_min_interaction_diff_to_base_circuit'],
                    plot_type='histplot',
                    out_name='binding_rates_min_freqs_diffs',
                    preprocessor_func=preprocessing_func,
                    exclude_rows_nonempty_in_cols=exclude_rows_via_cols,
                    threshold_value_max=binding_rates_threshold_upper,
                    log_axis=config_file.get('log_scale', (False, False)),
                    use_sns=True,
                    title=f'Difference between circuit\nand mutated (minimum k_d), {num_mutations} mutation{plot_grammar}',
                    xlabel='Difference in k_d (' +
                    f'{SIMULATOR_UNITS[source_config["interaction_simulator"]["name"]]["rate"]})' +
                    f'{binding_rates_threshold_upper_text}'),
            req_input=True,
            name='visualise_interactions_difference',
            skip=config_file.get('only_visualise_circuits', False)
        ),





        
        # eqconstants max int's og
        Protocol(
            partial(visualise_data, data_writer=data_writer, cols=['eqconstants_max_interaction'],
                    plot_type='histplot',
                    out_name='eqconstants_max_freqs',
                    preprocessor_func=preprocessing_func,
                    exclude_rows_nonempty_in_cols=['mutation_name'],
                    log_axis=config_file.get('log_scale', (False, False)),
                    use_sns=True,
                    title='Maximum equilibrium constant, unmutated circuits',
                    xlabel='Equilibrium constant'),
            req_input=True,
            name='visualise_circuit_interactions'
        ),
        # eqconstants max int's mutations
        Protocol(
            partial(visualise_data, data_writer=data_writer, cols=['eqconstants_max_interaction'],
                    plot_type='histplot',
                    out_name='eqconstants_max_freqs_mutations',
                    preprocessor_func=preprocessing_func,
                    exclude_rows_nonempty_in_cols=exclude_rows_via_cols,
                    log_axis=(False, False),
                    use_sns=True,
                    title=f'Maximum equilibrium constant, {num_mutations} mutation{plot_grammar}',
                    xlabel=f'Equilibrium constant'),
            req_input=True,
            name='visualise_mutated_interactions'
        ),
        # Log 0f ^eqconstants max int's mutations
        Protocol(
            partial(visualise_data, data_writer=data_writer, cols=['eqconstants_max_interaction'],
                    plot_type='histplot',
                    out_name='eqconstants_max_freqs_mutations_logarithm',
                    preprocessor_func=preprocessing_func,
                    exclude_rows_nonempty_in_cols=exclude_rows_via_cols,
                    log_axis=(True, False),
                    use_sns=True,
                    title=f'Maximum equilibrium constant, {num_mutations} mutation{plot_grammar}',
                    xlabel=f'Equilibrium constant'),
            req_input=True,
            name='visualise_mutated_interactions'
        ),
        # eqconstants max int's diff
        Protocol(
            partial(visualise_data, data_writer=data_writer, cols=['eqconstants_max_interaction_diff_to_base_circuit'],
                    plot_type='histplot',
                    out_name='eqconstants_max_freqs_diffs',
                    preprocessor_func=preprocessing_func,
                    exclude_rows_nonempty_in_cols=exclude_rows_via_cols,
                    log_axis=config_file.get('log_scale', (False, False)),
                    use_sns=True,
                    title=f'Difference between circuit\nand mutated equilibrium constant, {num_mutations} mutation{plot_grammar}',
                    xlabel='Equilibrium constant difference'),
            req_input=True,
            name='visualise_interactions_difference',
            skip=config_file.get('only_visualise_circuits', False)
        ),
        # Log of ^eqconstants max int's diff
        Protocol(
            partial(visualise_data, data_writer=data_writer, cols=['eqconstants_max_interaction_diff_to_base_circuit'],
                    plot_type='histplot',
                    out_name='eqconstants_max_freqs_diffs_logarithm',
                    preprocessor_func=preprocessing_func,
                    exclude_rows_nonempty_in_cols=exclude_rows_via_cols,
                    log_axis=(True, False),
                    use_sns=True,
                    title=f'Difference between circuit\nand mutated equilibrium constant, {num_mutations} mutation{plot_grammar}',
                    xlabel='Equilibrium constant difference'),
            req_input=True,
            name='visualise_interactions_difference',
            skip=config_file.get('only_visualise_circuits', False)
        ),






        # Analyse results reports
        Protocol()
    ]
    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
