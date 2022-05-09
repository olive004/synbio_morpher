from functools import partial
import logging
import os
import sys

from fire import Fire
from scripts.common.circuit import construct_circuit_from_cfg

from src.srv.io.results.experiments import Experiment, Protocol
from src.srv.io.results.result_writer import ResultWriter
from src.srv.io.results.visualisation import visualise_data
from src.srv.parameter_prediction.interactions import RawSimulationHandling
from src.srv.sequence_exploration.sequence_analysis import pull_circuits_from_stats, tabulate_mutation_info
from src.utils.data.data_format_tools.common import load_json_as_dict, process_json
from src.utils.evolution.mutation import Evolver
from src.utils.misc.io import get_pathnames
from src.utils.system_definition.agnostic_system.system_manager import CircuitModeller


def main(config_filepath=None):
    # Set configs
    if config_filepath is None:
        config_filepath = os.path.join(
            "scripts", "analyse_mutated_templates", "configs", "analyse_mutated_templates_tester.json")
    config_file = process_json(load_json_as_dict(config_filepath))

    # Start_experiment
    data_writer = ResultWriter(purpose='analyse_mutated_templates')
    source_dir = config_file.get('source_dir')
    if config_file.get('preprocessing_func') == 'rate_to_energy':
        logging.info(config_file.get('preprocessing_func'))
        preprocessing_func = RawSimulationHandling().rate_to_energy,
    else:
        preprocessing_func = None
    
    protocols = [
        Protocol(sys.exit),
        Protocol(
            partial(tabulate_mutation_info, source_dir=source_dir,
                    data_writer=data_writer),
            req_output=True,
            name='tabulate_mutation_info'
        ),
        Protocol(
            partial(visualise_data, data_writer=data_writer, cols=['interaction_strength'],
                    plot_type='histplot', out_name='interaction_strength_freqs',
                    preprocessor_func=preprocessing_func,
                    title='Maximum interaction strength, 1 mutation',
                    xlabel='Interaction strength', ylabel='Frequency count'),
            req_input=True,
            name='visualise_interactions'
        ),
        Protocol(
            partial(visualise_data, data_writer=data_writer, cols=['interaction_strength_diff_to_base_circuit'],
                    plot_type='histplot', out_name='interaction_strength_diffs',
                    preprocessor_func=RawSimulationHandling().rate_to_energy,
                    title='Difference btwn circuit and mutated interaction strengths, 1 mutation',
                    xlabel='Interaction strength difference', ylabel='Frequency count'),
            req_input=True,
            name='visualise_interactions_difference'
        ),
    ]
    experiment = Experiment(config_filepath, protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
