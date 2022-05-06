from functools import partial
import os
import sys

from fire import Fire
from scripts.common.circuit import construct_circuit_from_cfg

from src.srv.io.results.experiments import Experiment, Protocol
from src.srv.io.results.result_writer import ResultWriter
from src.srv.io.results.visualisation import visualise_data
from src.srv.sequence_exploration.sequence_analysis import pull_circuits_from_stats, tabulate_mutation_info
from src.utils.data.data_format_tools.common import load_json_as_dict, process_json
from src.utils.evolution.mutation import Evolver
from src.utils.misc.io import get_pathnames
from src.utils.system_definition.agnostic_system.system_manager import CircuitModeller


def main(config_filepath=None):
    # Set configs
    if config_filepath is None:
        config_filepath = os.path.join(
            "scripts", "mutation_effect_on_interactions", "configs", "fixed", "mutations_1_config.json")
    config_file = process_json(load_json_as_dict(config_filepath))
    # Start_experiment
    data_writer_kwargs = {'purpose': 'analyse_mutated_templates'}
    data_writer = ResultWriter(**data_writer_kwargs)

    source_dir = os.path.join(
        'data', 'mutation_effect_on_interactions', '2022_05_06_011502'
    )
    protocols = [
        Protocol(
            partial(tabulate_mutation_info, source_dir=source_dir,
                    data_writer=data_writer),
            req_output=True,
            name='tabulate_mutation_info'
        ),
        Protocol(
            partial(visualise_data, data_writer=data_writer, cols=[
                'interaction_strength'
            ], plot_type='histplot', title=''),
            req_input=True,
            name='visualise_interactions'
        ),
    ]
    experiment = Experiment(config_filepath, protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
