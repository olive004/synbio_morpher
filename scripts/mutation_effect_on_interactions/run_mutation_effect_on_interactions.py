from functools import partial
import os
import sys

from fire import Fire
from scripts.common.circuit import construct_circuit_from_cfg

from src.srv.io.results.experiments import Experiment, Protocol
from src.srv.io.results.result_writer import ResultWriter
from src.srv.sequence_exploration.sequence_analysis import pull_circuits_from_stats
from src.utils.data.data_format_tools.common import load_json_as_dict, process_json
from src.utils.evolution.mutation import Evolver
from src.utils.misc.io import get_pathnames
from src.utils.system_definition.agnostic_system.system_manager import CircuitModeller


def main(config_filepath=None):
    # set configs
    if config_filepath is None:
        config_filepath = os.path.join(
            "scripts", "mutation_effect_on_interactions", "configs", "base_mutation_config.json")
    config_file = process_json(load_json_as_dict(config_filepath))
    # start_experiment
    data_writer_kwargs = {'purpose': 'mutation_effect_on_interactions'}
    data_writer = ResultWriter(**data_writer_kwargs)

    source_experiment_dir = os.path.join(*list({
        'root_dir': "data",
        'purpose': "explore_species_templates",
        'experiment_key': "2022_04_27_154019",
    }.values()))
    protocols = [
        # load in templates: pathname for circuit stats is in config
        Protocol(
            partial(get_pathnames,
                    first_only=True,
                    file_key="circuit_stats",
                    search_dir=source_experiment_dir
                    ),
            req_output=True,
            name='get_pathname'
        ),
        # filter circuits
        Protocol(
            partial(pull_circuits_from_stats,
                    filters=config_file.get("filters")),
            req_input=True,
            req_output=True,
            name='pull_circuit_from_stats'
        ),
        [
            # construct circuit
            Protocol(
                partial(construct_circuit_from_cfg,
                        config_filepath=config_filepath),
                req_input=True,
                req_output=True,
                name='construct_circuit'
            ),
            # compile results and write to
            Protocol(
                Evolver(data_writer=data_writer).mutate,
                req_input=True,
                req_output=True,
                name="generate_mutations"
            ),
            # run interaction simulator
            Protocol(
                partial(CircuitModeller(result_writer=data_writer).wrap_mutations, methods={
                    "init_circuit": {},
                    "simulate_signal": {'save_numerical_vis_data': True},
                    "write_results": {}
                }
                ),
                req_input=True,
                name="visualise_signal"
            )
            # Protocol(sys.exit), 
        ]
    ]
    experiment = Experiment(config_filepath, protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
