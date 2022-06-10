from functools import partial
import os

from fire import Fire
from scripts.common.circuit import construct_circuit_from_cfg

from src.srv.io.results.experiments import Experiment, Protocol
from src.srv.io.results.result_writer import ResultWriter
from src.srv.sequence_exploration.sequence_analysis import pull_circuits_from_stats
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.evolution.mutation import Evolver
from src.utils.misc.io import get_pathnames, get_recent_experiment_folder, get_search_dir
from src.utils.system_definition.agnostic_system.system_manager import CircuitModeller


def main(config=None, data_writer=None):
    # Set configs
    if config is None:
        config = os.path.join(
            "scripts", "mutation_effect_on_interactions_signal", "configs", "fixed", "mutations_1_config.json")
    config_file = load_json_as_dict(config)

    # Start_experiment
    if data_writer is None:
        data_writer_kwargs = {'purpose': config_file.get(
            'purpose', 'mutation_effect_on_interactions_signal')}
        data_writer = ResultWriter(**data_writer_kwargs)

    config_file, source_experiment_dir = get_search_dir('source_of_interaction_stats', config_file=config_file)
    protocols = [
        # Load in templates: pathname for circuit stats is in config
        Protocol(
            partial(get_pathnames,
                    first_only=True,
                    file_key="circuit_stats",
                    search_dir=source_experiment_dir
                    ),
            req_output=True,
            name='get_pathname'
        ),
        # Filter circuits
        Protocol(
            partial(pull_circuits_from_stats,
                    filters=config_file.get("filters", {})),
            req_input=True,
            req_output=True,
            name='pull_circuit_from_stats'
        ),
        [ # for each circuit
            # Construct circuit
            Protocol(
                partial(construct_circuit_from_cfg,
                        config_file=config_file),
                req_input=True,
                req_output=True,
                name='construct_circuit'
            ),
            # Mutate circuit
            Protocol(
                partial(Evolver(data_writer=data_writer).mutate,
                        write_to_subsystem=True),
                req_input=True,
                req_output=True,
                name="generate_mutations"
            ),
            # Simulate signal and write results
            Protocol(
                partial(CircuitModeller(result_writer=data_writer).wrap_mutations,
                        write_to_subsystem=True,
                        methods={
                    "init_circuit": {},
                    "simulate_signal": {'save_numerical_vis_data': True},
                    "write_results": {}
                }
                ),
                req_input=True,
                name="simulate_visualisations"
            )
        ]
    ]
    experiment = Experiment(config_filepath=config, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()

    return config_file, data_writer


if __name__ == "__main__":
    Fire(main)
