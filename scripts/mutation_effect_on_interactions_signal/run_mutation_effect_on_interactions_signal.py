from functools import partial
import os

from fire import Fire
from scripts.common.circuit import construct_circuit_from_cfg

from src.srv.io.results.experiments import Experiment, Protocol
from src.srv.io.results.result_writer import ResultWriter
from src.srv.sequence_exploration.sequence_analysis import pull_circuits_from_stats
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.evolution.mutation import Evolver
from src.utils.misc.io import get_pathnames
from src.utils.system_definition.agnostic_system.system_manager import CircuitModeller


def main(config_filepath=None, writer=None):
    # Set configs
    if config_filepath is None:
        config_filepath = os.path.join(
            "scripts", "mutation_effect_on_interactions_signal", "configs", "fixed", "mutations_1_config.json")
    config_file = load_json_as_dict(config_filepath)

    # Start_experiment
    if writer=None:
        data_writer_kwargs = {'purpose': config_file.get('purpose', 'mutation_effect_on_interactions_signal')}
        data_writer = ResultWriter(**data_writer_kwargs)
    else:
        data_writer = writer

    source_experiment_dir = config_file.get(
        'source_dir_species_templates_exploration')
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
        [
            # Construct circuit
            Protocol(
                partial(construct_circuit_from_cfg,
                        config_filepath=config_filepath),
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
    experiment = Experiment(config_filepath, protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
