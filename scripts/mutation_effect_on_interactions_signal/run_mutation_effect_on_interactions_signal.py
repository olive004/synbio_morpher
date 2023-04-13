from functools import partial
import logging
import os
from fire import Fire

from src.utils.common.setup_new import construct_circuit_from_cfg, prepare_config
from src.srv.io.manage.script_manager import script_preamble
from src.utils.results.experiments import Experiment, Protocol
from src.srv.sequence_exploration.sequence_analysis import pull_circuits_from_stats
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.evolution.evolver import Evolver
from src.utils.misc.io import get_pathnames
from src.utils.misc.scripts_io import get_search_dir
from src.utils.common.setup_new import expand_config
from src.utils.circuit.agnostic_circuits.circuit_manager_new import CircuitModeller


def main(config=None, data_writer=None):
    # Set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        "scripts", "mutation_effect_on_interactions_signal", "configs", "base_config.json"))
    config_file = load_json_as_dict(config)

    # Start_experiment
    config_file, source_experiment_dir = get_search_dir(
        config_searchdir_key='source_of_interaction_stats', config_file=config_file)
    config_file = expand_config(config=config_file)
    config_file = prepare_config(config_file)

    def logging_circuit(clist: list):
        logging.warning(f'\t\tSimulating {len(clist)} circuits')
        return clist

    protocols = [
        # Load in templates: pathname for circuit stats is in config
        Protocol(
            partial(get_pathnames,
                    first_only=True,
                    file_key="circuit_stats",
                    search_dir=source_experiment_dir,
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
        Protocol(logging_circuit, req_input=True,
                 req_output=True, name='logging'),
        [  # for each circuit
            # Construct circuit
            Protocol(
                partial(construct_circuit_from_cfg,
                        config_file=config_file),
                req_input=True,
                req_output=True,
                name='construct_circuit'
            )
        ],
        [
            # Mutate circuit
            Protocol(
                partial(Evolver(data_writer=data_writer, sequence_type=config_file.get('system_type')).mutate,
                        write_to_subsystem=True,
                        algorithm=config_file.get('mutations_args', {}).get('algorithm', 'random')),
                req_input=True,
                req_output=True,
                name="generate_mutations"
            )
        ]
    ]

    if not config_file['simulation']['use_batch_mutations']:
        protocols[-1].append(
            # Simulate signal and write results
            Protocol(partial(CircuitModeller(result_writer=data_writer, config=config_file).wrap_mutations,
                             write_to_subsystem=True,
                             methods={
                "init_circuit": {},
                "simulate_signal": {'ref_circuit': None,
                                    'solver': config_file['signal'].get('solver', 'naive')},
                "write_results": {'no_visualisations': config_file['experiment'].get('no_visualisations', False),
                                  'no_numerical': config_file['experiment'].get('no_numerical', False)}
            }),
                req_input=True,
                name="simulate_visualisations"
            )
        )
    else:
        protocols.append(
            # Simulate signal and write results
            Protocol(partial(CircuitModeller(result_writer=data_writer, config=config_file).batch_circuits,
                             write_to_subsystem=True, batch_size=config_file['simulation'].get('batch_size', 100),
                             methods={
                "compute_interactions": {},
                "init_circuits": {'batch': True},
                "simulate_signal_batch": {'ref_circuit': None,
                                          'batch': True},
                "write_results": {'no_visualisations': config_file['experiment'].get('no_visualisations', True),
                                  'no_numerical': config_file['experiment'].get('no_numerical', False)}
            }),
                req_input=True,
                name="simulate_visualisations"
            ))

    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer, debug_inputs=False)
    experiment.run_experiment()

    return config_file, data_writer


if __name__ == "__main__":
    Fire(main)
