from functools import partial
import logging
import os
from fire import Fire

from src.utils.common.setup_new import construct_circuit_from_cfg
from src.srv.io.manage.script_manager import script_preamble
from src.utils.results.experiments import Experiment, Protocol
from src.utils.results.result_writer import ResultWriter
from src.srv.sequence_exploration.sequence_analysis import pull_circuits_from_stats
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.evolution.mutation import Evolver
from src.utils.misc.io import get_pathnames
from src.utils.misc.scripts_io import get_search_dir
from src.utils.circuit.agnostic_circuits.circuit_manager_new import CircuitModeller


def main(config=None, data_writer=None):
    # Set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        # "scripts", "mutation_effect_on_interactions_signal", "configs", "base_mutation_config_test.json")
        # "scripts", "mutation_effect_on_interactions_signal", "configs", "base_mutation_config_1.json")
        # "scripts", "mutation_effect_on_interactions_signal", "configs", "base_mutation_config_2.json")
        # "scripts", "mutation_effect_on_interactions_signal", "configs", "base_mutation_config_10.json")
        # "scripts", "mutation_effect_on_interactions_signal", "configs", "base_mutation_config_20.json")
        # "scripts", "mutation_effect_on_interactions_signal", "configs", "base_mutation_config_1_highmag.json")
        # "scripts", "mutation_effect_on_interactions_signal", "configs", "base_mutation_config_2_highmag.json")
        # "scripts", "mutation_effect_on_interactions_signal", "configs", "base_mutation_config_10_highmag.json")
        "scripts", "mutation_effect_on_interactions_signal", "configs", "base_mutation_config_20_highmag.json"))
    config_file = load_json_as_dict(config)

    # Start_experiment
    if data_writer is None:
        data_writer_kwargs = {'purpose': config_file.get(
            'purpose', 'mutation_effect_on_interactions_signal')}
        data_writer = ResultWriter(**data_writer_kwargs)

    config_file, source_experiment_dir = get_search_dir(
        config_searchdir_key='source_of_interaction_stats', config_file=config_file)

    if not config_file['simulation']['use_batch_mutations']:
        simulation_func = partial(CircuitModeller(result_writer=data_writer, config=config_file).wrap_mutations,
                                  write_to_subsystem=True,
                                  methods={
            "init_circuit": {},
            "simulate_signal": {'save_numerical_vis_data': True, 'ref_circuit': None,
                                'solver': config_file['signal'].get('solver', 'naive')},
            "write_results": {'no_visualisations': config_file['experiment'].get('no_visualisations', False)}
        })
    else:
        simulation_func = partial(CircuitModeller(result_writer=data_writer, config=config_file).batch_mutations,
                                  write_to_subsystem=True,
                                  methods={
            "init_circuit": {},
            "simulate_signal_batch": {'save_numerical_vis_data': True, 'ref_circuit': None,
                                      'batch': True},
            "write_results": {'no_visualisations': config_file['experiment'].get('no_visualisations', True)}
        })

    def logging_circuit(clist: list):
        logging.info(f'Simulating {len(clist)} circuits')
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
        Protocol(logging_circuit, req_input=True, req_output=True, name='logging'),
        [  # for each circuit
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
                partial(Evolver(data_writer=data_writer, sequence_type=config_file.get('system_type')).mutate,
                        write_to_subsystem=True,
                        algorithm=config_file.get('mutations', {}).get('algorithm', 'random')),
                req_input=True,
                req_output=True,
                name="generate_mutations"
            ),
            # Simulate signal and write results
            Protocol(simulation_func,
                     req_input=True,
                     name="simulate_visualisations"
                     )
        ]
    ]
    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer, debug_inputs=False)
    experiment.run_experiment()

    return config_file, data_writer


if __name__ == "__main__":
    Fire(main)
