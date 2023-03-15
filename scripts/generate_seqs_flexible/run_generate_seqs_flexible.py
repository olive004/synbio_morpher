from functools import partial
import os
from fire import Fire

from src.srv.io.manage.script_manager import script_preamble
from src.utils.common.setup_new import construct_circuit_from_cfg, prepare_config
from src.utils.circuit.agnostic_circuits.circuit_manager_new import CircuitModeller
from src.utils.evolution.evolver import Evolver
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.data.fake_data_generation.seq_generator import RNAGenerator
from src.utils.results.experiments import Experiment, Protocol
from src.utils.misc.type_handling import flatten_listlike


def generate_multiple_circuits(exp_configs, data_writer):
    circuit_paths = []
    for i, (r, s, l, gp, p, t) in enumerate(zip(*list(exp_configs.values()))):
        name = 'toy_circuit_combo' + str(i)
        circuit_paths.append(RNAGenerator(data_writer=data_writer).generate_circuits_batch(
            name=name,
            num_circuits=r,
            num_components=s, slength=l,
            proportion_to_mutate=p,
            protocol=gp,
            template=t),
        )
    return flatten_listlike(circuit_paths)


def sub_dir(data_writer, dirname):
    data_writer.update_ensemble(dirname)


def main(config=None, data_writer=None):
    # set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        "scripts", "generate_seqs_flexible", "configs", "base_config.json"))
    config_file = load_json_as_dict(config)
    config_file = prepare_config(config_file)
    exp_configs = config_file["circuit_generation"]

    protocols = [
        Protocol(
            partial(generate_multiple_circuits,
                    exp_configs=exp_configs,
                    data_writer=data_writer),
                # req_input=True,
                req_output=True,
                name="generate_circuits"
        ),
        Protocol(
            partial(sub_dir, 
                    data_writer=data_writer, dirname='simulated_circuits')
        ),
        [
            Protocol(
                partial(construct_circuit_from_cfg,
                        config_file=config_file),
                req_input=True,
                req_output=True,
                name="making_circuit"
            )
        ],
        [
            Protocol(
                partial(Evolver(data_writer=data_writer, sequence_type=config_file.get('system_type')).mutate,
                        write_to_subsystem=True,
                        algorithm=config_file.get('mutations', {}).get('algorithm')),
                req_input=True,
                req_output=True,
                name="generate_mutations"
            )
        ],
        Protocol(partial(CircuitModeller(result_writer=data_writer, config=config_file).batch_circuits,
                         write_to_subsystem=True, batch_size=config_file['simulation'].get('batch_size', 100),
                         methods={
            "compute_interactions": {},
            "find_steady_states": {'batch': True},
            "write_results": {'no_visualisations': config_file['experiment'].get('no_visualisations', True),
                              'no_numerical': config_file['experiment'].get('no_numerical', False)}
        }),
            req_input=True,
            name="simulate_visualisations"
        )
    ]
    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
