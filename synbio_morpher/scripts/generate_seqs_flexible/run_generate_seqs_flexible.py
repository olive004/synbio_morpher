
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
from functools import partial
import os
from fire import Fire
import numpy as np

from synbio_morpher.srv.io.manage.script_manager import script_preamble
from synbio_morpher.utils.common.setup import construct_circuit_from_cfg, prepare_config
from synbio_morpher.utils.circuit.agnostic_circuits.circuit_manager import CircuitModeller
from synbio_morpher.utils.evolution.evolver import Evolver
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
from synbio_morpher.utils.data.fake_data_generation.seq_generator import RNAGenerator
from synbio_morpher.utils.results.experiments import Experiment, Protocol
from synbio_morpher.utils.misc.type_handling import flatten_listlike


def generate_multiple_circuits(exp_configs, data_writer):
    circuit_paths = []
    for i, (r, s, l, gp, p, t, seed) in enumerate(zip(*list(exp_configs.values()))):
        name = 'toy_circuit_combo' + str(i)
        circuit_paths.append(RNAGenerator(data_writer=data_writer, seed=seed).generate_circuits_batch(
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


def batch_circuits(circuit_cfgs: list, config_file: dict, data_writer, max_circuits=1e4):
    for b in range(0, len(circuit_cfgs), int(max_circuits)):
        bf = int(min([b + max_circuits, len(circuit_cfgs)]))
        curr_circuit_cfgs = circuit_cfgs[b:bf]
        curr_circuits = [None] * len(curr_circuit_cfgs)
        for i, c in enumerate(curr_circuit_cfgs):
            circuit = construct_circuit_from_cfg(
                prev_configs=c, config_file=config_file)
            circuit = Evolver(data_writer=data_writer, sequence_type=config_file.get('system_type'),
                              seed=config_file.get('mutations_args', {}).get('seed', np.random.randint(1000))).mutate(
                circuit=circuit,
                write_to_subsystem=True,
                algorithm=config_file.get('mutations_args', {}).get('algorithm'))
            curr_circuits[i] = circuit

        curr_circuits = CircuitModeller(result_writer=data_writer, config=config_file).batch_circuits(
            circuits=curr_circuits, write_to_subsystem=True,
            batch_size=config_file['simulation'].get('batch_size', 100),
            methods={
                "compute_interactions": {}
                # "init_circuits": {'batch': True},
                # "write_results": {'no_visualisations': config_file['experiment'].get('no_visualisations', True),
                #                   'no_numerical': config_file['experiment'].get('no_numerical', False)}
            })


def main(config=None, data_writer=None):
    # set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        "synbio_morpher", "scripts", "generate_seqs_flexible", "configs", "randoms.json"))
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
        Protocol(
            partial(
                batch_circuits,
                config_file=config_file,
                data_writer=data_writer,
                max_circuits=1e4
            ),
            req_input=True)
    ]
    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()

    return config, data_writer


if __name__ == "__main__":
    Fire(main)
