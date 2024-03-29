
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
from functools import partial
import os

from fire import Fire
from synbio_morpher.utils.circuit.agnostic_circuits.circuit_manager import CircuitModeller
from synbio_morpher.utils.results.experiments import Experiment, Protocol
from synbio_morpher.utils.results.result_writer import ResultWriter
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
from synbio_morpher.utils.data.fake_data_generation.seq_generator import RNAGenerator
from synbio_morpher.utils.evolution.evolver import Evolver
from synbio_morpher.utils.common.setup import construct_circuit_from_cfg, prepare_config, expand_config


def main(config=None, data_writer=None):
    # set configs
    if config is None:
        config = os.path.join(
            "synbio_morpher", "scripts", "pair_species_mutation", "configs", "RNA_pair_species_mutation.json")
    config_file = load_json_as_dict(config)
    config_file = prepare_config(config_file)
    exp_configs = config_file.get("circuit_generation", {})

    # Start_experiment
    if data_writer is None:
        data_writer_kwargs = {'purpose': 'pair_species_mutation'}
        data_writer = ResultWriter(**data_writer_kwargs)
    protocols = [
        Protocol(
            partial(RNAGenerator(data_writer=data_writer).generate_circuit,
                    count=exp_configs.get("species_count"), slength=exp_configs.get("sequence_length"),
                    proportion_to_mutate=exp_configs.get(
                        "proportion_to_mutate"),
                    protocol=exp_configs.get("generator_protocol")),
            name="generating_sequences",
            req_output=True
        ),
        Protocol(
            partial(construct_circuit_from_cfg,
                    config_filepath=config),
            req_input=True,
            req_output=True,
            name="making_circuit"
        ),
        Protocol(
            partial(Evolver(data_writer=data_writer).mutate,
                    algorithm=config_file.get('mutations_args', {}).get('algorithm', "random")),
            req_input=True,
            req_output=True,
            name="generate_mutations"
        ),
        Protocol(
            partial(CircuitModeller(result_writer=data_writer).wrap_mutations, methods={
                "init_circuit": {},
                "simulate_signal": {},
                "visualise": {}
            }
            ),
            req_input=True,
            name="visualise_signal"
        )
    ]
    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()

    return config, data_writer

if __name__ == "__main__":
    Fire(main)
