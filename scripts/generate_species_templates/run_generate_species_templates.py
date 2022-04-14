from functools import partial
import logging
import os
import sys

from fire import Fire
from scripts.common.circuit import construct_circuit_from_cfg

from src.srv.results.experiments import Experiment, Protocol
from src.srv.results.result_writer import ResultWriter
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.data.fake_data_generation.seq_generator import RNAGenerator
from src.utils.system_definition.agnostic_system.system_manager import CircuitModeller


def main():
    ## set configs
    config_file = os.path.join(
        "scripts", "generate_species_templates", "configs", "generate_species_templates.json")
    exp_configs = load_json_as_dict(config_file).get("experiment")
    logging.info(exp_configs)

    ## start_experiment
    data_writer_kwargs = {'purpose': 'generate_species_templates'}
    data_writer = ResultWriter(**data_writer_kwargs)
    protocols = [
        Protocol(
            partial(RNAGenerator(data_writer=data_writer).generate_circuits,
                    iter_count=exp_configs.get("repetitions"),
                    count=3, slength=exp_configs.get("sequence_length"), 
                    protocol=exp_configs.get("generator_protocol")),
            name="generating_sequences",
            req_output=True
        ), [
            Protocol(
                partial(construct_circuit_from_cfg, config_file=config_file),
                req_input=True,
                req_output=True,
                name="making_circuit"
            ),
            # Protocol(sys.exit),
            Protocol(
                CircuitModeller(result_writer=data_writer).compute_interaction_strengths,
                req_input=True,
                name="compute_interaction_strengths"
            )
        ]
    ]
    experiment = Experiment(config_file, protocols, data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
