from functools import partial
import logging
import os

from fire import Fire
from scripts.common.circuit import construct_circuit_from_cfg

from src.srv.results.experiments import Experiment, Protocol
from src.srv.results.result_writer import ResultWriter
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.data.fake_data_generation.seq_generator import RNAGenerator
from src.utils.system_definition.agnostic_system.system_manager import CircuitModeller


def main():
    # set configs
    config_file = os.path.join(
        "scripts", "explore_species_templates", "configs", "explore_species_templates.json")
    exp_configs = load_json_as_dict(config_file)
    logging.info(exp_configs)
    exp_configs = exp_configs.get("experiment")
    # start_experiment
    data_writer_kwargs = {'purpose': 'pair_species_mutation'}
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
            Protocol(
                CircuitModeller(result_writer=data_writer).init_circuit,
                req_input=True,
                req_output=True,
                name="init_circuit"
            ),
            Protocol(
                CircuitModeller(result_writer=data_writer).visualise,
                req_input=True,
                name="visualise_circuit"
            )
        ]
        # Protocol(sys.exit),
    ]
    experiment = Experiment(config_file, protocols, data_writer=data_writer)
    experiment.run_experiment()

    # output_visualisations
    # write_report


if __name__ == "__main__":
    Fire(main)
