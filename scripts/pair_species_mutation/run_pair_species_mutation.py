from functools import partial
import os

from fire import Fire
from scripts.common.circuit import construct_circuit_from_cfg

from src.srv.io.results.experiments import Experiment, Protocol
from src.srv.io.results.result_writer import ResultWriter
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.data.fake_data_generation.seq_generator import RNAGenerator
from src.utils.evolution.mutation import Evolver
from src.utils.system_definition.agnostic_system.system_manager import CircuitModeller


def main():
    # set configs
    config_filepath = os.path.join(
        "scripts", "pair_species_mutation", "configs", "RNA_pair_species_mutation.json")
    config_file = load_json_as_dict(config_filepath)

    # Start_experiment
    data_writer_kwargs = {'purpose': 'pair_species_mutation'}
    data_writer = ResultWriter(**data_writer_kwargs)
    default_generator_params = {
        "count": 3, "slength": 25, "protocol": "template_mix"}
    protocols = [
        Protocol(
            partial(RNAGenerator(data_writer=data_writer).generate_circuit,
                    **config_file.get('circuit_generator_params', default_generator_params)),
            name="generating_sequences",
            req_output=True
        ),
        Protocol(
            partial(construct_circuit_from_cfg,
                    config_filepath=config_filepath),
            req_input=True,
            req_output=True,
            name="making_circuit"
        ),
        Protocol(
            Evolver(data_writer=data_writer).mutate,
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
    experiment = Experiment(config_filepath, protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
