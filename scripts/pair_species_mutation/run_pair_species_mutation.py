from functools import partial
import os

from fire import Fire
from src.utils.circuit.agnostic_circuits.circuit_manager_new import construct_circuit_from_cfg
from src.utils.results.experiments import Experiment, Protocol
from src.utils.results.result_writer import ResultWriter
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.data.fake_data_generation.seq_generator import RNAGenerator
from src.utils.evolution.mutation import Evolver
from src.utils.circuit.agnostic_circuits.circuit_manager import CircuitModeller


def main(config=None, data_writer=None):
    # set configs
    if config is None:
        config = os.path.join(
            "scripts", "pair_species_mutation", "configs", "RNA_pair_species_mutation.json")
    config_file = load_json_as_dict(config)
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
                    algorithm=config_file.get('mutations', {}).get('algorithm', "random")),
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


if __name__ == "__main__":
    Fire(main)
