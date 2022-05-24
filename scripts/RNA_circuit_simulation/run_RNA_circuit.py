

import os
from fire import Fire
from scripts.common.circuit import construct_circuit_from_cfg
from src.srv.io.results.result_writer import ResultWriter
from src.utils.misc.decorators import time_it
from src.utils.system_definition.agnostic_system.system_manager import CircuitModeller


@time_it
def main(config_filepath: str = None, data_writer=None):

    # from src.utils.data.fake_data_generation.seq_generator import RNAGenerator
    # RNAGenerator(purpose='example_data').generate_circuit(
    #     count=3, slength=25, protocol="template_mix")

    if config_filepath is None:
        config_filepath = os.path.join(
            "scripts", "RNA_circuit_simulation", "configs", "loaded_circuit.json")

    if data_writer is None:
        data_writer_kwargs = {'purpose': 'RNA_circuit_simulation'}
        data_writer = ResultWriter(**data_writer_kwargs)

    circuit = construct_circuit_from_cfg(None, config_filepath=config_filepath)
    modeller = CircuitModeller(result_writer=data_writer)
    circuit = modeller.compute_interaction_strengths(circuit)
    circuit = modeller.visualise_graph(circuit)


if __name__ == "__main__":
    Fire(main)
