from fire import Fire
import os
from scripts.common.circuit import construct_circuit_from_cfg
from src.srv.io.results.writer import DataWriter

from src.utils.misc.decorators import time_it
from src.utils.system_definition.agnostic_system.system_manager import CircuitModeller


@time_it
def main(config_filepath=None):

    # from src.utils.data.fake_data_generation.seq_generator import RNAGenerator
    # RNAGenerator(purpose='example_data').generate_circuit(
    #     count=3, slength=25, protocol="template_mix")


    config_filepath = os.path.join(
        "scripts", "RNA_circuit_simulation", "configs", "loaded_circuit.json")


    data_writer_kwargs = {'purpose': 'RNA_circuit_simulation'}
    data_writer = DataWriter(**data_writer_kwargs)

    circuit = construct_circuit_from_cfg(config_filepath=config_filepath)
    modeller = CircuitModeller(result_writer=data_writer)
    circuit = modeller.compute_interaction_strengths(circuit)
    circuit = modeller.visualise_graph(circuit)


if __name__ == "__main__":
    Fire(main)
