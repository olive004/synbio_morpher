import numpy as np
from scripts.common.circuit import construct_circuit_from_cfg
from src.srv.io.results.result_writer import ResultWriter
from src.utils.system_definition.agnostic_system.system_manager import CircuitModeller


def main(config=None, writer=None):

    interaction_min = 0
    interaction_max = 1
    interaction_step_size = 0.05
    interaction_array = np.arange(
        interaction_min, interaction_max, interaction_step_size)
    num_species = 3

    matrix_dimensions = np.power(len(interaction_array), num_species)


    circuit = construct_circuit_from_cfg(extra_configs=None, config_filepath=config)
    data_writer = ResultWriter(purpose='parameter_based_simulation')
    CircuitModeller(result_writer=data_writer).find_steady_states()