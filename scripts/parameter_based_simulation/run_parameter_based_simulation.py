import logging
import os
import numpy as np
from scripts.common.circuit import construct_circuit_from_cfg
from src.srv.io.results.result_writer import ResultWriter
from src.srv.parameter_prediction.simulator import SIMULATOR_UNITS
from src.utils.misc.numerical import make_symmetrical_matrix_from_sequence, triangular_sequence
from src.utils.system_definition.agnostic_system.system_manager import CircuitModeller


def main(config=None, data_writer=None):
    if config is None:
        config = os.path.join('scripts', 'parameter_based_simulation', 'configs', 'base_config.json')
    if data_writer is None:
        data_writer = ResultWriter(purpose='parameter_based_simulation')

    interaction_min = 0
    interaction_max = 1
    interaction_step_size = 0.05
    interaction_array = np.arange(
        interaction_min, interaction_max, interaction_step_size)
    size_interaction_array = np.size(interaction_array)

    num_species = 2
    num_unique_interactions = triangular_sequence(num_species)

    matrix_dimensions = tuple([size_interaction_array]*num_unique_interactions)
    matrix_size = np.power(size_interaction_array, num_unique_interactions)
    assert matrix_size == np.prod(list(matrix_dimensions)), 'Something is off about the intended size of the matrix'

    all_unique_interactions = np.repeat(
        [interaction_array], repeats=num_unique_interactions, axis=0)
    logging.info(all_unique_interactions)
    all_species_steady_states = np.zeros(matrix_dimensions)

    logging.info(num_species)
    logging.info(num_unique_interactions)
    logging.info(size_interaction_array)
    logging.info(matrix_dimensions)
    num_iterations = matrix_size
    for i in range(num_iterations):
        logging.info(f'Iteration {i}/{num_iterations}')
        flat_triangle = np.zeros(num_unique_interactions)
        for j in range(num_unique_interactions):
            v = np.mod(i, np.power(size_interaction_array, j))
            flat_triangle[j] = interaction_array[v]
        interaction_matrix = make_symmetrical_matrix_from_sequence(
            flat_triangle, num_species)
        cfg = {"interactions": {
            "interactions_matrix": interaction_matrix,
            "interactions_units": SIMULATOR_UNITS['IntaRNA']['rate']}
        }

        circuit = construct_circuit_from_cfg(
            extra_configs=cfg, config_filepath=config)
        circuit = CircuitModeller(result_writer=data_writer).find_steady_states(circuit)
        idx = [np.mod(i, np.power(size_interaction_array, j)) for j in range(num_unique_interactions)]
        all_species_steady_states[idx] = circuit.species.steady_state_copynums
        break
