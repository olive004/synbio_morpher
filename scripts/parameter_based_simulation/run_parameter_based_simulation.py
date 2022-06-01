import numpy as np
from scripts.common.circuit import construct_circuit_from_cfg
from src.srv.io.results.result_writer import ResultWriter
from src.srv.parameter_prediction.simulator import SIMULATOR_UNITS
from src.utils.misc.numerical import make_symmetrical_matrix_from_sequence, triangular_sequence
from src.utils.system_definition.agnostic_system.system_manager import CircuitModeller


def main(config=None, writer=None):

    interaction_min = 0
    interaction_max = 1
    interaction_step_size = 0.05
    interaction_array = np.arange(
        interaction_min, interaction_max, interaction_step_size)
    num_species = 3
    num_unique_interactions = triangular_sequence(num_species)
    size_interaction_array = np.size(interaction_array)

    matrix_dimensions = np.power(size_interaction_array, num_species)

    all_species_interactions_x = np.repeat(
        interaction_array, repeats=num_unique_interactions, axis=1)
    all_species_interactions_y = np.zeros(np.shape(all_species_interactions_x))
    for i in np.size(all_species_interactions_x):
        interaction_set = all_species_interactions_x[i]
        interaction_matrix = np.zeros((num_species, num_species))
        {
            "interactions_matrix": interaction_matrix,
            "interactions_units": SIMULATOR_UNITS['IntaRNA']['rate']
        }

    num_iterations = np.power(size_interaction_array, num_unique_interactions)
    iterator_types = np.arange(1, num_unique_interactions+1)
    for i in range(num_iterations):
        iterators = np.mod(i, np.power(size_interaction_array, iterator_types))
        flat_triangle = np.zeros(num_unique_interactions)
        for j in range(len(iterator_types)):
            v = np.mod(i, np.power(size_interaction_array, j))
            flat_triangle[j] = all_species_interactions_x[v]
        interaction_matrix = make_symmetrical_matrix_from_sequence(
            flat_triangle, num_species)
        cfg = {"interactions": {
            "interactions_matrix": interaction_matrix,
            "interactions_units": SIMULATOR_UNITS['IntaRNA']['rate']}
        }

    circuit = construct_circuit_from_cfg(
        extra_configs=None, config_file=config)
    data_writer = ResultWriter(purpose='parameter_based_simulation')
    CircuitModeller(result_writer=data_writer).find_steady_states()
