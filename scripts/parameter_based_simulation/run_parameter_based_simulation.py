import logging
import os
import numpy as np
from scripts.common.circuit import construct_circuit_from_cfg
from src.srv.io.results.result_writer import ResultWriter
from src.srv.parameter_prediction.simulator import SIMULATOR_UNITS
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.numerical import make_symmetrical_matrix_from_sequence, triangular_sequence
from src.utils.misc.type_handling import flatten_listlike
from src.utils.system_definition.agnostic_system.system_manager import CircuitModeller


def main(config=None, data_writer=None):
    if config is None:
        config = os.path.join('scripts', 'parameter_based_simulation', 'configs', 'base_config.json')
    config_file = load_json_as_dict(config)
    if data_writer is None:
        data_writer = ResultWriter(purpose=config_file.get('experiment').get('purpose', 'parameter_based_simulation'))

    interaction_min = 0
    interaction_max = 1
    interaction_step_size = 0.05
    interaction_array = np.arange(
        interaction_min, interaction_max, interaction_step_size)
    size_interaction_array = np.size(interaction_array)

    from src.utils.data.data_format_tools.manipulate_fasta import load_seq_from_FASTA
    sample_names = load_seq_from_FASTA(config_file.get("data_path"), as_type='dict')
    num_species = len(sample_names)
    num_unique_interactions = triangular_sequence(num_species)

    matrix_dimensions = tuple([num_species] + [size_interaction_array]*num_unique_interactions)
    matrix_size = num_species * np.power(size_interaction_array, num_unique_interactions)
    assert matrix_size == np.prod(list(matrix_dimensions)), 'Something is off about the intended size of the matrix'

    all_unique_interactions = np.repeat(
        [interaction_array], repeats=num_unique_interactions, axis=0)
    logging.info(all_unique_interactions)
    all_species_steady_states = np.zeros(matrix_dimensions)

    logging.info("num_species")
    logging.info(num_species)
    logging.info("num_unique_interactions")
    logging.info(num_unique_interactions)
    logging.info("size_interaction_array")
    logging.info(size_interaction_array)
    logging.info("matrix_dimensions")
    logging.info(matrix_dimensions)
    logging.info("all_species_steady_states")
    logging.info(np.shape(all_species_steady_states))
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
            "interactions_units": SIMULATOR_UNITS['IntaRNA']['rate']},
            "sample_names": sample_names
        }
        logging.info("interaction_matrix")
        logging.info(interaction_matrix)

        circuit = construct_circuit_from_cfg(
            extra_configs=cfg, config_filepath=config)
        circuit = CircuitModeller(result_writer=data_writer).find_steady_states(circuit)
        idx = [np.mod(i, np.power(size_interaction_array, j)) for j in range(num_unique_interactions)]
        logging.info("idx")
        logging.info(idx)
        logging.info("circuit.species.steady_state_copynums")
        logging.info(circuit.species.steady_state_copynums)
        all_species_steady_states[:, 0,0,0,0,0,0] = flatten_listlike(circuit.species.steady_state_copynums)
        logging.info(all_species_steady_states)
        break
