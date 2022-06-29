

from functools import reduce
import os

import numpy as np
from src.srv.io.loaders.data_loader import DataLoader, GeneCircuitLoader
from src.srv.io.results.result_writer import ResultWriter
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.io import get_pathnames, isolate_filename
from src.utils.misc.numerical import expand_matrix_triangle_idx, triangular_sequence
from src.utils.misc.scripts_io import get_search_dir


def main(config=None, data_writer=None):

    if config is None:
        config = os.path.join(
            'scripts', 'parameter_grid_analysis', 'configs', 'base_config.json')
    config_file = load_json_as_dict(config)
    if data_writer is None:
        data_writer = ResultWriter(purpose=config_file.get(
            'experiment').get('purpose', 'parameter_based_simulation'))

    # Load in parameter grid
    config_file, source_dir = get_search_dir(
        'source_parameter_dir', config_file=config_file)

    def load_parameter_grids():
        all_parameter_grids = {}
        parameter_grids = get_pathnames(source_dir, 'npy')
        for parameter_grid in parameter_grids:
            all_parameter_grids[isolate_filename(
                parameter_grid)] = DataLoader().load_data(parameter_grid)
        return all_parameter_grids
    all_parameter_grids = load_parameter_grids()
    num_species = np.shape(
        all_parameter_grids[list(all_parameter_grids.keys())[0]])

    # For each parameter grid

    # Slice in 2D or 3D
    def make_slice():

        # Choose which type of interactions to vary: self vs. inter
        # --> return dimension indices
        # --> get choice from config
        # --> might need lookup table
        # Example
        config_file['species_interactions_to_vary'] = [
            ['RNA1', 'RNA2'],
            ['RNA2', 'RNA2']
        ]
        species_interactions_to_vary_idxs = [
            [0, 1],
            [1, 1]
        ]
        # /Example

        def translate_interaction():
            species_interactions_index_map = {}
            flat_triangle_size = triangular_sequence(num_species)
            for combination in range(flat_triangle_size):
                species_interactions_index_map[combination] = expand_matrix_triangle_idx(
                    combination)
            return species_interactions_index_map
        species_interaction_flat_triangle_lookup = {

        }

        def translate_species_idx_to_name(idx):
            circuit_filepath = config_file('filepath')
            data = GeneCircuitLoader().load_data(circuit_filepath)
            return data.sample_names[translate_species_to_idx(idx)]

        def translate_species_to_idx(species: str, species_list: list):
            return species_list.index(species)

        species_interactions_to_vary_idxs = reduce(lambda x: translate_species_to_idx(x),
                                                   config_file.get('species_interactions_to_vary'))

        # Keep the rest constant - choose constant value for these
        # --> determine the index corresponding to this value
        def parameter_to_index(parameter):
            pass

        # Might want a slice window for the ones that are varying
        # --> make config accept "all" or named basis of starting and ending parameters
        if config_file.get('slice_choice') == 'all':
            slice = 
        start_parameter = parameter_to_index()
        end_parameter = parameter_to_index()
        slice_window = slice(start_parameter, end_parameter)
