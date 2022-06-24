

import os
from src.srv.io.loaders.data_loader import DataLoader, GeneCircuitLoader
from src.srv.io.results.result_writer import ResultWriter
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.io import get_pathnames, isolate_filename
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
    parameter_grids = load_parameter_grids()

    # For each parameter grid

    # Slice in 2D or 3D
    def make_slice():

        # Choose which type of interactions to vary: self vs. inter
        # --> return dimension indices
        # --> get choice from config
        # --> might need lookup table
        config_file['species_interactions_to_vary'] = ['RNA1', 'RNA2']
        def get_species_interaction_idx():
            circuit_filepath = config_file('filepath')
            data = GeneCircuitLoader().load_data(circuit_filepath)
            return data.names
        species_interaction_flat_triangle_lookup = {

        }
        species_interactions_to_vary = config_file.get('species_interactions_to_vary')

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

