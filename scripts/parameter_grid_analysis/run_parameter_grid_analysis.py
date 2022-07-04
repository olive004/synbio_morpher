

from functools import reduce
import logging
import os
from typing import Union

import numpy as np
from src.srv.io.loaders.data_loader import DataLoader, GeneCircuitLoader
from src.utils.misc.type_handling import merge_dicts
from src.utils.results.analytics.timeseries import Timeseries
from src.utils.results.result_writer import ResultWriter
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.io import get_pathnames, isolate_filename
from src.utils.misc.numerical import expand_matrix_triangle_idx, triangular_sequence
from src.utils.misc.scripts_io import get_search_dir, load_experiment_config, load_experiment_config_original
from src.utils.parameter_inference.interpolation_grid import create_parameter_range
from src.utils.results.results import ResultCollector
from src.utils.system_definition.agnostic_system.modelling import Deterministic


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

    # Get relevant configs
    slicing_configs = config_file['slicing']
    selected_species_interactions = slicing_configs['interactions']['interacting_species']
    unselected_species_interactions = slicing_configs['interactions']['non_varying_species_interactions']
    selected_analytics = slicing_configs['analytics']['names']
    if selected_analytics is None:
        selected_analytics = Timeseries(None).get_analytics_types()

    # Determine the indices to slice the parameter grid with
    def make_slice():

        # Choose which type of interactions to vary: self vs. inter
        # --> return dimension indices
        # --> get choice from config
        # --> might need lookup table

        def translate_interaction():
            species_interactions_index_map = {}
            flat_triangle_size = triangular_sequence(num_species)
            for combination in range(flat_triangle_size):
                species_interactions_index_map[combination] = expand_matrix_triangle_idx(
                    combination)
            return species_interactions_index_map

        def translate_species_to_idx(species: str, species_list: list):
            return species_list.index(species)

        def reduce_interacting_species_idx(pair_species_idxs: list):
            pair_species_idxs = sorted(pair_species_idxs)
            return pair_species_idxs[0] * num_species + pair_species_idxs[1]

        # Keep the rest constant - choose constant value for these
        # --> determine the index corresponding to this value

        def parameter_to_index(parameter):
            pass

        # Might want a slice window for the ones that are varying
        # --> make config accept "all" or named basis of starting and ending parameters
        def make_species_slice(selected_species: list, num_species: int) -> Union[tuple, slice]:
            if type(selected_species) == list:
                selected_species_idxs = sorted(reduce(lambda x: translate_species_to_idx(x),
                                               selected_species))
                if len(selected_species_idxs) == num_species:
                    selected_species = 'all'
                else:
                    return tuple(selected_species_idxs)
            if selected_species == 'all':
                return slice(0, num_species)
            else:
                raise ValueError(
                    'Please input the species you want to include in the visualisations')

        def make_parameter_slice(interacting_species_idxs, parameter_config) -> dict:

            def convert_parameter_values_to_slice(start_value, end_value, step_size) -> slice:

                def make_original_parameter_range(source_dir):
                    original_config = load_experiment_config_original(
                        source_dir, target_purpose='parameter_based_simulations')
                    return create_parameter_range(
                        original_config['parameter_based_simulation'])

                original_parameter_range = make_original_parameter_range(
                    source_dir)
                selected_parameter_range_idxs = np.arange(len(
                    original_parameter_range))[original_parameter_range >= start_value and
                                               original_parameter_range <= end_value]
                return slice(
                    selected_parameter_range_idxs[0], selected_parameter_range_idxs[-1], step_size)

            slice_idx_map = {}
            for grouping, interaction_idx in interacting_species_idxs.items():
                slice_idx_map[interaction_idx] = convert_parameter_values_to_slice(
                    **parameter_config[grouping])

            return slice_idx_map

        # Convert the names of the interacting species to the index of that interacting pair in the interpolation grid
        def collect_parameter_slices(species_interactions: dict, parameter_cfg_keyname: str) -> dict:
            species_idxs = {}
            for species_group, interacting_species_names in species_interactions.items():
                species_idxs[species_group] = reduce_interacting_species_idx(
                    reduce(lambda x: translate_species_to_idx(x), interacting_species_names))

            # species_slice = make_species_slice(selected_species, num_species)
            parameters_slices = make_parameter_slice(
                species_idxs, slicing_configs['interactions'][parameter_cfg_keyname])
            return parameters_slices
        parameters_slices = merge_dicts(collect_parameter_slices(
            selected_species_interactions, parameter_cfg_keyname='strengths'),
            collect_parameter_slices(unselected_species_interactions,
                                     parameter_cfg_keyname='non_varying_strengths'))
        species_slice = make_species_slice(
            slicing_configs['species_choices'], num_species)
        grid_slice = [species_slice] + \
            [p_slice for p_slice in parameters_slices]
        return tuple(grid_slice)

    # Make visualisations for each analytic chosen

    def get_sample_names(source_dir, target_purpose):
        circuit_filepath = load_experiment_config_original(
            source_dir, target_purpose)['data_path']
        data = GeneCircuitLoader().load_data(circuit_filepath)
        return data.sample_names
    
    target_purpose='parameter_based_simulations'
    
    sample_names = get_sample_names(
        source_dir, target_purpose)
    slice_indices = make_slice()
    result_collector = ResultCollector()
    for analytic_name in selected_analytics:
        data = all_parameter_grids[slice_indices]
        result_collector.add_result(data,
                                    name=analytic_name,
                                    category=None,
                                    vis_func=Deterministic().plot,
                                    save_numerical_vis_data=False,
                                    vis_kwargs={'legend': list(sample_names),
                                                'out_type': 'png'},
                                    analytics_kwargs={'signal_idx': load_experiment_config_original(
                                        source_dir, target_purpose)['identities']})
        data_writer.output()
