

from functools import reduce
import logging
import os
from typing import Union

import numpy as np
from src.srv.io.loaders.data_loader import DataLoader, GeneCircuitLoader
from src.utils.misc.type_handling import flatten_listlike, merge_dicts
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
        config_search_key='source_parameter_dir', config_file=config_file)

    def load_parameter_grids():
        all_parameter_grids = {}
        parameter_grids = get_pathnames(source_dir, 'npy')
        for parameter_grid in parameter_grids:
            all_parameter_grids[isolate_filename(
                parameter_grid)] = DataLoader().load_data(parameter_grid)
        return all_parameter_grids

    def get_sample_names(source_dir, target_purpose):
        circuit_filepath = load_experiment_config_original(
            source_dir, target_purpose)['data_path']
        data = GeneCircuitLoader().load_data(circuit_filepath)
        return data.sample_names

    target_purpose = 'parameter_based_simulation'
    all_parameter_grids = load_parameter_grids()
    num_species = np.shape(
        all_parameter_grids[list(all_parameter_grids.keys())[0]])[0]
    sample_names = get_sample_names(
        source_dir, target_purpose)
    assert num_species == len(
        sample_names), f'Number of species in parameter grid ({num_species}) does ' \
        f'not equal number of species in original configuration file {sample_names}.'

    # Get relevant configs
    slicing_configs = config_file['slicing']

    selected_species_interactions = slicing_configs['interactions']['interacting_species']
    unselected_species_interactions = slicing_configs['interactions']['non_varying_species_interactions']

    selected_analytics = slicing_configs['analytics']['names']
    if selected_analytics is None:
        selected_analytics = Timeseries(None).get_analytics_types()

    def validate_species_cfgs(*cfg_species_lists: list):
        def validate_each(species_name):
            return species_name in sample_names
        for cfg_species_list in cfg_species_lists:
            assert all([validate_each(species_name) for species_name in cfg_species_list]), \
                f'Species {cfg_species_list} from config were not found in list ' \
                f'of species from circuit {sample_names}'

    validate_species_cfgs(flatten_listlike(list(selected_species_interactions.values())),
                          flatten_listlike(list(unselected_species_interactions.values())))

    # Determine the indices to slice the parameter grid with
    def make_slice():

        # Choose which type of interactions to vary: self vs. inter
        # --> return dimension indices
        # --> get choice from config
        # --> might need lookup table

        def make_species_interactions_index_map(num_species) -> dict:
            species_interactions_index_map = {}
            flat_triangle_size = triangular_sequence(num_species)
            for combination in range(flat_triangle_size):
                logging.info(expand_matrix_triangle_idx(
                    combination))
                species_interactions_index_map[expand_matrix_triangle_idx(
                    combination)] = combination
            return species_interactions_index_map

        def translate_species_interaction_to_idx(species_interaction: list) -> int:
            logging.info(species_interaction)
            logging.info(make_species_interactions_index_map(num_species))
            logging.info(sample_names.index(species_interaction[0]))
            species_idxs = tuple(
                map(lambda x: sample_names.index(x), species_interaction))
            logging.info(species_idxs)
            species_interaction_idx = make_species_interactions_index_map(num_species)[
                species_idxs]
            return species_interaction_idx

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
                selected_species_idxs = sorted(reduce((lambda x: translate_species_interaction_to_idx(x)),
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

                def parse_parameter_range_kwargs(start_value, end_value, step_size, original_parameter_range):
                    if start_value is None:
                        start_value = 0
                    if end_value is None:
                        end_value = len(original_parameter_range)
                    if step_size is None:
                        step_size = 1
                    return start_value, end_value, step_size

                def make_original_parameter_range(source_dir: str, target_purpose: str) -> np.ndarray:
                    original_config = load_experiment_config_original(
                        source_dir, target_purpose=target_purpose)
                    return create_parameter_range(
                        original_config[target_purpose])

                original_parameter_range = make_original_parameter_range(
                    source_dir, target_purpose)
                start_value, end_value, step_size = parse_parameter_range_kwargs(start_value, end_value,
                                                                                 step_size, original_parameter_range)
                selected_parameter_range_idxs = np.arange(len(
                    original_parameter_range))[original_parameter_range >= start_value]
                #    original_parameter_range <= end_value]
                return slice(
                    selected_parameter_range_idxs[0], selected_parameter_range_idxs[-1], step_size)

            slice_idx_map = {}
            for grouping, interaction_idx in interacting_species_idxs.items():
                if type(parameter_config[grouping]) == dict:
                    slice_idx_map[interaction_idx] = convert_parameter_values_to_slice(
                        **parameter_config[grouping])
                else:
                    slice_idx_map[interaction_idx] = convert_parameter_values_to_slice(
                        start_value=parameter_config[grouping], end_value=parameter_config[grouping],
                        step_size=None)

            return slice_idx_map

        # Convert the names of the interacting species to the index of that interacting pair in the interpolation grid
        def collect_parameter_slices(species_interactions: dict, parameter_cfg_keyname: str) -> dict:
            species_idxs = {}
            logging.info(species_interactions)
            for species_group, interacting_species_names in species_interactions.items():
                # interacting_species_names
                # interacting_species_idxs_square_matrix
                # interacting_species_idxs_symmetrical_matrix
                # interacting_species_idxs_flat_symmetrical_matrix
                logging.info(species_group)
                logging.info(interacting_species_names)
                logging.info(reduce_interacting_species_idx(
                    translate_species_interaction_to_idx(interacting_species_names)))
                species_idxs[species_group] = reduce_interacting_species_idx(
                    # reduce((lambda x: translate_species_to_idx(x)), interacting_species_names))
                    translate_species_interaction_to_idx(interacting_species_names))

            # species_slice = make_species_slice(selected_species, num_species)
            parameters_slices = make_parameter_slice(
                species_idxs, slicing_configs['interactions'][parameter_cfg_keyname])
            return parameters_slices

        selected_parameters_slices = collect_parameter_slices(
            selected_species_interactions, parameter_cfg_keyname='strengths')
        unselected_parameters_slices = collect_parameter_slices(unselected_species_interactions,
                                                                parameter_cfg_keyname='non_varying_strengths')
        logging.info(selected_parameters_slices)
        logging.info(unselected_parameters_slices)
        parameters_slices = merge_dicts(
            selected_parameters_slices, unselected_parameters_slices)
        species_slice = make_species_slice(
            slicing_configs['species_choices'], num_species)
        grid_slice = [species_slice] + \
            [p_slice for p_slice in parameters_slices]
        logging.info(parameters_slices)
        logging.info(species_slice)
        logging.info(grid_slice)
        return tuple(grid_slice)

    # Make visualisations for each analytic chosen
    slice_indices = make_slice()
    result_collector = ResultCollector()
    logging.info(slice_indices)
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
