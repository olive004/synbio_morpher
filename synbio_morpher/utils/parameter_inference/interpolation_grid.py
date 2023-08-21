
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    


import logging
from typing import Union
import numpy as np

from synbio_morpher.utils.misc.numerical import calculate_num_decimals, expand_matrix_triangle_idx, triangular_sequence
from synbio_morpher.utils.misc.type_handling import merge_dicts


def parameter_range_creation(range_min, range_max, range_step, is_logscale=False) -> np.ndarray:
    """ Rounding numbers created with arange to nearest decimal of the range_step 
    to avoid numerical errors downstream """
    assert type(
        range_step) is not None, f'The step size {range_step} should have a numerical value.'
    if not is_logscale:
        parameter_range = np.arange(
            range_min, range_max, range_step).astype(np.float64)
        return np.around(parameter_range, np.power(10, calculate_num_decimals(range_step)-1))
    else:
        num_parameters = int(np.ceil((range_max - range_min) / range_step))
        log_scale = np.logspace(range_min, range_max, num=num_parameters)
        return np.interp(log_scale, (log_scale.min(), log_scale.max()), (range_min, range_max)).astype(np.float64)


def create_parameter_range(range_configs: dict) -> np.ndarray:
    """ Creates parameter range 1D """
    logging.info('create_parameter_range')
    logging.info(range_configs)
    min_key = [k for k in range_configs.keys(
    ) if 'min' in k or 'start' in k][0]
    max_key = [k for k in range_configs.keys() if 'max' in k or 'end' in k][0]
    step_key = [k for k in range_configs.keys() if 'step' in k][0]
    is_logscale = range_configs.get('log_scale', False)
    return parameter_range_creation(
        range_configs[min_key], range_configs[max_key], range_configs[step_key], is_logscale=is_logscale)


# Helper funcs
def convert_parameter_values_to_slice(start_value, end_value, step_size, original_config) -> Union[slice, tuple]:

    def parse_parameter_range_kwargs(start_value, end_value, original_parameter_range):
        if start_value is None:
            start_value = np.float64(0)
        if end_value is None:
            end_value = np.float64(original_parameter_range[-1])
        return np.float64(start_value), np.float64(end_value)

    original_parameter_range = create_parameter_range(
        original_config['parameter_based_simulation'])
    start_value, end_value = parse_parameter_range_kwargs(start_value, end_value,
                                                            original_parameter_range)
    selected_parameter_range_idxs = np.arange(len(
        original_parameter_range))[(original_parameter_range >= start_value) == (original_parameter_range <= end_value)]
    if len(selected_parameter_range_idxs) > 1:
        return slice(
            selected_parameter_range_idxs[0], selected_parameter_range_idxs[-1]+1) #, step_size)
    else:
        # Keep the rest constant - choose constant value for these
        # --> determine the index corresponding to this value
        return tuple(selected_parameter_range_idxs)


# Convenience table
def make_species_interaction_summary(species_interactions, strength_config, original_config, sample_names):
    all_refs = {}
    for grouping, species_interaction in species_interactions.items():
        species_interactions_ref = {}

        species_interactions_ref['species_idxs'] = tuple(
            map(lambda x: translate_species_idx(x, sample_names), species_interaction))
        species_interactions_ref['species_interaction_idx'] = make_species_interactions_index_map(len(sample_names))[
            species_interactions_ref['species_idxs']]
        species_interactions_ref['interaction_params'] = strength_config[grouping]
        parameter_creation_cfg = strength_config[grouping]

        key_translation = {
            'start_value': 'interaction_min',
            'end_value': 'interaction_max'
        }
        for k, v in parameter_creation_cfg.items():
            if v is None:
                parameter_creation_cfg[k] = original_config['parameter_based_simulation'][key_translation.get(
                    k, k)]
        species_interactions_ref['parameter_range'] = create_parameter_range(
            parameter_creation_cfg)

        if type(strength_config[grouping]) == dict:
            species_interactions_ref['interaction_slice'] = convert_parameter_values_to_slice(
                original_config=original_config, **strength_config[grouping])
        else:
            species_interactions_ref['interaction_slice'] = convert_parameter_values_to_slice(
                start_value=strength_config[grouping], end_value=strength_config[grouping],
                step_size=None, original_config=original_config)
        all_refs[tuple(species_interaction)] = species_interactions_ref
    return all_refs


def make_species_interactions_index_map(num_species) -> dict:
    species_interactions_index_map = {}
    flat_triangle_size = triangular_sequence(num_species)
    for combination in range(flat_triangle_size):
        triangle_idx = expand_matrix_triangle_idx(combination)
        species_interactions_index_map[triangle_idx] = combination
        species_interactions_index_map[tuple(
            reversed(triangle_idx))] = combination
    return species_interactions_index_map


def translate_species_idx(species_name: str, sample_names: list):
    return sample_names.index(species_name)


# Determine the indices to slice the parameter grid with
def make_slice(selected_species_interactions, unselected_species_interactions,
                slicing_configs, shape_parameter_grid, sample_names: list, original_config: dict) -> dict:

    # Choose which type of interactions to vary: self vs. inter
    # --> return dimension indices
    # --> get choice from config
    # --> might need lookup table

    def translate_species_interaction_to_idx(species_interaction: list, sample_names: list) -> int:
        species_idxs = tuple(
            map(lambda x: translate_species_idx(x, sample_names), species_interaction))
        species_interaction_idx = make_species_interactions_index_map(len(sample_names))[
            species_idxs]
        return species_interaction_idx

    # Might want a slice window for the ones that are varying
    # --> make config accept "all" or named basis of starting and ending parameters
    def make_species_slice(selected_species: list, sample_names: list) -> Union[tuple, slice]:
        if type(selected_species) == list:
            selected_species_idxs = sorted(
                [translate_species_idx(s, sample_names) for s in selected_species])
            if len(selected_species_idxs) == len(sample_names):
                selected_species = 'all'
            else:
                return tuple(selected_species_idxs)
        if selected_species == 'all':
            return slice(0, len(sample_names))
        else:
            raise ValueError(
                'Please input the species you want to include in the visualisations')

    def make_parameter_slice(interacting_species_idxs, parameter_config, original_config) -> dict:

        slice_idx_map = {}
        for grouping, interaction_idx in interacting_species_idxs.items():
            if type(parameter_config[grouping]) == dict:
                slice_idx_map[interaction_idx] = convert_parameter_values_to_slice(
                    original_config=original_config, **parameter_config[grouping])
            else:
                slice_idx_map[interaction_idx] = convert_parameter_values_to_slice(
                    start_value=parameter_config[grouping], end_value=parameter_config[grouping],
                    step_size=None, original_config=original_config)

        logging.info(parameter_config)

        return slice_idx_map

    # Convert the names of the interacting species to the index of that interacting pair in the interpolation grid
    def collect_parameter_slices(species_interactions: dict, parameter_choice_cfg_keyname: str, original_config: dict, sample_names: list) -> dict:
        species_idxs = {}
        for species_grouping, interacting_species_names in species_interactions.items():
            species_idxs[species_grouping] = translate_species_interaction_to_idx(
                interacting_species_names, sample_names)

        parameters_slices = make_parameter_slice(
            species_idxs, slicing_configs['interactions'][parameter_choice_cfg_keyname], original_config)
        return parameters_slices

    def make_grid_slice(species_slice, parameters_slices, shape_parameter_grid):
        grid_slice = [None] * len(shape_parameter_grid)
        grid_slice[0] = species_slice
        for grid_dimension, param_slice in parameters_slices.items():
            grid_slice[grid_dimension +
                        len(np.shape(species_slice))] = param_slice
        return grid_slice

    selected_parameters_slices = collect_parameter_slices(
        selected_species_interactions, parameter_choice_cfg_keyname='strengths', 
        original_config=original_config, sample_names=sample_names)
    unselected_parameters_slices = collect_parameter_slices(unselected_species_interactions,
                                                            parameter_choice_cfg_keyname='non_varying_strengths', 
                                                            original_config=original_config, sample_names=sample_names)
    parameters_slices = merge_dicts(
        selected_parameters_slices, unselected_parameters_slices)
    species_slice = make_species_slice(
        slicing_configs['species_choices'], sample_names=sample_names)
    grid_slice = make_grid_slice(
        species_slice, parameters_slices, shape_parameter_grid)
    return tuple(grid_slice)