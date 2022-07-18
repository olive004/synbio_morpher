

from copy import deepcopy
from functools import partial
import logging
import os
from typing import Union
import numpy as np
import pandas as pd

from src.srv.io.loaders.data_loader import DataLoader, GeneCircuitLoader
from src.srv.io.manage.script_manager import script_preamble
from src.srv.parameter_prediction.simulator import SIMULATOR_UNITS
from src.utils.misc.type_handling import flatten_listlike, merge_dicts
from src.utils.results.analytics.timeseries import Timeseries
from src.utils.results.experiments import Experiment, Protocol
from src.utils.results.result_writer import ResultWriter
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.io import get_pathnames, isolate_filename
from src.utils.misc.numerical import expand_matrix_triangle_idx, triangular_sequence
from src.utils.misc.scripts_io import get_search_dir, load_experiment_config_original
from src.utils.parameter_inference.interpolation_grid import create_parameter_range
from src.utils.results.results import ResultCollector
from src.utils.results.visualisation import VisODE


def main(config=None, data_writer=None):

    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        'scripts', 'parameter_grid_analysis', 'configs', 'heatmap_cfg.json'))

    # Load in parameter grid
    config_file = load_json_as_dict(config)
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
    shape_parameter_grid = np.shape(
        all_parameter_grids[list(all_parameter_grids.keys())[0]])
    num_species = shape_parameter_grid[0]
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

    # Helper funcs
    def convert_parameter_values_to_slice(start_value, end_value, step_size) -> Union[slice, tuple]:

        def parse_parameter_range_kwargs(start_value, end_value, original_parameter_range):
            if start_value is None:
                start_value = np.float64(0)
            if end_value is None:
                end_value = np.float64(original_parameter_range[-1])
            return np.float64(start_value), np.float64(end_value)

        def make_original_parameter_range(source_dir: str, target_purpose: str) -> np.ndarray:
            original_config = load_experiment_config_original(
                source_dir, target_purpose=target_purpose)
            return create_parameter_range(
                original_config[target_purpose])

        original_parameter_range = make_original_parameter_range(
            source_dir, target_purpose)
        start_value, end_value = parse_parameter_range_kwargs(start_value, end_value,
                                                              original_parameter_range)
        selected_parameter_range_idxs = np.arange(len(
            original_parameter_range))[(original_parameter_range >= start_value) == (original_parameter_range <= end_value)]
        if len(selected_parameter_range_idxs) > 1:
            return slice(
                selected_parameter_range_idxs[0], selected_parameter_range_idxs[-1]+1, step_size)
        else:
            # Keep the rest constant - choose constant value for these
            # --> determine the index corresponding to this value
            return tuple(selected_parameter_range_idxs)

    def make_species_interactions_index_map(num_species) -> dict:
        species_interactions_index_map = {}
        flat_triangle_size = triangular_sequence(num_species)
        for combination in range(flat_triangle_size):
            triangle_idx = expand_matrix_triangle_idx(combination)
            species_interactions_index_map[triangle_idx] = combination
            species_interactions_index_map[tuple(
                reversed(triangle_idx))] = combination
        return species_interactions_index_map

    def translate_species_idx(species_name):
        return sample_names.index(species_name)

    # Determine the indices to slice the parameter grid with
    def make_slice(selected_species_interactions, unselected_species_interactions,
                   slicing_configs, num_species, shape_parameter_grid) -> dict:

        # Choose which type of interactions to vary: self vs. inter
        # --> return dimension indices
        # --> get choice from config
        # --> might need lookup table

        def translate_species_interaction_to_idx(species_interaction: list) -> int:
            species_idxs = tuple(
                map(lambda x: translate_species_idx(x), species_interaction))
            species_interaction_idx = make_species_interactions_index_map(num_species)[
                species_idxs]
            return species_interaction_idx

        # Might want a slice window for the ones that are varying
        # --> make config accept "all" or named basis of starting and ending parameters
        def make_species_slice(selected_species: list, num_species: int) -> Union[tuple, slice]:
            if type(selected_species) == list:
                selected_species_idxs = sorted(
                    [translate_species_idx(s) for s in selected_species])
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
        def collect_parameter_slices(species_interactions: dict, parameter_choices_cfg_keyname: str) -> dict:
            species_idxs = {}
            for species_grouping, interacting_species_names in species_interactions.items():
                species_idxs[species_grouping] = translate_species_interaction_to_idx(
                    interacting_species_names)

            parameters_slices = make_parameter_slice(
                species_idxs, slicing_configs['interactions'][parameter_choices_cfg_keyname])
            return parameters_slices

        def make_grid_slice(species_slice, parameters_slices, shape_parameter_grid):
            grid_slice = [None] * len(shape_parameter_grid)
            grid_slice[0] = species_slice
            for grid_dimension, param_slice in parameters_slices.items():
                grid_slice[grid_dimension +
                           len(np.shape(species_slice))] = param_slice
            return grid_slice

        selected_parameters_slices = collect_parameter_slices(
            selected_species_interactions, parameter_choices_cfg_keyname='strengths')
        unselected_parameters_slices = collect_parameter_slices(unselected_species_interactions,
                                                                parameter_choices_cfg_keyname='non_varying_strengths')
        parameters_slices = merge_dicts(
            selected_parameters_slices, unselected_parameters_slices)
        species_slice = make_species_slice(
            slicing_configs['species_choices'], num_species)
        grid_slice = make_grid_slice(
            species_slice, parameters_slices, shape_parameter_grid)
        return tuple(grid_slice)

    # Convenience table
    def make_species_interaction_ref(species_interactions, parameter_config):
        all_refs = {}
        for grouping, species_interaction in species_interactions.items():
            species_interactions_ref = {}
            all_refs[tuple(species_interaction)] = species_interactions_ref
            species_interactions_ref['species_idxs'] = tuple(
                map(lambda x: translate_species_idx(x), species_interaction))
            species_interactions_ref['species_interaction_idx'] = make_species_interactions_index_map(num_species)[
                species_interactions_ref['species_idxs']]
            species_interactions_ref['interaction_params'] = parameter_config[grouping]
            parameter_creation_cfg = deepcopy(parameter_config[grouping])
            parameter_creation_cfg['step_size'] = load_experiment_config_original(
                source_dir, target_purpose=target_purpose)[target_purpose]['interaction_step_size']
            species_interactions_ref['parameter_range'] = create_parameter_range(
                parameter_creation_cfg)
            if type(parameter_config[grouping]) == dict:
                species_interactions_ref['interaction_slice'] = convert_parameter_values_to_slice(
                    **parameter_config[grouping])
            else:
                species_interactions_ref['interaction_slice'] = convert_parameter_values_to_slice(
                    start_value=parameter_config[grouping], end_value=parameter_config[grouping],
                    step_size=None)
        return all_refs

    def run_visualisation(all_parameter_grids, data_writer, selected_analytics,
                          selected_species_interactions, unselected_species_interactions,
                          slicing_configs, num_species, shape_parameter_grid):
        species_interaction_refs = make_species_interaction_ref(
            selected_species_interactions, slicing_configs['interactions']['strengths'])

        slice_indices = make_slice(selected_species_interactions, unselected_species_interactions,
                                   slicing_configs, num_species, shape_parameter_grid)
        info_text = '\n'.join(
            ['Held constant:'] + [f'{n}: {v}' for n, v in zip(
                unselected_species_interactions.values(), slicing_configs['interactions']['non_varying_strengths'].values())]
        )
        for analytic_name in selected_analytics:
            data = all_parameter_grids[analytic_name][slice_indices]
            result_collector = ResultCollector()
            for i, species_name in enumerate(slicing_configs['species_choices']):
                data_per_species = data[i]
                species_interaction_idxs = {
                    species_interaction_refs[k]['species_interaction_idx']: k for k in species_interaction_refs.keys()}
                sorted_species_interactions = [
                    species_interaction_idxs[k] for k in sorted(species_interaction_idxs.keys())]
                ind, cols = list(map(
                    lambda k: species_interaction_refs[k]['parameter_range'], sorted_species_interactions[:2]))
                data_container = pd.DataFrame(
                    data=np.squeeze(data_per_species),
                    index=ind,
                    columns=cols)
                result_collector.add_result(data_container,
                                            name=analytic_name,
                                            category=None,
                                            vis_func=VisODE(
                                                figsize=(14, 8)).heatmap,
                                            # vis_func=custom_3D_visualisation,
                                            save_numerical_vis_data=False,
                                            vis_kwargs={'legend': slicing_configs['species_choices'],
                                                        'out_type': 'png',
                                                        # '__setattr__': {'figsize': (10, 10)},
                                                        'xlabel': f'{sorted_species_interactions[0]} interaction strength '\
                                                        f'({SIMULATOR_UNITS["IntaRNA"]["energy"]})',
                                                        'ylabel': f'{sorted_species_interactions[1]} interaction strength '\
                                                        f'({SIMULATOR_UNITS["IntaRNA"]["energy"]})',
                                                        'title': f'{analytic_name.replace("_", " ")} for {sorted_species_interactions[0]} and {sorted_species_interactions[1]}',
                                                        'text': {'x': 12, 'y': 0.85, 's': info_text,
                                                                 'fontsize': 10,
                                                                 'bbox': dict(boxstyle='round', facecolor='wheat', alpha=1)},
                                                        'vmin': np.min(data_per_species),
                                                        'vmax': np.max(data_per_species)
                                                        # 'figure': {'figsize': (15, 15)}
                                                        })
            data_writer.write_results(result_collector.results, new_report=False,
                                      no_visualisations=False, only_numerical=False,
                                      no_analytics=True, no_numerical=True)

    experiment = Experiment(config_filepath=config, protocols=[
        Protocol(partial(run_visualisation,
                         all_parameter_grids=all_parameter_grids,
                         data_writer=data_writer,
                         selected_analytics=selected_analytics,
                         selected_species_interactions=selected_species_interactions,
                         unselected_species_interactions=unselected_species_interactions,
                         slicing_configs=slicing_configs,
                         num_species=num_species,
                         shape_parameter_grid=shape_parameter_grid))
    ], data_writer=data_writer)
    experiment.run_experiment()

    return config, data_writer
