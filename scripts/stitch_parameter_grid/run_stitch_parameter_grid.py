

from copy import deepcopy
from functools import reduce
import logging
import os

import numpy as np
from src.srv.io.loaders.data_loader import DataLoader
from src.srv.io.results.result_writer import ResultWriter
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.io import get_pathnames, isolate_filename
from src.utils.misc.numerical import triangular_sequence
from src.utils.misc.scripts_io import get_search_dir, get_subprocesses_dirnames, load_experiment_config
from src.utils.misc.string_handling import sort_by_ordinal_number


def main(config=None, data_writer=None):

    if config is None:
        config = os.path.join(
            'scripts', 'stitch_parameter_grid', 'configs', 'base_config.json')
    config_file = load_json_as_dict(config)
    if data_writer is None:
        data_writer = ResultWriter(purpose=config_file[
            'experiment'].get('purpose', 'stitch_parameter_grid'))

    # Load in parameter grids
    config_file, source_dir = get_search_dir(
        'source_parameter_dir', config_file=config_file)
    experiment_config = load_experiment_config(source_dir)
    experiment_settings = experiment_config['experiment']
    num_subprocesses = 1
    if experiment_settings['parallelise']:
        num_subprocesses = experiment_settings['num_subprocesses']

    # If there was multithreading, load each parameter_grid one by one from subfolders
    def load_parameter_grids():
        all_parameter_grids = {}
        subprocess_dirs = get_subprocesses_dirnames(source_dir)
        for subprocess_dir in sort_by_ordinal_number(subprocess_dirs):
            # logging.info(os.path.basename(subprocess_dir))
            parameter_grids = get_pathnames(subprocess_dir, 'npy')
            for parameter_grid in parameter_grids:
                analytic_name = isolate_filename(parameter_grid)
                if analytic_name not in all_parameter_grids.keys():
                    all_parameter_grids[analytic_name] = []
                all_parameter_grids[analytic_name].append(
                    DataLoader().load_data(parameter_grid))
        return all_parameter_grids
    all_parameter_grids = load_parameter_grids()

    # Stitch grids together
    def stitch_grids():
        # Find the starting and ending indices

        a_parameter_grid = all_parameter_grids[list(
            all_parameter_grids.keys())[0]][0]
        matrix_size = np.size(a_parameter_grid)
        num_species = np.shape(a_parameter_grid)[0]
        num_unique_interactions = triangular_sequence(num_species)
        # interaction_strengths = np.arange(np.shape(a_parameter_grid)[-1])
        # size_interaction_array = np.size(interaction_strengths)
        size_interaction_array = np.shape(a_parameter_grid)[-1]

        logging.info(matrix_size)
        logging.info(np.shape(a_parameter_grid))
        # logging.info(a_parameter_grid[1])

        total_iterations = np.power(
            size_interaction_array, num_unique_interactions)
        total_iterations = matrix_size
        num_iterations = int(total_iterations / num_subprocesses)
        logging.info(num_iterations)

        stitched_parameter_grids = {k: np.zeros(
            np.shape(a_parameter_grid)) for k in all_parameter_grids.keys()}
        logging.info(
            f'Number of zeros before stitching: {np.sum(stitched_parameter_grids[list(stitched_parameter_grids.keys())[0]] == 0)}')
        for analytic_name in stitched_parameter_grids.keys():
            logging.info('\n\n')
            logging.info(analytic_name)
            for i in range(num_subprocesses):
                start_idx = num_iterations*i
                end_idx = num_iterations*(i+1)-1
                # Fixing my original error in dimensionality specification in the original parameter_based_simulation script
                end_idx = np.min(num_iterations*(i+1)-1,
                                 int(total_iterations / num_species))

                start_indices = [int(np.mod(start_idx / np.power(size_interaction_array, unique_interaction),
                                            size_interaction_array)) for unique_interaction in range(num_unique_interactions)]
                end_indices = [int(np.mod(end_idx / np.power(size_interaction_array, unique_interaction),
                                          size_interaction_array)) + 1 for unique_interaction in range(num_unique_interactions)]
                idxs = [slice(0, num_species)] + [slice(s, e)
                                                  for s, e in zip(start_indices, end_indices)]
                logging.info(start_idx)
                logging.info(end_idx)
                logging.info(start_indices)
                logging.info(end_indices)
                logging.info(idxs)

                logging.info(
                    f'Expected size of slice: {end_idx+1 - start_idx}')
                logging.info(
                    f'Projected size of slice: {reduce(lambda x, y: x*y, np.subtract(end_indices, start_indices)) }')
                # for analytic_name in stitched_parameter_grids.keys():
                stitched_parameter_grids[analytic_name][tuple(
                    idxs)] = all_parameter_grids[analytic_name][i][tuple(idxs)]
                logging.info(
                    f'Number of zeros in subprocess matrix: {np.sum(all_parameter_grids[analytic_name][i] == 0)}')
                logging.info(
                    f'Number of zeros in stiched matrix: {np.sum(stitched_parameter_grids[analytic_name] == 0)}')
                logging.info(
                    f'Difference in zeros: {np.absolute(np.sum(all_parameter_grids[analytic_name][i] == 0) - np.sum(stitched_parameter_grids[analytic_name] == 0))}')

                # Fixing og error
                if end_idx == total_iterations / num_species:
                    break
            break
        return stitched_parameter_grids

    stitched_parameter_grids = stitch_grids()

    # for analytic_name in stitched_parameter_grids.keys():
    #     logging.info(analytic_name)
    #     logging.info(np.sum(stitched_parameter_grids[analytic_name] == 0))

    # interaction_strength_choices = []
    # counts = np.array([0] * num_unique_interactions)
    # prev_counts = np.array([0] * num_unique_interactions)
    # for i in range(0*num_iterations, num_subprocesses * num_iterations):
    #     interaction_strength_choice = [int(np.mod(i / np.power(size_interaction_array, unique_interaction),
    #                                               size_interaction_array)) for unique_interaction in range(num_unique_interactions)]
    #     interaction_strength_choices.append(tuple(interaction_strength_choice))
    #     # logging.info(list(interaction_strength_choice))
    #     current_counts = (np.array(interaction_strength_choice) == size_interaction_array-1).astype(int)
    #     flagged_counts = current_counts * (prev_counts == 0).astype(int)
    #     prev_counts = deepcopy(current_counts)
    #     counts += flagged_counts
    #     # if i == num_iterations -1:
    #     #     remainders = [int(np.mod(strength, size_interaction_array-1)) for strength in list(interaction_strength_choice)]
    #     #     counts += remainders
    # logging.info(len(interaction_strength_choices))
    # logging.info(len(set(interaction_strength_choices)))
    # logging.info(counts)

    #     interaction_strength_choices = [int(np.mod(i / np.power(size_interaction_array, unique_interaction),
    #                                                size_interaction_array)) for unique_interaction in range(num_unique_interactions)]
    #     idxs = [slice(0, num_species)] + [[strength_idx]
    #                                       for strength_idx in interaction_strength_choices]

    #     logging.info(a_parameter_grid[0:3, 0, 0, 0, 0, 0, 0])
    # logging.info(np.all(a_parameter_grid[0:3][:num_iterations]))

    # If there was multithreading, indices will be different

    # Write full matrices
