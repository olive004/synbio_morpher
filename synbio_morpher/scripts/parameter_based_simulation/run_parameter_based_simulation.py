
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from fire import Fire
from multiprocessing import Process
from functools import partial
import jax
import jax.numpy as jnp
import os
from typing import List
import numpy as np
from synbio_morpher.srv.io.manage.script_manager import script_preamble
from synbio_morpher.srv.io.manage.sys_interface import make_filename_safely
from synbio_morpher.srv.parameter_prediction.simulator import SIMULATOR_UNITS
from synbio_morpher.utils.common.setup import construct_circuit_from_cfg, prepare_config
from synbio_morpher.utils.results.analytics.naming import get_analytics_types_all
from synbio_morpher.utils.results.experiments import Experiment, Protocol
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
from synbio_morpher.utils.misc.numerical import make_symmetrical_matrix_from_sequence, triangular_sequence
from synbio_morpher.utils.parameter_inference.interpolation_grid import create_parameter_range
from synbio_morpher.utils.circuit.agnostic_circuits.circuit_manager import CircuitModeller


# Create matrices
def define_matrices(num_species, size_interaction_array, num_unique_interactions, analytic_types) -> List[np.ndarray]:
    matrix_dimensions = tuple(
        [len(analytic_types), num_species] + [size_interaction_array]*num_unique_interactions)
    matrix_size = num_species * \
        np.power(size_interaction_array, num_unique_interactions)
    assert matrix_size == np.prod(list(
        matrix_dimensions)), 'Something is off about the intended size of the matrix'

    all_analytic_matrices = np.zeros(matrix_dimensions, dtype=np.float32)
    return all_analytic_matrices


def make_interaction_matrices(num_species: int,
                              interaction_strengths: np.ndarray,
                              num_unique_interactions: int,
                              starting_iteration: int,
                              end_iteration: int):

    def idx_sampler(i, size_interaction_array, unique_interaction):
        return jnp.mod(i / jnp.power(size_interaction_array, unique_interaction), size_interaction_array).astype(int)

    unique_interaction = np.arange(num_unique_interactions)
    idx_sampler = partial(idx_sampler, size_interaction_array=np.size(
        interaction_strengths), unique_interaction=unique_interaction)

    idxs = np.arange(starting_iteration, end_iteration)[:, None]
    interaction_strength_choices = jax.vmap(idx_sampler)(idxs)
    flat_triangle = interaction_strengths[list(interaction_strength_choices)]
    interaction_matrices = jax.vmap(
        partial(make_symmetrical_matrix_from_sequence, side_length=num_species))(flat_triangle),
    return interaction_matrices, interaction_strength_choices


def create_circuits(config: dict, interaction_matrices: np.ndarray):
    circuits = [None] * len(interaction_matrices)
    for i, interaction_matrix in enumerate(interaction_matrices):
        cfg = {"interactions": {
            "interactions_matrix": interaction_matrix,
            "interactions_units": SIMULATOR_UNITS['IntaRNA']['rate']}
        }
        circuits[i] = construct_circuit_from_cfg(
            extra_configs=cfg, config_file=config)
    return circuits


def write_all(all_analytic_matrices, analytic_types, data_writer, out_type='npy'):
    for m, analytic in enumerate(analytic_types):
        data_writer.output(out_type, out_name=analytic,
                           data=all_analytic_matrices[m].astype(np.float32), overwrite=True,
                           write_to_top_dir=True)


def simulate(circuits: list, config: dict, data_writer):
    modeller = CircuitModeller(result_writer=data_writer, config=config)
    methods = {
        'init_circuits': {},
        'simulate_signal_batch': {},
    }
    if config['debug_mode']:
        methods['write_results'] = {
            'no_visualisations': False,
            'no_numerical': False}

    return modeller.batch_circuits(
        circuits=circuits,
        methods=methods
    )


def run_batch(batch_i: int,
              config: dict,
              data_writer,
              interaction_strengths: np.ndarray,
              num_iterations: int,
              num_species: int,
              num_unique_interactions: int,
              ):
    starting_iteration = int(num_iterations * batch_i)
    end_iteration = int(num_iterations * (batch_i + 1))

    interaction_matrices, interaction_strength_choices = make_interaction_matrices(
        num_species, interaction_strengths, num_unique_interactions, starting_iteration, end_iteration)

    circuits = create_circuits(config, interaction_matrices)
    circuits = simulate(circuits, config, data_writer)
    return circuits, interaction_strength_choices


def main(config: dict, data_writer, batch_size: int):

    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        'synbio_morpher', 'scripts', 'parameter_based_simulation', 'configs', 'base_config.json'))
    config = load_json_as_dict(config)
    config = prepare_config(config)

    def core_func(config):
        interaction_strengths = create_parameter_range(
            config['parameter_based_simulation'])

        num_species = config['parameter_based_simulation'].get('num_species', 3)
        num_unique_interactions = np.sum(np.triu(num_species))
        analytic_types = get_analytics_types_all()

        all_analytic_matrices = define_matrices(num_species=num_species,
                                                size_interaction_array=np.size(interaction_strengths), 
                                                num_unique_interactions=num_unique_interactions, 
                                                analytic_types=analytic_types)

        # Set loop vars
        total_iterations = np.power(
            np.size(interaction_strengths), num_unique_interactions)
        num_iterations = int(total_iterations / batch_size)
        
        logging.info(total_iterations)
        logging.info(np.size(all_analytic_matrices) / len(analytic_types))
        logging.info(np.shape(all_analytic_matrices))

        for batch_i in range(batch_size):
            circuits, all_interaction_strength_choices = run_batch(batch_i=batch_i,
                                                                config=config,
                                                                data_writer=data_writer,
                                                                interaction_strengths=interaction_strengths,
                                                                num_iterations=num_iterations,
                                                                num_species=num_species,
                                                                num_unique_interactions=num_unique_interactions)
            for circuit, interaction_strength_choices in zip(circuits, all_interaction_strength_choices):
                idxs = [slice(0, num_species)] + [[strength_idx]
                                                for strength_idx in interaction_strength_choices]

                for j, analytic in enumerate(analytic_types):
                    all_analytic_matrices[j][tuple(
                        idxs)] = circuit.result_collector.results['signal'].analytics.get(analytic)

                # @time_it
                # if np.mod(i, 100) == 0:
                write_all(all_analytic_matrices, analytic_types, data_writer)
                
    
    experiment = Experiment(config=config,
                            protocols=[Protocol(partial(core_func, config=config))],
                            data_writer=data_writer)
    experiment.run_experiment()
    return config, data_writer


if __name__ == "__main__":
    Fire(main)
