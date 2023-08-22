
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
import logging
from fire import Fire
from multiprocessing import Process
import os
from typing import List
import numpy as np
from synbio_morpher.srv.io.manage.script_manager import script_preamble
from synbio_morpher.srv.io.manage.sys_interface import make_filename_safely
from synbio_morpher.srv.parameter_prediction.simulator import SIMULATOR_UNITS
from synbio_morpher.utils.circuit.agnostic_circuits.circuit_manager import construct_circuit_from_cfg, prepare_config
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


def create_circuits(config: dict, batch_size: int, batch_i: int, interaction_strengths: np.ndarray):
    
    interaction_strengths = create_parameter_range(
        config['parameter_based_simulation'])
    num_species = 3
    num_unique_interactions = np.sum(np.triu(num_species))
    analytic_types = get_analytics_types_all()
    size_interaction_array = np.size(interaction_strengths)
    
    all_analytic_matrices = define_matrices(
        num_species, size_interaction_array, num_unique_interactions, analytic_types)


    # Set loop vars
    total_iterations = np.power(
        size_interaction_array, num_unique_interactions)
    logging.info(total_iterations)
    logging.info(np.size(all_analytic_matrices) / len(analytic_types))
    logging.info(np.shape(all_analytic_matrices))
    num_iterations = int(total_iterations / batch_size)
    starting_iteration = int(num_iterations * batch_i)
    end_iteration = int(num_iterations * (batch_i + 1))
    
    def idx_sampler(i, size_interaction_array, unique_interaction):
        int(np.mod(i / np.power(size_interaction_array, unique_interaction), size_interaction_array))
        
    np.arange(num_unique_interactions)[:, None]
    np.arange(starting_iteration, end_iteration)[None, :]

    for i in range(starting_iteration, end_iteration):
        interaction_strength_choices = [int(np.mod(i / np.power(size_interaction_array, unique_interaction),
                                                size_interaction_array)) for unique_interaction in range(num_unique_interactions)]
        flat_triangle = interaction_strengths[list(interaction_strength_choices)]
        interaction_matrix = make_symmetrical_matrix_from_sequence(
            flat_triangle, num_species)
    
    


def main_subprocess(config, config_file, data_writer, sub_process, total_processes):
    debug_mode = False

    def make_interaction_interpolation_matrices():
        # Parameter space to scan
        # interaction_min = config_file['parameter_based_simulation']['interaction_min']
        # interaction_max = config_file['parameter_based_simulation']['interaction_max']
        # step_size = config_file['parameter_based_simulation']['step_size']
        interaction_strengths = create_parameter_range(
            config_file['parameter_based_simulation'])
        size_interaction_array = np.size(interaction_strengths)

        # Load data names
        def load_local_data(config_file):
            from synbio_morpher.utils.data.data_format_tools.manipulate_fasta import load_seq_from_FASTA
            sample_names = load_seq_from_FASTA(
                make_filename_safely(config_file.get("data_path")), as_type='dict')
            num_species = len(sample_names)
            num_unique_interactions = triangular_sequence(num_species)
            return num_species, num_unique_interactions

        analytic_types = get_analytics_types_all()
        num_species, num_unique_interactions = load_local_data(
            config_file)

        # Create matrices
        def define_matrices(num_species, size_interaction_array, num_unique_interactions, analytic_types) -> List[np.ndarray]:
            matrix_dimensions = tuple(
                [num_species] + [size_interaction_array]*num_unique_interactions)
            matrix_size = num_species * \
                np.power(size_interaction_array, num_unique_interactions)
            assert matrix_size == np.prod(list(
                matrix_dimensions)), 'Something is off about the intended size of the matrix'

            all_analytic_matrices = []
            for _analytic in analytic_types:
                all_analytic_matrices.append(np.zeros(
                    matrix_dimensions, dtype=np.float32))
            return all_analytic_matrices

        all_analytic_matrices = define_matrices(
            num_species, size_interaction_array, num_unique_interactions, analytic_types)

        # Set loop vars
        total_iterations = np.power(
            size_interaction_array, num_unique_interactions)
        logging.info(total_iterations)
        logging.info(np.size(all_analytic_matrices) / len(analytic_types))
        logging.info(np.shape(all_analytic_matrices))
        num_iterations = int(total_iterations / total_processes)
        starting_iteration = int(num_iterations * sub_process)
        end_iteration = int(num_iterations * (sub_process + 1))

        logging.info('-----------------------')
        logging.info(f'Total data: {total_iterations}')
        logging.info(f'Projected size (inc signal writing):')
        logging.info('\t12 * 108Mb = 1.3Gb')
        logging.info(
            f'\t{np.round(100.7/500*num_iterations/1000, decimals=3)}Gb')
        modeller = CircuitModeller(result_writer=data_writer)

        """ Slow: 0.2938s with 1 subprocess and signal length = 15000. """
        """ Slow: 0.2343s with 1 subprocess and signal length = 12000. """
        """ Slow: 0.2264s with 1 subprocess and signal length = 12000 with use_old_steadystates=True. """
        """ ????: 0.5561s medium cfg with 12 subprocess and with use_old_steadystates=True. """
        """ ????: 0.5613s logscale cfg with 12 subprocess and with use_old_steadystates=True. """

        def loop_modelling(i):
            interaction_strength_choices = [int(np.mod(i / np.power(size_interaction_array, unique_interaction),
                                                       size_interaction_array)) for unique_interaction in range(num_unique_interactions)]
            flat_triangle = interaction_strengths[list(
                interaction_strength_choices)]
            interaction_matrix = make_symmetrical_matrix_from_sequence(
                flat_triangle, num_species)
            cfg = {"interactions": {
                "interactions_matrix": interaction_matrix,
                "interactions_units": SIMULATOR_UNITS['IntaRNA']['rate']}
            }

            circuit = construct_circuit_from_cfg(
                extra_configs=cfg, config_file=config_file)
            circuit = modeller.init_circuit(circuit)
            circuit = modeller.simulate_signal(
                circuit, use_solver=config_file.get('signal').get('use_solver', 'naive'), use_old_steadystates=True)

            idxs = [slice(0, num_species)] + [[strength_idx]
                                              for strength_idx in interaction_strength_choices]

            for j, analytic in enumerate(analytic_types):
                all_analytic_matrices[j][tuple(
                    idxs)] = circuit.result_collector.results['signal'].analytics.get(analytic)

            if debug_mode and i == 0:
                modeller.write_results(circuit)

            # @time_it
            if np.mod(i, 100) == 0:
                write_all()

        def write_all(out_type='npy'):
            for m, analytic in enumerate(analytic_types):
                data_writer.output(out_type, out_name=analytic,
                                   data=all_analytic_matrices[m].astype(np.float32), overwrite=True,
                                   write_to_top_dir=True)

        # Main loop
        for i in range(starting_iteration, end_iteration):
            if np.mod(i, 1000) == 0 or i == starting_iteration:
                # data_writer.unsubdivide()
                # data_writer.subdivide_writing(
                #     f'{i}-{+1000, 1000)-1}')
                logging.info(
                    f'Iteration {i}/{total_iterations}, stopping at {end_iteration}')
            # data_writer.subdivide_writing(str(i), safe_dir_change=False)

            loop_modelling(i)
            # data_writer.unsubdivide_last_dir()

        logging.info('Finished: outputting final matrices')
        write_all()

    experiment = Experiment(config=config, config_file=config_file,
                            protocols=[
                                Protocol(make_interaction_interpolation_matrices)],
                            data_writer=data_writer)
    experiment.run_experiment()
    return config, data_writer


def main(config=None, data_writer=None):

    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        'synbio_morpher', 'scripts', 'parameter_based_simulation', 'configs', 'testing.json'))
    config_file = load_json_as_dict(config)
    config_file = prepare_config(config_file)

    if config_file.get('experiment').get('parallelise'):
        num_subprocesses = config_file.get(
            'experiment').get('num_subprocesses', 1)
    else:
        num_subprocesses = 1

    for subprocess in range(num_subprocesses):
        if num_subprocesses != 1:
            data_writer.update_ensemble('subprocess_' + str(subprocess+1))
        p = Process(target=main_subprocess, args=(
            config, config_file, data_writer, subprocess, num_subprocesses))
        p.start()
        
        
if __name__ == "__main__":
    Fire(main)
