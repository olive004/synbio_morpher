import logging
from multiprocessing import Process
import os
import numpy as np
from scripts.common.circuit import construct_circuit_from_cfg
from src.srv.io.manage.sys_interface import make_filename_safely
from src.srv.io.results.analytics.timeseries import Timeseries
from src.srv.io.results.experiments import Experiment, Protocol
from src.srv.io.results.result_writer import ResultWriter
from src.srv.parameter_prediction.simulator import SIMULATOR_UNITS
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.decorators import time_it
from src.utils.misc.numerical import make_symmetrical_matrix_from_sequence, triangular_sequence
from src.utils.parameter_inference.interpolation_grid import create_parameter_range
from src.utils.system_definition.agnostic_system.system_manager import CircuitModeller


def main(config=None, data_writer=None):

    if config is None:
        config = os.path.join(
            'scripts', 'parameter_based_simulation', 'configs', 'medium_parameter_space.json')
    config_file = load_json_as_dict(config)
    if data_writer is None:
        data_writer = ResultWriter(purpose=config_file.get(
            'experiment').get('purpose', 'parameter_based_simulation'))

    if config_file.get('experiment').get('parallelise'):
        num_subprocesses = config_file.get(
            'experiment').get('num_subprocesses', 1)
    else:
        num_subprocesses = 1

    for subprocess in range(num_subprocesses):
        data_writer.update_ensemble('subprocess_' + str(subprocess+1))
        p = Process(target=main_subprocess, args=(
            config, data_writer, subprocess, num_subprocesses))
        p.start()
    # p.join()


def main_subprocess(config, data_writer, sub_process, total_processes):
    config_file = load_json_as_dict(config)

    def make_interaction_interpolation_matrices():
        # Parameter space to scan
        interaction_min = config_file['parameter_based_simulation']['interaction_min']
        interaction_max = config_file['parameter_based_simulation']['interaction_max']
        interaction_step_size = config_file['parameter_based_simulation']['interaction_step_size']
        interaction_strengths = create_parameter_range(config_file['parameter_based_simulation'])
        size_interaction_array = np.size(interaction_strengths)

        # Load data names
        from src.utils.data.data_format_tools.manipulate_fasta import load_seq_from_FASTA
        sample_names = load_seq_from_FASTA(
            make_filename_safely(config_file.get("data_path")), as_type='dict')
        num_species = len(sample_names)
        num_unique_interactions = triangular_sequence(num_species)

        # Create matrices
        matrix_dimensions = tuple(
            [num_species] + [size_interaction_array]*num_unique_interactions)
        matrix_size = num_species * \
            np.power(size_interaction_array, num_unique_interactions)
        assert matrix_size == np.prod(list(
            matrix_dimensions)), 'Something is off about the intended size of the matrix'

        all_analytic_matrices = []
        analytic_types = Timeseries(None).get_analytics_types()
        for analytic in analytic_types:
            all_analytic_matrices.append(np.zeros(
                matrix_dimensions, dtype=np.float32))

        # Set loop vars
        total_iterations = np.power(size_interaction_array, num_unique_interactions)
        num_iterations = int(total_iterations / total_processes)
        starting_iteration = int(num_iterations * sub_process)
        end_iteration = int(num_iterations * (sub_process + 1))

        logging.info('-----------------------')
        # logging.info('Rate: ca. 8000 / min')
        # logging.info('Total estimated time (steady state):')
        # logging.info(f'\t{193000/(13*60)} or {10000/24} in mins')
        # logging.info(f'\t{193000/(13*60)/60} or {10000/24/60} in hours')
        # logging.info(f'\t{193000/(13*60)/60 /24} or {10000/24/60/24} in days')
        logging.info(f'Total data: {total_iterations}')
        logging.info(f'Projected size (inc signal writing):')
        logging.info('\t12 * 108Mb = 1.3Gb')
        logging.info(
            f'\t{np.round(100.7/500*num_iterations/1000, decimals=3)}Gb')
        modeller = CircuitModeller(result_writer=data_writer)

        # @time_it
        def loop_iter(i):
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
                circuit, use_solver=config_file.get('signal').get('use_solver', 'naive'))

            idxs = [slice(0, num_species)] + [[strength_idx]
                                              for strength_idx in interaction_strength_choices]
            for j, analytic in enumerate(analytic_types):
                all_analytic_matrices[j][tuple(
                    idxs)] = circuit.result_collector.results['signal'].analytics.get(analytic)

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

            loop_iter(i)
            # data_writer.unsubdivide_last_dir()

        logging.info('Finished: outputting final matrices')
        write_all()

    experiment = Experiment(config_filepath=config, protocols=[Protocol(make_interaction_interpolation_matrices)],
                            data_writer=data_writer)
    experiment.run_experiment()
