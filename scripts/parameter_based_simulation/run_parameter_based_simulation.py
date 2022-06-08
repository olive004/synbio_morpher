import logging
import os
import numpy as np
from scripts.common.circuit import construct_circuit_from_cfg
from src.srv.io.results.experiments import Experiment, Protocol
from src.srv.io.results.result_writer import ResultWriter
from src.srv.parameter_prediction.simulator import SIMULATOR_UNITS
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.decorators import time_it
from src.utils.misc.numerical import make_symmetrical_matrix_from_sequence, triangular_sequence
from src.utils.system_definition.agnostic_system.system_manager import CircuitModeller


def main(config=None, data_writer=None):
    if config is None:
        config = os.path.join(
            'scripts', 'parameter_based_simulation', 'configs', 'base_config.json')
    config_file = load_json_as_dict(config)
    if data_writer is None:
        data_writer = ResultWriter(purpose=config_file.get(
            'experiment').get('purpose', 'parameter_based_simulation'))

    def make_interaction_interpolation_matrices():
        interaction_min = 0
        interaction_max = 1
        interaction_step_size = 0.1
        interaction_array = np.arange(
            interaction_min, interaction_max, interaction_step_size)
        size_interaction_array = np.size(interaction_array)

        from src.utils.data.data_format_tools.manipulate_fasta import load_seq_from_FASTA
        sample_names = load_seq_from_FASTA(
            config_file.get("data_path"), as_type='dict')
        num_species = len(sample_names)
        num_unique_interactions = triangular_sequence(num_species)

        matrix_dimensions = tuple(
            [num_species] + [size_interaction_array]*num_unique_interactions)
        matrix_size = num_species * \
            np.power(size_interaction_array, num_unique_interactions)
        assert matrix_size == np.prod(list(
            matrix_dimensions)), 'Something is off about the intended size of the matrix'

        all_species_steady_states = np.zeros(
            matrix_dimensions, dtype=np.float32)
        all_species_response_time = np.zeros(
            matrix_dimensions, dtype=np.float32)
        all_species_response_time_low = np.zeros(
            matrix_dimensions, dtype=np.float32)
        all_species_response_time_high = np.zeros(
            matrix_dimensions, dtype=np.float32)

        num_iterations = matrix_size

        logging.info('-----------------------')
        logging.info('Rate: ca. 8000 / min')
        logging.info('Total estimated time (steady state):')
        logging.info(f'\t{359500/(13*60)} or {10000/24} in mins')
        logging.info(f'\t{359500/(13*60)/60} or {10000/24/60} in hours')
        logging.info(f'\t{359500/(13*60)/60 /24} or {10000/24/60/24} in days')
        logging.info(f'Total data: {num_iterations}')
        logging.info(f'Projected size (inc signal writing):')
        logging.info('\tRate of 100.7Mb / 500 iterations')
        logging.info(
            f'\t{np.round(100.7/500*num_iterations/1000, decimals=3)}Gb')
        modeller = CircuitModeller(result_writer=data_writer)
        for i in range(1010, num_iterations):  # 8100 per min
            if np.mod(i, 1000) == 0:
                data_writer.unsubdivide()
                data_writer.subdivide_writing(f'{i}-{i+1000-1}')
                logging.info(f'Iteration {i}/{num_iterations}')
            data_writer.subdivide_writing(str(i), safe_dir_change=False)

            @time_it
            def loop_iter():
                """ Timings:
                'naive' signal simulation
                0.0010s 
                Function name: construct_circuit_from_cfg

                0.0224s 
                Function name: init_circuit

                0.2604s 
                Function name: simulate_signal

                0.0393s 
                Function name: write_all


                'ivp' signal simulation
                0.1569s 
                Function name: loop_iter

                0.0008s 
                Function name: construct_circuit_from_cfg

                0.0183s 
                Function name: init_circuit

                0.1026s 
                Function name: simulate_signal

                0.0396s 
                Function name: write_all

                """

                iterators = [int(np.mod(i / np.power(size_interaction_array, j),
                                        size_interaction_array)) for j in range(num_unique_interactions)]
                flat_triangle = interaction_array[list(iterators)]
                interaction_matrix = make_symmetrical_matrix_from_sequence(
                    flat_triangle, num_species)
                cfg = {"interactions": {
                    "interactions_matrix": interaction_matrix,
                    "interactions_units": SIMULATOR_UNITS['IntaRNA']['rate']}
                }

                circuit = construct_circuit_from_cfg(
                    extra_configs=cfg, config_filepath=config)
                circuit = modeller.init_circuit(circuit)
                circuit = modeller.simulate_signal(circuit, use_solver=config_file.get('signal').get('use_solver', 'naive'))

                idxs = [slice(0, num_species)] + [[ite] for ite in iterators]
                all_species_steady_states[tuple(
                    idxs)] = circuit.species.steady_state_copynums[:].astype(np.float32)
                all_species_response_time[tuple(
                    idxs)] = circuit.result_collector.results['signal'].analytics.get('response_time')
                all_species_response_time_low[tuple(
                    idxs)] = circuit.result_collector.results['signal'].analytics.get('response_time_low')
                all_species_response_time_high[tuple(
                    idxs)] = circuit.result_collector.results['signal'].analytics['response_time_high']

                # logging.info(circuit.result_collector.results['signal'].analytics)
                

                # logging.info(circuit.result_collector.results['signal'].analytics.get('response_time'))
                # logging.info(all_species_response_time[all_species_response_time>0])

                @time_it
                def write_all():
                    data_writer.output(
                        'csv', out_name='flat_triangle_interaction_matrix', data=flat_triangle)
                    data_writer.output('csv', out_name='steady_state',
                                    data=circuit.species.steady_state_copynums[:])
                    # modeller.write_results(circuit=circuit, no_visualisations=False, only_numerical=False)

                    for out_type in ['npy']:
                        data_writer.output(out_type, out_name='all_species_steady_states',
                                        data=all_species_steady_states.astype(np.float32), overwrite=True,
                                        write_to_top_dir=True)
                        data_writer.output(out_type, out_name='all_species_response_time',
                                        data=all_species_response_time.astype(np.float32), overwrite=True,
                                        write_to_top_dir=True)
                        data_writer.output(out_type, out_name='all_species_response_time_low',
                                        data=all_species_response_time_low.astype(np.float32), overwrite=True,
                                        write_to_top_dir=True)
                        data_writer.output(out_type, out_name='all_species_response_time_high',
                                        data=all_species_response_time_high.astype(np.float32), overwrite=True,
                                        write_to_top_dir=True)
                if np.mod(i, 1) == 0:
                    write_all()
            # experiment = Experiment(config_filepath=config, protocols=[Protocol(loop_iter)],
            #                         data_writer=data_writer)
            # experiment.run_experiment()
            loop_iter()
            data_writer.unsubdivide_last_dir()

        logging.info('Finished: outputting final matrices')
        data_writer.output('npy', out_name='steady_state_interpolation',
                            data=all_species_steady_states.astype(np.float32), overwrite=True,
                            write_to_top_dir=True)
        data_writer.output('npy', out_name='all_species_response_time',
                            data=all_species_response_time.astype(np.float32), overwrite=True,
                            write_to_top_dir=True)
        data_writer.output('npy', out_name='all_species_response_time_low',
                            data=all_species_response_time_low.astype(np.float32), overwrite=True,
                            write_to_top_dir=True)
        data_writer.output('npy', out_name='all_species_response_time_high',
                            data=all_species_response_time_high.astype(np.float32), overwrite=True,
                            write_to_top_dir=True)

    experiment = Experiment(config_filepath=config, protocols=[Protocol(make_interaction_interpolation_matrices)],
                            data_writer=data_writer)
    experiment.run_experiment()
