import numpy as np
import pandas as pd
import os
from src.utils.common.setup_new import construct_circuit_from_cfg, prepare_config
from src.srv.sequence_exploration.sequence_analysis import load_tabulated_info, b_tabulate_mutation_info
from src.srv.io.manage.script_manager import script_preamble
from src.utils.circuit.agnostic_circuits.circuit_manager_new import CircuitModeller


def five_circuits():

    config = {
        "interaction_simulator": {
            "name": "IntaRNA",
            "postprocess": True
        },
        "experiment": {
            "purpose": "tests",
            "test_mode": True
        },
        "molecular_params": "./src/utils/common/configs/RNA_circuit/molecular_params.json",
        "circuit_generation": {
            "repetitions": 2000,
            "species_count": 3,
            "sequence_length": 20,
            "generator_protocol": "template_mutate",
            "proportion_to_mutate": 0.5
        },
        "mutations": {
            "algorithm": "all",
            "mutation_counts": 10,
            "mutation_nums_within_sequence": [1],
            "mutation_nums_per_position": 1,
            "concurrent_species_to_mutate": "single_species_at_a_time"
        },
        "filters": {
            "min_num_interacting": None,
            "max_self_interacting": None,
            "max_total": None
        },
        "signal": {
            "inputs": ["RNA0"],
            "outputs": ["RNA1"],
            "function_name": "step_function",
            "function_kwargs": {
                "impulse_center": 400,
                "impulse_halfwidth": 5,
                "target": 10
            }
        },
        "simulation": {
            "dt": 0.1,
            "t0": 0,
            "t1": 1200,
            "solver": "diffrax",
            "use_batch_mutations": True,
            "batch_size": 100,
            "max_circuits": 1000,
            "device": "cpu"
        },
        "system_type": "RNA"
    }
    config, data_writer = script_preamble(config=config, data_writer=None)
    config = prepare_config(config_file=config)

    default_interaction = 0.0015
    species_num = 9

    paths = [
        # toy_mRNA_circuit_1890
        os.path.join('tests', 'configs', 'circuits', '0_weak.fasta'),
        # toy_mRNA_circuit_1916
        os.path.join('tests', 'configs', 'circuits', '1_med_weak.fasta'),
        # toy_mRNA_circuit_1598
        os.path.join('tests', 'configs', 'circuits', '2_medium.fasta'),
        # toy_mRNA_circuit_1196
        os.path.join('tests', 'configs', 'circuits', '3_med_strong.fasta'),
        # toy_mRNA_circuit_1491
        os.path.join('tests', 'configs', 'circuits', '4_strong.fasta')
    ]

    interaction_paths = []
    for inter in ['binding_rates_association', 'eqconstants']:
        interaction_paths.append([
            # toy_mRNA_circuit_1890
            os.path.join('tests', 'configs', inter, f'0_weak_{inter}.csv'),
            # toy_mRNA_circuit_1916
            os.path.join('tests', 'configs', inter, f'1_med_weak_{inter}.csv'),
            # toy_mRNA_circuit_1598
            os.path.join('tests', 'configs', inter, f'2_medium_{inter}.csv'),
            # toy_mRNA_circuit_1196
            os.path.join('tests', 'configs', inter,
                         f'3_med_strong_{inter}.csv'),
            # toy_mRNA_circuit_1491
            os.path.join('tests', 'configs', inter, f'4_strong_{inter}.csv')
        ])

    # interactions = np.expand_dims(np.expand_dims(np.arange(
    #     len(paths)), axis=1), axis=2) * np.ones((len(paths), species_num, species_num)) * default_interaction
    interactions_cfg = [
        {'binding_rates_association': config['molecular_params']
            ['creation_rate'], 'binding_rates_dissociation': bp, 'eqconstants': ep}
        for bp, ep in zip(interaction_paths[0], interaction_paths[1])]

    return [construct_circuit_from_cfg(
        {'data_path': p, 'interactions': i}, config) for p, i in zip(paths, interactions_cfg)], config, data_writer

    CircuitModeller(result_writer=data_writer, config=config).batch_circuits(
        circuits=circuits,
        write_to_subsystem=True,
        batch_size=config['simulation'].get('batch_size', 100),
        methods={
            "init_circuit": {},
            "simulate_signal_batch": {'save_numerical_vis_data': True, 'ref_circuit': None,
                                      'batch': True},
            "write_results": {'no_visualisations': False,
                              'no_numerical': False}
        })
