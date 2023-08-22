
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
import numpy as np
import pandas as pd
import os
from synbio_morpher.srv.io.manage.script_manager import script_preamble
from synbio_morpher.srv.io.manage.sys_interface import PACKAGE_DIR
from synbio_morpher.srv.sequence_exploration.sequence_analysis import b_tabulate_mutation_info
from synbio_morpher.utils.circuit.agnostic_circuits.circuit_manager import CircuitModeller
from synbio_morpher.utils.common.setup import construct_circuit_from_cfg, prepare_config
from synbio_morpher.utils.evolution.evolver import Evolver
from synbio_morpher.utils.misc.type_handling import merge_dicts, replace_value


ENSEMBLE_CONFIG = {
    "experiment": {
        "purpose": "tests",
        "debug_mode": False,
        "no_visualisations": False,
        "no_numerical": False
    },
    "base_configs_ensemble": {
        "generate_species_templates": {
            "experiment": {
                "purpose": "generate_species_templates",
                "debug_mode": False
            },
            "interaction_simulator": {
                "name": "IntaRNA",
                "postprocess": True
            },
            "molecular_params": "synbio_morpher/utils/common/configs/RNA_circuit/molecular_params.json",
            "system_type": "RNA"
        },
        "gather_interaction_stats": {
            "experiment": {
                "purpose": "gather_interaction_stats",
                "debug_mode": False
            },
            "interaction_simulator": {
                "name": "IntaRNA",
                "postprocess": True
            },
            "interaction_file_keyword": ["eqconstants", "binding_rates_dissociation"],
            "molecular_params": "synbio_morpher/utils/common/configs/RNA_circuit/molecular_params.json",
            "source_of_interactions": {
                "is_source_dir_incomplete": True,
                "source_dir": "data/tests",
                "purpose_to_get_source_dir_from": "generate_species_templates",
                "source_dir_actually_used_POSTERITY": None
            }
        },
        "mutation_effect_on_interactions_signal": {
            "experiment": {
                "purpose": "mutation_effect_on_interactions_signal",
                "debug_mode": False
            },
            "mutations_args": {
                "algorithm": "random",
                "mutation_counts": 0,
                "mutation_nums_within_sequence": [1],
                "mutation_nums_per_position": 1,
                "seed": 3,
                "concurrent_species_to_mutate": "single_species_at_a_time"
            },
            "filters": {
                "min_num_interacting": None,
                "max_self_interacting": None,
                "max_total": None
            },
            "signal": {
                "inputs": ["RNA_0"],
                "outputs": ["RNA_1"],
                "function_name": "step_function_integrated",
                # "function_name": "sine_step_function",
                "function_kwargs": {
                    "impulse_center": 5,
                    # "impulse_halfwidth": 1,
                    "target": 4
                }
            },
            "simulation": {
                "dt0": 0.1,
                "t0": 0,
                "t1": 100,
                "tmax": 20000,
                "solver": "diffrax",
                "use_batch_mutations": True,
                "interaction_factor": 1,
                "batch_size": 100,
                "max_circuits": 1000,
                "device": "cpu",
                "threshold_steady_states": 0.001,
                "use_rate_scaling": True
            },
            "source_of_interaction_stats": {
                "is_source_dir_incomplete": True,
                "source_dir": "data/tests",
                "purpose_to_get_source_dir_from": "gather_interaction_stats",
                "source_dir_actually_used_POSTERITY": None
            },
            "system_type": "RNA"
        },
    }
}

CONFIG = merge_dicts(*list(ENSEMBLE_CONFIG["base_configs_ensemble"].values()) + [
                     {k: v for k, v in ENSEMBLE_CONFIG.items() if k == 'experiment'}])
TEST_CONFIG = replace_value(CONFIG, 'debug_mode', True)


def five_circuits(config: dict, data_writer=None):
    """ Use a config with params from generate_species_templates """
    # All circuits from ensemble_mutation_effect_analysis/2022_12_16_160419

    config, data_writer = script_preamble(
        config=config, data_writer=data_writer)
    config = prepare_config(config_file=config)

    paths = [
        # toy_mRNA_circuit_0
        os.path.join(PACKAGE_DIR, 'utils', 'common', 'testing', 'configs',
                     'circuits', '0_weak.fasta'),
        # toy_mRNA_circuit_940
        os.path.join(PACKAGE_DIR, 'utils', 'common', 'testing', 'configs',
                     'circuits', '1_med_weak.fasta'),
        # toy_mRNA_circuit_1306
        os.path.join(PACKAGE_DIR, 'utils', 'common', 'testing', 'configs',
                     'circuits', '2_medium.fasta'),
        # toy_mRNA_circuit_648
        os.path.join(PACKAGE_DIR, 'utils', 'common', 'testing', 'configs',
                     'circuits', '3_med_strong.fasta'),
        # toy_mRNA_circuit_999
        os.path.join(PACKAGE_DIR, 'utils', 'common', 'testing', 'configs',
                     'circuits', '4_strong.fasta')
    ]

    interaction_paths = []
    for inter in ['binding_rates_dissociation', 'eqconstants', 'energies', 'binding_sites']:
        interaction_paths.append([
            # toy_mRNA_circuit_0
            os.path.join(PACKAGE_DIR, 'utils', 'common', 'testing', 'configs',
                         inter, f'0_weak_{inter}.csv'),
            # toy_mRNA_circuit_940
            os.path.join(PACKAGE_DIR, 'utils', 'common', 'testing', 'configs',
                         inter, f'1_med_weak_{inter}.csv'),
            # toy_mRNA_circuit_1306
            os.path.join(PACKAGE_DIR, 'utils', 'common', 'testing', 'configs',
                         inter, f'2_medium_{inter}.csv'),
            # toy_mRNA_circuit_648
            os.path.join(PACKAGE_DIR, 'utils', 'common', 'testing', 'configs', inter,
                         f'3_med_strong_{inter}.csv'),
            # toy_mRNA_circuit_999
            os.path.join(PACKAGE_DIR, 'utils', 'common', 'testing', 'configs',
                         inter, f'4_strong_{inter}.csv')
        ])

    # interactions = np.expand_dims(np.expand_dims(np.arange(
    #     len(paths)), axis=1), axis=2) * np.ones((len(paths), species_num, species_num)) * default_interaction
    interactions_cfg = [
        {'binding_rates_association': config['molecular_params']['association_binding_rate' + '_per_molecule'],
         'binding_rates_dissociation': bp,
         'eqconstants': ep,
         'energies': eg,
         'binding_sites': bs}
        for bp, ep, eg, bs in zip(interaction_paths[0], interaction_paths[1], interaction_paths[2], interaction_paths[3])]

    return [construct_circuit_from_cfg(
        {'data_path': p,
         'interactions': i
         }, config) for p, i in zip(paths, interactions_cfg)], config, data_writer


def mutate(circuits, config, data_writer):

    for c in circuits:
        c = Evolver(data_writer=data_writer,
                    sequence_type=config.get('system_type'),
                    seed=config.get("mutations_args", {}).get('seed')).mutate(
            c,
            write_to_subsystem=True,
            algorithm=config.get("mutations_args", {}).get('algorithm', 'random'))
    return circuits, config, data_writer


def simulate(circuits, config, data_writer):

    circuits = CircuitModeller(result_writer=data_writer, config=config).batch_circuits(
        circuits=circuits,
        write_to_subsystem=True,
        batch_size=config['simulation'].get('batch_size', 100),
        methods={
            "compute_interactions": {},
            "init_circuits": {'batch': True},
            "simulate_signal_batch": {'ref_circuit': None,
                                      'batch': config['simulation']['use_batch_mutations']},
            "write_results": {'no_visualisations': config['experiment']['no_visualisations'],
                              'no_numerical': config['experiment']['no_numerical']}
        })

    return circuits, config, data_writer


def create_test_inputs(config: dict):
    circuits, config, data_writer = five_circuits(config, data_writer=None)
    circuits, config, data_writer = mutate(circuits, config, data_writer)
    circuits, config, data_writer = simulate(circuits, config, data_writer)

    info = b_tabulate_mutation_info(data_writer.ensemble_write_dir,
                                    data_writer=data_writer, experiment_config=config)

    return circuits, config, data_writer, info
