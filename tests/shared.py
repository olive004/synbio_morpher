import numpy as np
import pandas as pd
import os
from src.srv.io.manage.script_manager import script_preamble
from src.utils.circuit.agnostic_circuits.circuit_manager_new import CircuitModeller
from src.utils.common.setup_new import construct_circuit_from_cfg, prepare_config
from src.utils.evolution.evolver import Evolver
from src.utils.misc.type_handling import merge_dicts, replace_value


ENSEMBLE_CONFIG = {
    "experiment": {
        "purpose": "tests",
        "test_mode": False,
        "no_visualisations": True,
        "no_numerical": True
    },
    "base_configs_ensemble": {
        "generate_species_templates": {
            "experiment": {
                "purpose": "generate_species_templates",
                "test_mode": False
            },
            "interaction_simulator": {
                "name": "IntaRNA",
                "postprocess": True
            },
            "molecular_params": "./src/utils/common/configs/RNA_circuit/molecular_params.json",
            "system_type": "RNA"
        },
        "gather_interaction_stats": {
            "experiment": {
                "purpose": "gather_interaction_stats",
                "test_mode": False
            },
            "interaction_simulator": {
                "name": "IntaRNA",
                "postprocess": True
            },
            "interaction_file_keyword": ["eqconstants", "binding_rates_dissociation"],
            "molecular_params": "./src/utils/common/configs/RNA_circuit/molecular_params.json",
            "source_of_interactions": {
                "is_source_dir_incomplete": True,
                "source_dir": "./data/tests",
                "purpose_to_get_source_dir_from": "generate_species_templates",
                "source_dir_actually_used_POSTERITY": None
            }
        },
        "mutation_effect_on_interactions_signal": {
            "experiment": {
                "purpose": "mutation_effect_on_interactions_signal",
                "test_mode": False
            },
            "mutations": {
                "algorithm": "random",
                "mutation_counts": 1,
                "mutation_nums_within_sequence": [1],
                "mutation_nums_per_position": 1,
                "seed": 1,
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
            "source_of_interaction_stats": {
                "is_source_dir_incomplete": True,
                "source_dir": "./data/tests",
                "purpose_to_get_source_dir_from": "gather_interaction_stats",
                "source_dir_actually_used_POSTERITY": None
            },
            "system_type": "RNA"
        },
    }
}

CONFIG = merge_dicts(*list(ENSEMBLE_CONFIG["base_configs_ensemble"].values()) + [
                     {k: v for k, v in ENSEMBLE_CONFIG.items() if k == 'experiment'}])
TEST_CONFIG = replace_value(CONFIG, 'test_mode', True)


def five_circuits(config: dict, data_writer=None):
    """ Use a config with params from generate_species_templates """

    config, data_writer = script_preamble(
        config=config, data_writer=data_writer)
    config = prepare_config(config_file=config)

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
    for inter in ['binding_rates_dissociation', 'eqconstants']:
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


def mutate(circuits, config, data_writer):

    for c in circuits:
        c = Evolver(data_writer=data_writer,
                    sequence_type=config.get('system_type'),
                    seed=config.get('mutations', {}).get('seed')).mutate(
            c,
            write_to_subsystem=True,
            algorithm=config.get('mutations', {}).get('algorithm', 'random'))
    return circuits, config, data_writer


def simulate(circuits, config, data_writer):

    CircuitModeller(result_writer=data_writer, config=config).batch_circuits(
        circuits=circuits,
        write_to_subsystem=True,
        batch_size=config['simulation'].get('batch_size', 100),
        methods={
            "init_circuit": {},
            "simulate_signal_batch": {'ref_circuit': None,
                                      'batch': config['simulation']['use_batch_mutations']},
            "write_results": {'no_visualisations': config['experiment']['no_visualisations'],
                              'no_numerical': config['experiment']['no_numerical']}
        })

    return circuits, config, data_writer