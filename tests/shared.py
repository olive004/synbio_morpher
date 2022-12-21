import numpy as np
import os
from src.utils.common.setup_new import construct_circuit_from_cfg, prepare_config
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
        os.path.join('tests', 'configs', '0_weak.fasta'), # toy_mRNA_circuit_1890
        os.path.join('tests', 'configs', '1_med_weak.fasta'), # toy_mRNA_circuit_1916
        os.path.join('tests', 'configs', '2_medium.fasta'), # toy_mRNA_circuit_1598
        os.path.join('tests', 'configs', '3_med_strong.fasta'), # toy_mRNA_circuit_1196
        os.path.join('tests', 'configs', '4_strong.fasta') # toy_mRNA_circuit_1491
    ]
    interactions = np.expand_dims(np.expand_dims(np.arange(
        len(paths)), axis=1), axis=2) * np.ones((len(paths), species_num, species_num)) * default_interaction

    circuits = [construct_circuit_from_cfg(
        {'data_path': p, 'interactions_loaded': {
            'binding_rates_association': np.ones_like(i)*default_interaction,
            'binding_rates_dissociation': i,
            'eqconstants': i
        }}, config) for p, i in zip(paths, interactions[::-1])]

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