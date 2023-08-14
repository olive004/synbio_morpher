


from bioreaction.simulation.manager import simulate_steady_states
from functools import partial
from typing import Optional, Tuple, List
from datetime import datetime
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import logging


from synbio_morpher.utils.circuit.agnostic_circuits.circuit_manager import CircuitModeller
from synbio_morpher.utils.common.setup import construct_circuit_from_cfg, prepare_config
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
from synbio_morpher.utils.data.data_format_tools.manipulate_fasta import load_seq_from_FASTA
from synbio_morpher.utils.evolution.evolver import Evolver
from synbio_morpher.utils.evolution.mutation import implement_mutation, apply_mutation_to_sequence, get_mutation_type_mapping, reverse_mut_mapping
from synbio_morpher.utils.misc.type_handling import flatten_listlike
from synbio_morpher.utils.misc.numerical import add_recursively
from synbio_morpher.utils.misc.helper import vanilla_return
from synbio_morpher.utils.modelling.deterministic import bioreaction_sim_dfx_expanded
from synbio_morpher.utils.results.analytics.naming import get_analytics_types_all, get_true_names_analytics, get_true_interaction_cols
from synbio_morpher.utils.results.analytics.timeseries import generate_analytics
from synbio_morpher.utils.results.writer import DataWriter
from synbio_morpher.srv.io.loaders.circuit_loader import load_circuit
from synbio_morpher.srv.io.manage.script_manager import script_preamble


def pick_circuits(config: dict, data) -> pd.DataFrame:

    percentile = 0.9
    sensitivity_range = data['sensitivity_wrt_species-6'] > (data['sensitivity_wrt_species-6'].max() *
                                                                percentile)
    starting_circ_rows = data[sensitivity_range].sort_values(
        by=['sensitivity_wrt_species-6', 'precision_wrt_species-6'], ascending=False)

    logging.info(
        f'Picking circuits that have a sensitivity in the {percentile * 100}th percentile of at least {data["sensitivity_wrt_species-6"].max() * percentile}')
    return starting_circ_rows


def tweak_cfg(config_og: dict, config: dict) -> dict:
    config_og['experiment']['purpose'] = config['experiment']['purpose']
    config_og['simulation']['device'] = 'gpu'
    config_og['mutations_args'] = {
        'algorithm': 'random',
        'mutation_counts': 2,
        'mutation_nums_within_sequence': [1],
        'mutation_nums_per_position': 1,
        'concurrent_species_to_mutate': 'single_species_at_a_time',
        'seed': 0
    }
    config_og['simulation']['threshold_steady_states'] = 0.1
    config_og['experiment']['no_numerical'] = False
    config_og['experiment']['no_visualisations'] = True
    return config_og


def plot_starting_circuits(data, starting_circ_rows, filt, data_writer, save: bool = True):
    data['Starting circuit'] = (data['circuit_name'].isin(starting_circ_rows['circuit_name'])) & \
        (data['mutation_name'].isin(starting_circ_rows['mutation_name'])) & filt

    sns.scatterplot(
        data.sort_values(by='Starting circuit'), x='sensitivity_wrt_species-6', y='precision_wrt_species-6',
        hue='Starting circuit', alpha=((data.sort_values(by='Starting circuit')['Starting circuit'] + 0.1)/1.1)
    )
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sensitivity')
    plt.ylabel('Precision')
    
    fig_path = os.path.join(data_writer.top_write_dir, 'chosen_circuits.svg')
    plt.savefig(fig_path)
    

## Simulation functions

def mutate(circuits: list, evolver: Evolver, algorithm: str = 'random'):
    
    for c in circuits:
        c = evolver.mutate(
            c, write_to_subsystem=True, algorithm=algorithm)

    return circuits


def simulate(circuits: list, modeller: CircuitModeller, config: dict) -> list:
    
    circuits = modeller.batch_circuits(
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
        }
    )
    
    return circuits


# Helper functions

def load_all_analytics(circuit_dirs):
    """ Top (starting) directory should be the starting circuit for each round """
    
    fbn = 'report_signal.json'
    circuits = {}
    for circuit_dir in circuit_dirs:
        if 'mutations' in os.listdir(circuit_dir):
            circuits[os.path.basename(circuit_dir)] = load_all_analytics(
                [os.path.join(circuit_dir, sc) for sc in os.listdir(circuit_dir)]) 
            circuits[os.path.basename(circuit_dir)]['ref_circuit'] = load_json_as_dict(
                os.path.join(circuit_dir, fbn))
        else:
            circuits[os.path.basename(circuit_dir)] = load_json_as_dict(os.path.join(circuit_dir, fbn))
    return circuits

def make_next_name(name: str):
    special_char = '_N_'
    return name.split(special_char)[0] + special_char + ''.join(str(datetime.now()).replace('.', '').split(':')[1:])


# Choose next

def choose_next(batch: list, data_writer, choose_max: int = 4, target_species: List[str] = ['RNA_1', 'RNA_2']):
    
    def make_data(batch, batch_analytics, target_species: List[str]):
        # mutated_species = [[(c.name, jax.tree_util.tree_flatten(v.keys())) for k, v in c.mutations.items()] for c in batch]
        d = pd.DataFrame(
            data=np.concatenate(
                [
                    np.asarray([c.name for c in batch])[:, None],
                    np.asarray([c.subname for c in batch])[:, None]
                    # np.asarray(flatten_listlike([[m.template_name for m in jax.tree_util.tree_flatten(c.mutations)[0]] for c in batch]))[:, None]
                ], axis=1
            ),
            columns=['Name', 'Subname']
        )
        d['Circuit Obj'] = batch
        t_idxs = {s.name: batch[0].model.species.index(s.name) for s in batch[0].model.species if s.name in target_species}
        for t in target_species:
            t_idx = t_idxs[t]
            d[f'Sensitivity species-{t}'] = np.asarray([b['sensitivity_wrt_species-6'][t_idx] for b in batch_analytics])
            d[f'Precision species-{t}'] = np.asarray([b['precision_wrt_species-6'][t_idx] for b in batch_analytics])
        return d
        
    scale_sensitivity = 1
    scale_precision = 1

    batch_analytics = [load_json_as_dict(os.path.join(data_writer.top_write_dir, c.name, 'report_signal.json')) for c in batch]
    batch_analytics = jax.tree_util.tree_map(lambda x: np.float32(x), batch_analytics)
    # starting_analytics = [circuit.result_collector.get_result('signal').analytics for circuit in starting]
    # batch_analytics = [circuit.result_collector.get_result('signal').analytics for circuit in batch]
    data_1 = make_data(batch, batch_analytics, target_species)
    
    rs = data_1[data_1['Subname'] == 'ref_circuit']
    data_1['Parent Sensitivity'] = jax.tree_util.tree_map(lambda n: rs[rs['Name'] == n]['Sensitivity'].iloc[0], data_1['Name'].to_list())
    data_1['Parent Precision'] = jax.tree_util.tree_map(lambda n: rs[rs['Name'] == n]['Precision'].iloc[0], data_1['Name'].to_list())
    
    data_1['dS'] = data_1['Sensitivity'] - data_1['Parent Sensitivity']
    data_1['dP'] = data_1['Precision'] - data_1['Parent Precision']
    
    circuits_chosen = data_1[(data_1['dS'] >= 0) & (data_1['dP'] >= 0)].sort_values(by=['Sensitivity', 'Precision'], ascending=False)['Circuit Obj'].iloc[:choose_max].to_list()
    data_1['Next selected'] = data_1['Circuit Obj'].isin(circuits_chosen)
    return circuits_chosen, data_1


# Process mutations between runs

def get_mutated_sequences(path, circ_row, mutation_type_mapping) -> dict:
    
    if not os.path.isfile(path):
        path = os.path.join('..', path)
        assert os.path.isfile(path), f'Path {path} is not valid.'
    
    if circ_row['mutation_name'] == 'ref_circuit': 
        return path

    sequences = load_seq_from_FASTA(path, as_type = 'dict')
    mutated_species = circ_row['mutation_name'][:5]
    mutation_types = jax.tree_util.tree_map(lambda x: mutation_type_mapping[x], circ_row['mutation_type'])
    mutated_sequence = apply_mutation_to_sequence(
        sequences[mutated_species], circ_row['mutation_positions'], mutation_types)
    
    sequences[mutated_species] = mutated_sequence
    return sequences

    
def process_for_next_run(circuits: list, data_writer: DataWriter):
    
    for i, c in enumerate(circuits):
        circuits[i].name = make_next_name(c.name)
        # sequences = {s.name: s.physical_data for s in c.model.species if s.physical_data}
        sequences = load_seq_from_FASTA(c.data.source, as_type='dict')
        circuits[i].data.source = write_mutated_circuit(
            name=circuits[i].name, subname='ref_circuit', sequences=sequences, data_writer=data_writer)
    return circuits
    
    
def write_mutated_circuit(
    name: str, subname: str, sequences, data_writer: DataWriter):
    
    data_writer.subdivide_writing(name)
    if subname != 'ref_circuit':
        data_writer.subdivide_writing(subname, safe_dir_change=False)
    
    new_path = data_writer.output(
        out_name=name,
        out_type='fasta', return_path=True,
        data=sequences, byseq=True
    )
    
    data_writer.unsubdivide()
    return new_path


def make_starting_circuits(starting_circuits: pd.DataFrame, config: dict, data_writer):
    d = {v: v for v in jax.tree_util.tree_flatten(get_mutation_type_mapping('RNA'))[0]}
    mutation_type_mapping_rev = jax.tree_util.tree_map(lambda x: reverse_mut_mapping(x), d)

    circuits = []
    for _, circ_row in starting_circuits.iterrows():
        curr_config = config
        sequences = get_mutated_sequences(circ_row['path_to_template_circuit'], circ_row, mutation_type_mapping_rev)
        path = write_mutated_circuit(
            name=circ_row['circuit_name'],
            subname=circ_row['mutation_name'],
            sequences=sequences,
            data_writer=data_writer)
        curr_config['data_path'] = path
        circuit = construct_circuit_from_cfg(prev_configs=None, config_file=curr_config) 
        circuit.name = make_next_name(circ_row['circuit_name'])
        circuits.append(circuit)
        
    return circuits


def loop(config, data_writer, modeller, evolver, starting_circ_rows):
    target_species = ['RNA_1', 'RNA_2']
    choose_max = 20
    total_steps = 100

    starting = make_starting_circuits(starting_circ_rows.iloc[:choose_max], config, data_writer)
    starting = simulate(starting, modeller, config)
    
    summary = {}
    summary_batch = {}
    summary_datas = {}
    summary[0] = starting
    for step in range(total_steps):
        
        print(f'\n\nStarting batch {step+1} out of {total_steps}\n\n')

        batch = mutate(starting, evolver, algorithm='random')
        batch = simulate(batch, modeller, config)
        expanded_batchs = []
        for b in batch:
            config['data_path'] = b.data.source
            expanded_batchs.append(load_circuit(
                os.path.join(data_writer.top_write_dir, b.name), 
                name=b.name, config=config, load_mutations_as_circuits=True))
        expanded_batchs = flatten_listlike(expanded_batchs, safe=True)
        starting, summary_data = choose_next(batch=expanded_batchs, data_writer=data_writer, choose_max=choose_max, target_species=target_species)
        starting = process_for_next_run(starting, data_writer=data_writer, run=step)
        
        summary[step+1] = starting
        summary_batch[step] = expanded_batchs
        summary_datas[step] = summary_data
        
        for i in range(len(starting)):
            starting[i].subname = 'ref_circuit'
            
    return summary_datas
            

# Visualise circuit trajectory

def visualise_step_plot(summary_datas: pd.DataFrame, data_writer):
    plt.figure(figsize=(len(summary_datas) * 7, 7))
    for step, sdata in summary_datas.items():
        ax = plt.subplot(1,len(summary_datas), step+1)
        sns.scatterplot(sdata.sort_values(by=['Next selected']), x='Sensitivity', y='Precision', hue='Next selected', alpha=0.1)
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'Step {step}')
    
    fig_path = os.path.join(data_writer.top_write_dir, 'step_change.svg')
    plt.savefig(fig_path)



def main(config=None, data_writer=None):
    
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        "synbio_morpher", "scripts", "mc_evolution", "configs", "base_config.json"))
    
    fn = config['info_path']
    data = pd.read_csv(fn)
    data['mutation_type'] = data['mutation_type'].str.strip('[]').str.split(',').apply(lambda x: [int(xx) for xx in x if xx])
    data['mutation_positions'] = data['mutation_positions'].str.strip('[]').str.split(',').apply(lambda x: [int(xx) for xx in x if xx])

    config_og = load_json_as_dict(os.path.join(fn.split('summarise')[
                            0], 'mutation_effect_on_interactions_signal', 'experiment.json'))
    config_og = config_og['config_filepath']
    config_og = prepare_config(config_file=config_og)
    config_og = tweak_cfg(config_og, config)

    modeller = CircuitModeller(data_writer, config_og)
    evolver = Evolver(data_writer, mutation_type='random', sequence_type='RNA', seed=0)

    signal_species = config_og['signal']['inputs']
    filt = (
        (data[get_true_interaction_cols(data, 'energies')].sum(axis=1) != 0) &
        (data['sample_name'].isin(signal_species) != True) &
        (data['overshoot'] > 0)
    )

    starting_circ_rows = pick_circuits(config_og, data[filt])

    plot_starting_circuits(data, starting_circ_rows, filt, data_writer)
    
    summary_datas = loop(config_og, data_writer, modeller, evolver, starting_circ_rows)
    
    data_writer.subdivide_writing('summary_datas')
    for step, sdata in summary_datas.items():
        data_writer.output('csv', out_name=step, write_master=False, data=sdata)
    
    data_writer.unsubdivide()
    visualise_step_plot(summary_datas, data_writer)

