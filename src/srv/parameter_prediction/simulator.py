
import logging
from subprocess import PIPE, run
import numpy as np
from functools import partial
from src.utils.misc.helper import vanilla_return, processor
from src.utils.misc.numerical import SCIENTIFIC
from src.utils.misc.units import per_mol_to_per_molecule
from src.srv.parameter_prediction.IntaRNA.bin.copomus.IntaRNA import IntaRNA


SIMULATOR_UNITS = {
    'IntaRNA': {
        'energy': 'kJ/mol',
        'postprocessing': {
            'energy': 'J/mol',
            'energy_molecular': 'J/molecule'
        },
        'rate': r'$s^{-1}$',
        'coupled_rates': r'$s^{-1}$'
    },
}
MIN_INTERACTION_EQCONSTANT = 0.000000001


class RawSimulationHandling():

    def __init__(self, config: dict = None) -> None:
        self.simulator_name = config.get('name', 'IntaRNA')
        self.postprocess = config.get('postprocess')
        self.sim_kwargs = config.get('simulator_kwargs', {})
        self.fixed_rate_k_a = config.get(
            'molecular_params').get('association_binding_rate' + '_per_molecule')
        self.units = ''

    def get_sim_interpretation_protocol(self):

        def process_IntaRNA_sample(sample: dict):
            """ There are a variety of parameters that IntaRNA spits out. E is hybridisation energy"""
            return {
                'matrix': sample.get('E', 100),
                'binding': sample.get('bpList', '')
            }
        
        def process_interaction(sample):
            if sample == False:  # No interactions
                sample = {}
            return process_IntaRNA_sample(sample)

        if self.simulator_name == "IntaRNA":
            return process_interaction

    def get_postprocessing(self):

        def energy_to_eqconstant(energies):
            """ Translate interaction binding energy to the
            equilibrium rate of binding. Output in mol:
            AG = - RT ln(K)
            AG = - RT ln(kb/kd)
            K = e^(- G / RT)
            """
            energies = energies * 1000  # convert kJ/mol to J/mol
            K = np.exp(np.divide(- energies, SCIENTIFIC['RT']))
            return K

        def eqconstant_to_rates(eqconstants):
            """ Translate the equilibrium rate of binding to
            the rate of binding (either association or dissociation
            rate - in this case dissociation). Input in mol, output in molecules:
            k_a: binding rate per Ms
            eqconstants: unitless but in terms of mol
            k_d: unbinding rate per s"""
            k_a = self.fixed_rate_k_a
            k_d = np.divide(k_a, eqconstants)
            return k_a*np.ones_like(k_d), k_d

        def return_both_eqconstants_and_rates(eqconstants):
            return eqconstants, eqconstant_to_rates(eqconstants)

        if self.simulator_name == "IntaRNA":
            if self.postprocess:
                self.units = SIMULATOR_UNITS[self.simulator_name]['rate']
                return partial(processor, funcs=[
                    energy_to_eqconstant,
                    return_both_eqconstants_and_rates])
            else:
                return vanilla_return

    def get_simulator(self, allow_self_interaction: bool):

        if self.simulator_name == "IntaRNA":
            self.units = SIMULATOR_UNITS[self.simulator_name]['energy']
            if check_IntaRNA_path():
                import sys
                sys.path.append(check_IntaRNA_path())
            return partial(simulate_IntaRNA,
                           allow_self_interaction=allow_self_interaction,
                           sim_kwargs=self.sim_kwargs,
                           simulator=IntaRNA())

        if self.simulator_name == "CopomuS":
            # from src.utils.parameter_prediction.IntaRNA.bin.CopomuS import CopomuS
            raise NotImplementedError

        else:
            from src.srv.parameter_prediction.simulator import simulate_vanilla
            return simulate_vanilla


def simulate_vanilla(batch):
    raise NotImplementedError


def check_IntaRNA_path():

    p = run('which IntaRNA', shell=True, stdout=PIPE,
            stderr=PIPE, universal_newlines=True)
    p2 = run('IntaRNA --version', shell=True, stdout=PIPE,
             stderr=PIPE, universal_newlines=True)
    if p.returncode == 0:
        if p2.returncode == 1:
            return p.stdout
    else:
        logging.warning(f'Could not detect IntaRNA on system: {p}')


def simulate_IntaRNA(input, compute_by_filename: bool, allow_self_interaction: bool, sim_kwargs: dict, simulator):
    if compute_by_filename:
        f = simulate_IntaRNA_fn
    else:
        f = simulate_IntaRNA_data
    return f(input, allow_self_interaction=allow_self_interaction, sim_kwargs=sim_kwargs, simulator=simulator)


def simulate_IntaRNA_data(batch: dict, allow_self_interaction: bool, sim_kwargs: dict, simulator):
    """ Possible outputs of IntaRNA can be found in their README https://github.com/BackofenLab/IntaRNA 
    Simply add any parameter column id's (E, ED1, seq1, ...) into the simulation arguments dict using the 
    `outcsvcols` variable. """

    if batch is not None:
        data = {}
        for i, (label_i, sample_i) in enumerate(batch.items()):
            current_pair = {}
            for j, (label_j, sample_j) in enumerate(batch.items()):
                if not allow_self_interaction and i == j:
                    continue
                if i > j:  # Skip symmetrical
                    current_pair[label_j] = data[label_j][label_i]
                else:
                    sim_kwargs["query"] = sample_i
                    sim_kwargs["target"] = sample_j
                    current_pair[label_j] = simulator.run(**sim_kwargs)
            data[label_i] = current_pair
    else:
        data = simulator.run(**sim_kwargs)
    return data


def simulate_IntaRNA_fn(input: tuple, allow_self_interaction: bool, sim_kwargs: dict, simulator):
    """ Use the FASTA filename to compute the interactions between all RNAs.
    If threads = 0, interactions will be computed in parallel """
    filename, species = input
    data = {k: {k: False for k in species.keys()} for k in species.keys()}
    species_str = {s.name: s for s in species}

    sim_kwargs["query"] = filename
    sim_kwargs["target"] = filename
    output = simulator.run(**sim_kwargs)
    if output:
        if type(output) == dict:
            output = [output]
        for s in output:
            if not allow_self_interaction and s['id1'] == s['id2']:
                continue
            data[species_str[s['id1']]][species_str[s['id2']]] = s
    return data
