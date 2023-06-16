
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    

import logging
from subprocess import PIPE, run
from functools import partial
from synbio_morpher.utils.misc.helper import vanilla_return, processor
from synbio_morpher.utils.modelling.physical import equilibrium_constant_reparameterisation, eqconstant_to_rates
from synbio_morpher.srv.parameter_prediction.IntaRNA.bin.copomus.IntaRNA import IntaRNA


SIMULATOR_UNITS = {
    'IntaRNA': {
        'energy': 'kcal/mol',
        'postprocessing': {
            'energy': 'J/mol',
            'energy_molecular': 'J/molecule'
        },
        'rate': r'$s^{-1}$',
        'coupled_rates': r'$s^{-1}$'
    },
}
NO_INTERACTION_EQCONSTANT = 0


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
                'energies': sample.get('E', NO_INTERACTION_EQCONSTANT),
                'binding': sample.get('bpList', '')
            }

        def process_interaction(sample):
            if sample == False:  # No interactions
                sample = {}
            return process_IntaRNA_sample(sample)

        if self.simulator_name == "IntaRNA":
            return process_interaction

    def get_postprocessing(self, **processor_kwrgs):

        def return_both_eqconstants_and_rates(eqconstants):
            return eqconstants, eqconstant_to_rates(eqconstants, self.fixed_rate_k_a)

        if self.simulator_name == "IntaRNA":
            if self.postprocess:
                self.units = SIMULATOR_UNITS[self.simulator_name]['rate']
                processor_f = [
                    partial(
                        equilibrium_constant_reparameterisation, 
                        **processor_kwrgs),
                    return_both_eqconstants_and_rates
                ]
                return partial(
                    processor, funcs=processor_f)
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
            from synbio_morpher.srv.parameter_prediction.simulator import simulate_vanilla
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
