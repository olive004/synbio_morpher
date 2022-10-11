
import logging
import numpy as np
from functools import partial
from src.utils.misc.helper import vanilla_return
from src.utils.misc.numerical import SCIENTIFIC
from src.utils.misc.units import per_mol_to_per_molecules


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

    def __init__(self, config_args: dict = None) -> None:
        self.simulator_name = config_args.get('name', 'IntaRNA')
        self.postprocess = config_args.get('postprocess')
        self.sim_kwargs = config_args.get('simulator_kwargs', {})
        self.fixed_rate_k_a = config_args.get(
            'molecular_params').get('association_binding_rate')
        self.units = ''

    def get_protocol(self, custom_prot: str = None):

        def intaRNA_calculator(sample):
            raw_sample = sample.get('E', 0)
            return raw_sample

        def intaRNA_test_protocol(sample):
            return sample

        if self.simulator_name == "IntaRNA":
            return intaRNA_calculator

    @staticmethod
    def rate_to_energy(rate):
        """ Reverse translation of interaction binding energy to binding rate:
        AG = RT ln(K)
        AG = RT ln(kb/kd)
        K = e^(G / RT)
        """
        rate[rate == 0] = 1
        energy = np.multiply(SCIENTIFIC['RT'], np.log(rate.astype('float64')))
        energy = energy / 1000
        return energy

    def get_postprocessing(self):

        def energy_to_eqconstant(energies):
            """ Translate interaction binding energy to the
            equilibrium rate of binding. Output in mol:
            AG = RT ln(K)
            AG = RT ln(kb/kd)
            K = e^(G / RT)
            """
            energies = energies * 1000  # convert kJ/mol to J/mol
            K = np.exp(np.divide(energies, SCIENTIFIC['RT']))
            return K

        def eqconstant_to_rate(eqconstants):
            """ Translate the equilibrium rate of binding to
            the rate of binding (either association or dissociation
            rate - in this case dissociation). Input in mol, output in molecules:
            k_a: binding rate per Ms 
            eqconstants: unitless but in terms of mol
            k_d: unbinding rate per s"""
            k_a = per_mol_to_per_molecules(self.fixed_rate_k_a)
            k_d = np.divide(k_a, eqconstants)
            return k_d

        def return_both_eqconstants_and_rates(eqconstants):
            return eqconstants, eqconstant_to_rate(eqconstants)

        def processor(input, funcs):
            for func in funcs:
                input = func(input)
            return input

        if self.simulator_name == "IntaRNA":
            if self.postprocess:
                self.units = SIMULATOR_UNITS[self.simulator_name]['rate']
                return partial(processor, funcs=[
                    energy_to_eqconstant,
                    return_both_eqconstants_and_rates])
            else:
                return vanilla_return

    def get_simulator(self, allow_self_interaction=True):

        if self.simulator_name == "IntaRNA":
            from src.srv.parameter_prediction.simulator import simulate_intaRNA_data
            self.units = SIMULATOR_UNITS[self.simulator_name]['energy']
            return partial(simulate_intaRNA_data,
                           allow_self_interaction=allow_self_interaction,
                           sim_kwargs=self.sim_kwargs)

        if self.simulator_name == "CopomuS":
            # from src.utils.parameter_prediction.IntaRNA.bin.CopomuS import CopomuS
            # simulator = CopomuS(self.sim_config_args)
            # simulator.main()
            raise NotImplementedError

        else:
            from src.srv.parameter_prediction.simulator import simulate_vanilla
            return simulate_vanilla

    def calculate_full_coupling_of_rates(self, k_d, degradation_rates):
        k_a = per_mol_to_per_molecules(self.fixed_rate_k_a)
        full_interactions = np.divide(k_a, (k_d + degradation_rates.flatten()))
        return full_interactions


class InteractionSimulator():
    def __init__(self, sim_args: dict = None):

        self.simulation_handler = RawSimulationHandling(sim_args)

    def run(self, batch: dict = None, allow_self_interaction=True):
        """ Makes nested dictionary for querying interactions as 
        {sample1: {sample2: interaction}} """

        simulator = self.simulation_handler.get_simulator(
            allow_self_interaction)
        data = simulator(batch)
        data = InteractionData(
            data, simulation_handler=self.simulation_handler)
        return data


def simulate_vanilla(batch):
    raise NotImplementedError
    return None


def simulate_intaRNA_data(batch: dict, allow_self_interaction: bool, sim_kwargs: dict):
    from src.srv.parameter_prediction.IntaRNA.bin.copomus.IntaRNA import IntaRNA
    simulator = IntaRNA()
    if batch is not None:
        data = {}
        batch_data, batch_labels = list(
            batch.values()), list(batch.keys())
        for i, (label_i, sample_i) in enumerate(zip(batch_labels, batch_data)):
            current_pair = {}
            for j, (label_j, sample_j) in enumerate(zip(batch_labels, batch_data)):
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


class InteractionData():

    def __init__(self, data, simulation_handler: RawSimulationHandling,
                 test_mode=False):
        self.simulation_handler = simulation_handler
        self.simulation_protocol = simulation_handler.get_protocol()
        self.simulation_postproc = simulation_handler.get_postprocessing()
        if not test_mode:
            self.data, self.eqconstants, self.binding_rates = self.parse(data)
        else:
            self.data = self.simulation_protocol()
        self.units = simulation_handler.units

    def calculate_full_coupling_of_rates(self, degradation_rates):
        self.coupled_binding_rates = self.simulation_handler.calculate_full_coupling_of_rates(
            k_d=self.binding_rates, degradation_rates=degradation_rates
        )
        return self.coupled_binding_rates

    def parse(self, data):
        matrix, rates = self.make_matrix(data)
        return data, matrix, rates

    def make_matrix(self, data):
        matrix = np.zeros((len(data), len(data)))
        for i, (sample_i, sample_interactions) in enumerate(data.items()):
            for j, (sample_j, raw_sample) in enumerate(sample_interactions.items()):
                matrix[i, j] = self.get_interaction(raw_sample)
        matrix, rates = self.simulation_postproc(matrix)
        return matrix, rates

    def get_interaction(self, sample):
        if sample == False:
            return 0
        return self.simulation_protocol(sample)
