
import numpy as np
from functools import partial
from src.srv.parameter_prediction.interactions import InteractionData
from src.utils.misc.numerical import SCIENTIFIC


SIMULATOR_UNITS = {
    'IntaRNA': {
        'energy': 'kJ/mol',
        'rate': 'rate'
    }
}


class RawSimulationHandling():

    def __init__(self, config_args: dict = None, simulator: str = 'IntaRNA') -> None:
        self.simulator_name = simulator if config_args is None else config_args.get(
            'name', simulator)
        self.sim_kwargs = config_args if config_args is not None else {}
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

        def vanilla(data):
            return data

        def energy_to_rate(energies):
            """ Translate interaction binding energy to binding rate:
            AG = RT ln(K)
            AG = RT ln(kb/kd)
            K = e^(G / RT)
            """
            energies = energies * 1000  # convert kJ/mol to J/mol
            K = np.exp(np.divide(energies, SCIENTIFIC['RT']))
            return K

        def zero_false_rates(rates):
            """ Exponential of e^0 is equal to 1, but IntaRNA sets energies 
            equal to 0 for non-interactions. """
            rates[rates == 1] = 0
            return rates

        def processor(input, funcs):
            for func in funcs:
                input = func(input)
            return input

        if self.simulator_name == "IntaRNA":
            if self.sim_kwargs.get('postprocess', None):
                self.units = SIMULATOR_UNITS[self.simulator_name]['rate']
                return partial(processor, funcs=[
                    energy_to_rate,
                    zero_false_rates])
            else:
                return vanilla

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


class InteractionSimulator():
    def __init__(self, config_args: dict = None):

        self.simulation_handler = RawSimulationHandling(config_args)

    def run(self, batch: dict=None, allow_self_interaction=True):
        """ Makes nested dictionary for querying interactions as 
        {sample1: {sample2: interaction}} """

        simulator = self.simulation_handler.get_simulator(allow_self_interaction)
        data = simulator(batch)
        data = InteractionData(data, simulation_handler=self.simulation_handler)
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
