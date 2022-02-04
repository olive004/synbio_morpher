import numpy as np
from functools import partial
from src.utils.misc.decorators import time_it
from src.utils.data.fake_data_generation.toy_graphs import square_matrix_rand


class RawSimulationHandling():
    def __init__(self, simulator, config_args) -> None:
        self.simulator = simulator
        self.sim_kwargs = config_args[simulator]

    def get_protocol(self):

        def intaRNA_calculator(data):
            return data.get('E', 0)

        if self.simulator == "IntaRNA":
            return intaRNA_calculator

    def get_simulation(self, allow_self_interaction=True):

        def simulate_vanilla(batch):
            return None

        @time_it
        def simulate_intaRNA_data(batch, allow_self_interaction, sim_kwargs):
            from src.utils.parameter_prediction.IntaRNA.bin.copomus.IntaRNA import IntaRNA
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

        if self.simulator == "IntaRNA":
            return partial(simulate_intaRNA_data,
                           allow_self_interaction=allow_self_interaction,
                           sim_kwargs=self.sim_kwargs)

        if self.simulator == "CopomuS":
            # from src.utils.parameter_prediction.IntaRNA.bin.CopomuS import CopomuS
            # simulator = CopomuS(self.sim_config_args)
            # simulator.main()
            raise NotImplementedError

        else:
            return simulate_vanilla


class InteractionMatrix():
    def __init__(self, config_args=None,
                 num_nodes=None,
                 toy=False):
        super().__init__()

        self.toy = toy
        self.config_args = config_args

        if toy:
            self.matrix = self.make_toy_matrix(num_nodes)
        else:
            self.matrix = self.make_rand_matrix(num_nodes)

    def make_rand_matrix(self, num_nodes):
        if num_nodes is None or num_nodes == 0:
            num_nodes = 1
        return square_matrix_rand(num_nodes)

    def make_toy_matrix(self, num_nodes=None):
        if not num_nodes:
            min_nodes = 2
            max_nodes = 15
            num_nodes = np.random.randint(min_nodes, max_nodes)
        return self.make_rand_matrix(num_nodes)


class InteractionData():
    # R = the gas constant = 8.314 J/mol·K
    # T = 298 K
    RT = np.multiply(8.314, 298)
    def __init__(self, data, simulation_handler: RawSimulationHandling):
        self.simulation_handling = simulation_handler
        self.data, self.matrix = self.parse(data)

    def parse(self, data):
        matrix = self.make_matrix(data)
        return data, matrix

    def make_matrix(self, data):
        matrix = np.zeros((len(data), len(data)))
        for i, (sample_i, sample_interactions) in enumerate(data.items()):
            for j, (sample_j, raw_data) in enumerate(sample_interactions.items()):
                matrix[i, j] = self.calculate_interaction(raw_data)
        return matrix

    def calculate_interaction(self, data):
        if data == False:
            return 0
        interaction_calculator = self.simulation_handling.get_protocol()
        return interaction_calculator(data)

    def energy_to_rate(self, energies):
        """ Translate interaction binding energy to binding rate """
        # AG = RT ln(K)
        # AG = RT ln(kb/kd)
        # K = e^(G / RT)

        K = np.exp(np.divide(energies, self.RT))
        return K
