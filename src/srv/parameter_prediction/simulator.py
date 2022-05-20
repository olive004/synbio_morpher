from src.srv.parameter_prediction.interactions import InteractionData, RawSimulationHandling


class InteractionSimulator():
    def __init__(self, config_args):

        self.simulation_handler = RawSimulationHandling(config_args)

    def run(self, batch=None, allow_self_interaction=True):
        """ Makes nested dictionary for querying interactions as 
        {sample1: {sample2: interaction}} """

        simulation_func = self.simulation_handler.get_simulation(allow_self_interaction)
        data = simulation_func(batch)
        data = InteractionData(data, simulation_handler=self.simulation_handler)
        return data

        
def simulate_vanilla(batch):
    raise NotImplementedError
    return None

def simulate_intaRNA_data(batch, allow_self_interaction, sim_kwargs):
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
