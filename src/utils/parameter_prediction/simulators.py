import numpy as np


class InteractionSimulator():
    def __init__(self, config_args, simulator_choice: str):

        self.simulator_choice = simulator_choice
        self.sim_config_args = config_args[self.simulator_choice]

    def run(self, batch=None, allow_self_interaction=True):
        """ Makes nested dictionary for querying interactions as 
        {sample1: {sample2: interaction}} """
        if self.simulator_choice == "IntaRNA":
            data = self.simulate_intaRNA_data(batch, allow_self_interaction)

        if self.simulator_choice == "CopomuS":
            # from src.utils.parameter_prediction.IntaRNA.bin.CopomuS import CopomuS
            # simulator = CopomuS(self.sim_config_args)
            # simulator.main()
            raise NotImplementedError

        else:
            data = None
        data = InteractionData(data, simulator_type="IntaRNA")
        return data

    def run_iteratively(self, batch):
        if self.simulator_choice == "IntaRNA":
            for data_pair in zip(batch[0:-1], batch[1:]):
                self.sim_config_args["query"] = data_pair[0]
                self.sim_config_args["target"] = data_pair[1]

    def simulate_intaRNA_data(self, batch, allow_self_interaction):
        from src.utils.parameter_prediction.IntaRNA.bin.copomus.IntaRNA import IntaRNA
        simulator = IntaRNA()
        if batch is not None:
            data = {}
            batch_data, batch_labels = list(batch.values()), list(batch.keys())
            for i, (label_i, sample_i) in enumerate(zip(batch_labels, batch_data)):
                current_pair = {}
                for j, (label_j, sample_j) in enumerate(zip(batch_labels, batch_data)):
                    if not allow_self_interaction and i==j:
                        continue
                    if i>j:  # Skip symmetrical
                        current_pair[label_j] = data[label_j][label_i]
                    else:
                        self.sim_config_args["query"] = sample_i
                        self.sim_config_args["target"] = sample_j
                        current_pair[label_j] = simulator.run(**self.sim_config_args)
                data[label_i] = current_pair
        else: 
            data = simulator.run(**self.sim_config_args)
        return data
        