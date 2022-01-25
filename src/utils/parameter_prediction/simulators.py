import logging


class InteractionSimulator():
    def __init__(self, config_args, simulator_choice: str):

        self.simulator_choice = simulator_choice
        self.sim_config_args = config_args[self.simulator_choice]

    def run(self, batch=None, self_interaction=True):
        
        if self.simulator_choice == "IntaRNA":
            from src.utils.parameter_prediction.IntaRNA.bin.copomus.IntaRNA import IntaRNA
            simulator = IntaRNA()
            if batch is not None:
                data = {}
                batch_data, batch_labels = list(batch.values()), list(batch.keys())
                for i, (label_i, sample_i) in enumerate(zip(batch_labels, batch_data)):
                    current_pair = {}
                    for j, (label_j, sample_j) in enumerate(zip(batch_labels, batch_data)):
                        if not self_interaction and i==j:
                            continue
                        self.sim_config_args["query"] = sample_i
                        self.sim_config_args["target"] = sample_j
                        current_pair[label_j] = simulator.run(**self.sim_config_args)
                        logging.debug(current_pair[label_j])
                    data[label_i] = current_pair
            else: 
                data = simulator.run(**self.sim_config_args)
            return data

        if self.simulator_choice == "CopomuS":
            from src.utils.parameter_prediction.IntaRNA.bin.CopomuS import CopomuS
            simulator = CopomuS(self.sim_config_args)
            simulator.main()
            raise NotImplementedError

        else:
            data = None
    
        return data

    def run_iteratively(self, batch):
        if self.simulator_choice == "IntaRNA":
            for data_pair in zip(batch[0:-1], batch[1:]):
                self.sim_config_args["query"] = data_pair[0]
                self.sim_config_args["target"] = data_pair[1]

    def make_interactions(self):
        data = self.run()

        

