


class InteractionSimulator():
    def __init__(self, config_args, simulator_choice: str):

        self.simulator_choice = simulator_choice
        self.sim_config_args = config_args[self.simulator_choice]

    def run(self, batch=None):
        
        if self.simulator_choice == "IntaRNA":
            from src.utils.parameter_prediction.IntaRNA.bin.copomus.IntaRNA import IntaRNA
            simulator = IntaRNA()
            if batch is not None:
                for batch_pair in zip(batch[0:-1], batch[1:]):
                    self.sim_config_args["query"] = batch_pair[0]
                    self.sim_config_args["target"] = batch_pair[1]
                    data = simulator.run(**self.sim_config_args)
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

        

