


class InteractionSimulator():
    def __init__(self, config_args, simulator_choice: str):
        super().__init__(config_args)
        
        self.simulator_choice = simulator_choice
        self.sim_config_args = config_args[self.simulator_choice]

    def run(self):
        
        if self.simulator_choice == "IntaRNA":
            from src.utils.parameter_prediction.IntaRNA.bin.copomus.IntaRNA import IntaRNA
            simulator = IntaRNA()
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

    def make_interactions(self):
        data = self.run()

        

