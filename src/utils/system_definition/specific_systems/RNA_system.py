import logging
import os
from src.utils.system_definition.agnostic_system.base_system import BaseSystem


class RNASystem(BaseSystem):
    def __init__(self, config_args=None, simulator="IntaRNA"):
        super().__init__(config_args)

        self.config_args = config_args
        self.simulator_choice = simulator

        self.run_simulator()

    def run_simulator(self):
        if self.simulator_choice == "IntaRNA":
            from src.utils.parameter_prediction.IntaRNA.bin.CopomuS import CopomuS
            logging.debug("Running CopomuS")
            sim_config_args = self.config_args[self.simulator_choice]
            simulator = CopomuS(sim_config_args)
            simulator.main()

            # intaRNA_cmd = "python3 -m src.utils.parameter_prediction.IntaRNA.bin.copomus.CopomuS " + 
            # os.system(intaRNA_cmd)
