import logging
import os
from src.utils.system_definition.agnostic_system.base_system import BaseSystem
from src.utils.parameter_prediction.simulators import Simulator


class RNASystem(BaseSystem):
    def __init__(self, config_args, simulator="IntaRNA"):
        super().__init__(config_args)

        self.config_args = config_args
        self.simulator_choice = simulator

        self.simulator = Simulator(self.config_args, self.simulator_choice)

        self.run_simulator()

    def run_simulator(self):
        self.simulator.run()
        
