import logging
import os
from src.utils.system_definition.agnostic_system.base_system import BaseSystem, BaseSpecies
from src.utils.parameter_prediction.simulators import InteractionSimulator


class RNASystem(BaseSystem):
    def __init__(self, config_args, simulator="IntaRNA"):
        super().__init__(config_args)

        self.config_args = config_args
        self.simulator_choice = simulator

        self.simulator = InteractionSimulator(self.config_args, self.simulator_choice)

        self.run_simulator()

    def run_simulator(self):
        return self.simulator.run()

if input_type=="RNA":
            return RNASpecies

    def simulate_interaction_strengths(self):
        self.interactions = self.get_part_to_part_intrs()
        
    def get_part_to_part_intrs(self):
        data = self.run_simulator()


class RNASpecies(BaseSpecies):
    def __init__(self, config_args, simulator="IntaRNA"):
        super().__init__(config_args)