import logging
import os
from src.utils.system_definition.agnostic_system.base_system import BaseSystem, BaseSpecies
from src.utils.parameter_prediction.simulators import InteractionSimulator


class RNASystem(BaseSystem):
    def __init__(self, simulator_args, simulator="IntaRNA"):
        super(RNASystem, self).__init__(simulator_args)

        self.process_data()

        self.simulator_args = simulator_args
        self.simulator_choice = simulator

        self.simulator = InteractionSimulator(
            self.simulator_args, self.simulator_choice)

        self.run_simulator()
        
    def get_part_to_part_intrs(self):
        data = self.run_simulator()

    def run_simulator(self):
        return self.simulator.run()

    def process_data(self):
        self.nodes_labels = self.data.sample_names

    def simulate_interaction_strengths(self):
        self.interactions = self.get_part_to_part_intrs()


class RNASpecies(BaseSpecies):
    def __init__(self, simulator_args, simulator="IntaRNA"):
        super().__init__(simulator_args)