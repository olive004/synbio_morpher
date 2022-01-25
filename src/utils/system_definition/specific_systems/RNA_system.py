import logging
import os
from src.utils.system_definition.agnostic_system.base_system import BaseSystem, BaseSpecies
from src.utils.parameter_prediction.simulators import InteractionSimulator

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
FORMAT = "%(filename)s:%(funcName)s():%(lineno)i: %(message)s %(levelname)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RNASystem(BaseSystem):
    def __init__(self, simulator_args, simulator="IntaRNA"):
        super(RNASystem, self).__init__(simulator_args)

        self.process_data()

        self.simulator_args = simulator_args
        self.simulator_choice = simulator

        self.simulate_interaction_strengths()

    def get_part_to_part_intrs(self):
        logger.debug(self.data)
        data = self.run_simulator()
        logger.debug(data)

    def run_simulator(self):
        self.simulator = InteractionSimulator(
            self.simulator_args, self.simulator_choice)
        return self.simulator.run(self.data.data)

    def process_data(self):
        self.node_labels = self.data.sample_names

    def simulate_interaction_strengths(self):
        self.get_part_to_part_intrs()
        pass


class RNASpecies(BaseSpecies):
    def __init__(self, simulator_args, simulator="IntaRNA"):
        super().__init__(simulator_args)
