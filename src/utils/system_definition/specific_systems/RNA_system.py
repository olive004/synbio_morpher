import logging
from src.utils.misc.decorators import time_it
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

        self.process_species()

        self.simulator_args = simulator_args
        self.simulator_choice = simulator

        self.simulate_interaction_strengths()
        self.model_circuit()

    @time_it
    def get_part_to_part_intrs(self):
        self.species.data = self.run_simulator()
        return self.species.data.matrix

    def run_simulator(self, data=None):
        data = data if data is not None else self.species.data.data
        self.simulator = InteractionSimulator(
            self.simulator_args, self.simulator_choice)
        return self.simulator.run(data)

    def process_species(self):
        self.node_labels = self.species.data.sample_names

    def simulate_interaction_strengths(self):
        self.species.interactions = self.get_part_to_part_intrs()

    def model_circuit(self):
        logging.basicConfig(level=logging.INFO)
        from src.utils.system_definition.agnostic_system.modelling import Deterministic
        modeller = Deterministic()
        data = modeller.dxdt_RNA(self.species.copynumbers, self.species.interactions,
                                 self.species.creation_rates, self.species.degradation_rates)
        modeller.plot(data)


class RNASpecies(BaseSpecies):
    def __init__(self, simulator_args, simulator="IntaRNA"):
        super().__init__(simulator_args)
