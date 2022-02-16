from copy import deepcopy
import numpy as np
import logging
from src.utils.misc.decorators import time_it
from src.utils.system_definition.agnostic_system.base_system import BaseSystem, BaseSpecies
from src.srv.parameter_prediction.simulators import InteractionSimulator

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
FORMAT = "%(filename)s:%(funcName)s():%(lineno)i: %(message)s %(levelname)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RNASystem(BaseSystem):
    def __init__(self, simulator_args, simulator="IntaRNA"):
        super(RNASystem, self).__init__(simulator_args)

        self.process_species()

        self.simulator_args = simulator_args
        self.simulator_choice = simulator

        # 2 h^-1 or 30 mins or 1800s - Dilution rate
        self.cell_dbl_growth_rate = 2 / 3600
        self.avg_RNA_pcell = 100

        self.transcription_rate = self.cell_dbl_growth_rate * self.avg_RNA_pcell
        self.species.copynumbers = self.species.init_matrix(ndims=1, init_type="uniform",
                                                            uniform_val=100)
        self.species.degradation_rates = self.species.init_matrix(ndims=1, init_type="uniform",
                                                                  uniform_val=self.cell_dbl_growth_rate)
        self.species.creation_rates = self.species.init_matrix(ndims=1, init_type="uniform",
                                                               uniform_val=self.transcription_rate)

        self.simulate_interaction_strengths()
        self.model_circuit()

    @time_it
    def get_part_to_part_intrs(self):
        interactions = self.run_simulator()
        return interactions.matrix

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
        from src.utils.misc.numerical import zero_out_negs
        modeller = Deterministic()

        max_time = 100
        self.species.all_copynumbers = np.zeros(
            (max_time, self.species.data.size))
        self.species.all_copynumbers[0] = deepcopy(self.species.copynumbers)
        current_copynumbers = deepcopy(self.species.copynumbers)
        for tstep in range(max_time-1):
            current_copynumbers += modeller.dxdt_RNA(self.species.all_copynumbers[tstep],
                                                     self.species.interactions,
                                                     self.species.creation_rates,
                                                     self.species.degradation_rates)
            current_copynumbers = zero_out_negs(current_copynumbers)
            self.species.all_copynumbers[tstep+1] = current_copynumbers

        self.result_writer.add_result(self.species.all_copynumbers,
                                      visualisation_type='time_series',
                                      vis_func=modeller.plot,
                                      **{'legend_keys': list(self.species.data.sample_names)})


class RNASpecies(BaseSpecies):
    def __init__(self, simulator_args, simulator="IntaRNA"):
        super().__init__(simulator_args)
