from copy import deepcopy
from functools import partial
import numpy as np
from scipy import integrate
import logging
from src.utils.misc.decorators import time_it
from src.utils.misc.numerical import zero_out_negs
from src.utils.signal.inputs import Signal
from src.utils.system_definition.agnostic_system.base_system import BaseSystem, BaseSpecies
from src.srv.parameter_prediction.simulator import InteractionSimulator
from src.utils.system_definition.agnostic_system.modelling import Deterministic

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
FORMAT = "%(filename)s:%(funcName)s():%(lineno)i: %(message)s %(levelname)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RNASystem(BaseSystem):
    def __init__(self, simulator_args, simulator="IntaRNA"):
        super(RNASystem, self).__init__(simulator_args)

        self.species = self.init_species(simulator_args)
        self.process_species()

        self.simulator_args = simulator_args
        self.simulator_choice = simulator

        # Rates
        # 2 h^-1 or 30 mins or 1800s - Dilution rate
        cell_dbl_growth_rate = 2 / 3600
        avg_RNA_pcell = 100
        starting_copynumber = 50
        self.transcription_rate = cell_dbl_growth_rate * avg_RNA_pcell

        self.species.copynumbers = self.species.init_matrix(ndims=1, init_type="uniform",
                                                            uniform_val=starting_copynumber)
        self.species.degradation_rates = self.species.init_matrix(ndims=1, init_type="uniform",
                                                                  uniform_val=cell_dbl_growth_rate)
        self.species.creation_rates = self.species.init_matrix(ndims=1, init_type="uniform",
                                                               uniform_val=self.transcription_rate)

        self.simulate_interaction_strengths()
        self.find_steady_states()

    def init_species(self, args):
        return RNASpecies(args)

    def process_species(self):
        self.node_labels = self.species.data.sample_names

    def simulate_interaction_strengths(self):
        self.species.interactions = self.get_part_to_part_intrs()

    @time_it
    def get_part_to_part_intrs(self):
        interactions = self.run_simulator()
        return interactions.matrix

    def run_simulator(self, data=None):
        data = data if data is not None else self.species.data.data
        self.simulator = InteractionSimulator(
            self.simulator_args, self.simulator_choice)
        return self.simulator.run(data)

    def find_steady_states(self):
        self.model_steady_state()
        steady_state_metrics = self.result_writer.get_result(key='steady_state')[
            'metrics']
        if steady_state_metrics['steady_state']:
            pass
        self.species.steady_state_copynums = steady_state_metrics['steady_state']['steady_states']

    def model_steady_state(self):
        steady_state_time = 500
        time_step = 5
        modeller_steady_state = Deterministic(
            max_time=steady_state_time, time_step=time_step)

        self.species.all_copynumbers = np.zeros(
            (self.species.data.size, modeller_steady_state.max_time))
        self.species.all_copynumbers[:, 0] = deepcopy(self.species.copynumbers)
        current_copynumbers = deepcopy(self.species.copynumbers)
        self.species.all_copynumbers = self.model_circuit(modeller_steady_state,
                                                          current_copynumbers,
                                                          self.species.all_copynumbers,
                                                          use_solver='ivp')
        self.species.copynumbers = self.species.all_copynumbers[:, -1]

        self.result_writer.add_result(self.species.all_copynumbers,
                                      name='steady_state',
                                      category='time_series',
                                      vis_func=modeller_steady_state.plot,
                                      **{'legend_keys': list(self.species.data.sample_names),
                                         'save_name': 'steady_state_plot'})

    def model_circuit(self, modeller, current_copynumbers, all_copynumbers,
                      signal=None, signal_idx=None, use_solver='naive'):
        modelling_func = partial(modeller.dxdt_RNA, interactions=self.species.interactions,
                                 creation_rates=self.species.creation_rates,
                                 degradation_rates=self.species.degradation_rates,
                                 num_samples=np.shape(
                                     self.species.all_copynumbers)[0]
                                 )

        if use_solver == 'naive':
            modelling_func = partial(modelling_func, t=None)

            for tstep in range(0, modeller.max_time-1):
                dxdt = modelling_func(
                    copynumbers=all_copynumbers[:, tstep]) * modeller.time_step

                current_copynumbers = np.add(
                    dxdt, current_copynumbers).flatten()

                if signal is not None:
                    current_copynumbers[signal_idx] = signal[tstep]
                current_copynumbers = zero_out_negs(current_copynumbers)
                all_copynumbers[:, tstep +
                                1] = current_copynumbers
        elif use_solver == 'ivp':
            all_copynumbers = integrate.solve_ivp(modelling_func, (0, modeller.max_time),
                                                  y0=current_copynumbers.flatten()).y
        return all_copynumbers

    def simulate_signal(self, signal: Signal):
        modeller_signal = Deterministic(
            max_time=signal.total_time, time_step=1
        )
        init_copynums = np.zeros(
            (self.species.data.size, modeller_signal.max_time))
        init_copynums[:, 0] = self.species.steady_state_copynums

        all_copynums = self.model_circuit(modeller_signal,
                                          self.species.steady_state_copynums,
                                          init_copynums,
                                          signal=signal.real_signal,
                                          signal_idx=signal.idx_identity)

        self.species.all_copynumbers = np.concatenate(
            (self.species.all_copynumbers, all_copynums[:, 1:]), axis=1)
        self.result_writer.add_result(all_copynums,
                                      name='signal',
                                      category='time_series',
                                      vis_func=modeller_signal.plot,
                                      **{'legend_keys': list(self.species.data.sample_names),
                                         'save_name': 'signal_plot'})


class RNASpecies(BaseSpecies):
    def __init__(self, simulator_args):
        super().__init__(simulator_args)

        self.interactions = self.init_matrix(ndims=2, init_type="randint")
        self.complexes = self.init_matrix(ndims=2, init_type="zeros")
        self.degradation_rates = self.init_matrix(ndims=1, init_type="uniform",
                                                  uniform_val=20)
        self.creation_rates = self.init_matrix(ndims=1, init_type="uniform",
                                               uniform_val=50)
        self.copynumbers = self.init_matrix(ndims=1, init_type="uniform",
                                            uniform_val=5)
        self.all_copynumbers = None  # For modelling

        self.params = {
            "creation_rates": self.creation_rates,
            "copynumbers": self.copynumbers,
            "complexes": self.complexes,
            "degradation_rates": self.degradation_rates,
            "interactions": self.interactions
        }
