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

        self.simulator_args = simulator_args
        self.simulator_choice = simulator

        self.species = self.init_species(simulator_args)
        self.process_species()

        # Rates
        # 2 h^-1 or 30 mins or 1800s - Dilution rate
        cell_dbl_growth_rate = 2 / 3600
        avg_RNA_pcell = 100
        starting_copynumber = 50
        self.transcription_rate = cell_dbl_growth_rate * avg_RNA_pcell

        self.species.all_copynumbers, \
            self.species.degradation_rates, \
            self.species.creation_rates = self.species.init_matrices(ndims=1, init_type="uniform",
                                                                     uniform_vals=[starting_copynumber,
                                                                                   cell_dbl_growth_rate,
                                                                                   self.transcription_rate])
        logging.info(self.species.degradation_rates)
        logging.info(self.species.creation_rates)

        self.simulate_interaction_strengths()
        self.find_steady_states()

    def init_species(self, args):
        return RNASpecies(args)

    def process_species(self):
        self.node_labels = self.species.data.sample_names

    def get_modelling_func(self, modeller):
        return partial(modeller.dxdt_RNA, interactions=self.species.interactions,
                       creation_rates=self.species.creation_rates,
                       degradation_rates=self.species.degradation_rates,
                       num_samples=self.species.data.size
                       )

    @time_it
    def simulate_interaction_strengths(self):
        interactions = self.run_simulator()
        self.species.interactions = interactions.matrix

    def run_simulator(self, data=None):
        self.simulator = InteractionSimulator(
            self.simulator_args, self.simulator_choice)
        data = data if data is not None else self.species.data.data
        return self.simulator.run(data)

    def find_steady_states(self):
        modeller_steady_state = Deterministic(
            max_time=50, time_step=1)

        self.species.all_copynumbers = np.concatenate((
            self.species.all_copynumbers,
            np.zeros(
                (self.species.data.size, modeller_steady_state.max_time-1))
        ), axis=1)
        self.species.all_copynumbers = self.compute_steady_state(modeller_steady_state,
                                                                 self.species.all_copynumbers,
                                                                 use_solver='ivp')

        self.result_writer.add_result(self.species.all_copynumbers,
                                      name='steady_state',
                                      category='time_series',
                                      vis_func=modeller_steady_state.plot,
                                      **{'legend_keys': list(self.species.data.sample_names),
                                         'save_name': 'steady_state_plot'})
        steady_state_metrics = self.result_writer.get_result(key='steady_state')[
            'metrics']
        self.species.steady_state_copynums = steady_state_metrics['steady_state']['steady_states']

    def compute_steady_state(self, modeller, all_copynumbers,
                             use_solver='naive'):
        if use_solver == 'naive':
            self.model_circuit(modeller, all_copynumbers)
        elif use_solver == 'ivp':
            logging.info(
                all_copynumbers[:, -1].reshape(all_copynumbers.shape[0], 1))
            y0 = all_copynumbers[:, -1]
            y0 = self.species.init_matrix(ndims=1, init_type="uniform",
                                          uniform_val=10).flatten()
            steady_state_result = integrate.solve_ivp(self.get_modelling_func(modeller),
                                                      (0, modeller.max_time),
                                                      y0=y0)
            logging.info(all_copynumbers[:, -1].flatten())
            if not steady_state_result.success:
                raise ValueError(
                    'Steady state could not be found through solve_ivp.')
            all_copynumbers = steady_state_result.y
        return all_copynumbers

    def model_circuit(self, modeller, all_copynumbers,
                      signal=None, signal_idx=None):
        modelling_func = self.get_modelling_func(modeller)
        modelling_func = partial(modelling_func, t=None)
        current_copynumbers = all_copynumbers[:, -1].flatten()
        all_copynumbers = np.concatenate((all_copynumbers, np.zeros(
            (self.species.data.size, modeller.max_time-1))
        ), axis=1)

        for tstep in range(0, modeller.max_time-1):
            dxdt = modelling_func(
                copynumbers=all_copynumbers[:, tstep]) * modeller.time_step

            current_copynumbers = np.add(dxdt, current_copynumbers).flatten()

            if signal is not None:
                current_copynumbers[signal_idx] = signal[tstep]
            current_copynumbers = zero_out_negs(current_copynumbers)
            all_copynumbers[:, tstep +
                            1] = current_copynumbers
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
                                          all_copynumbers=init_copynums,
                                          signal=signal.real_signal,
                                          signal_idx=signal.identities_idx)

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
