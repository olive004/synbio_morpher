from functools import partial
import logging
import os
import numpy as np
from scipy import integrate
from src.srv.io.results.result_writer import ResultWriter

from src.utils.misc.decorators import time_it
from src.utils.misc.numerical import zero_out_negs
from src.utils.misc.type_handling import flatten_nested_dict
from src.utils.signal.inputs import Signal
from src.srv.parameter_prediction.simulator import InteractionSimulator
from src.utils.system_definition.agnostic_system.base_system import BaseSystem
from src.utils.system_definition.agnostic_system.modelling import Deterministic


class SystemManager():

    def __init__(self, circuits) -> None:
        self.circuits = circuits


class CircuitModeller():

    def __init__(self, result_writer=None) -> None:
        if result_writer is None:
            self.result_writer = ResultWriter()
        else:
            self.result_writer = result_writer

    def init_circuit(self, circuit: BaseSystem):
        circuit = self.compute_interaction_strengths(circuit)
        circuit = self.find_steady_states(circuit)
        return circuit

    def get_modelling_func(self, modeller, circuit):
        return partial(modeller.dxdt_RNA, interactions=circuit.species.interactions,
                       creation_rates=circuit.species.creation_rates,
                       degradation_rates=circuit.species.degradation_rates,
                       num_samples=circuit.species.data.size
                       )

    @time_it
    def compute_interaction_strengths(self, circuit: BaseSystem):
        if not circuit.species.loaded_interactions:
            interactions = self.run_interaction_simulator(circuit,
                                                          circuit.species.data.data)
            circuit.species.interactions = interactions.matrix
            filename_addon = 'interactions'
            self.result_writer.output(
                out_type='csv', out_name=circuit.name, data=circuit.species.interactions_to_df(), overwrite=False,
                new_file=True, filename_addon=filename_addon, subfolder=filename_addon)
        return circuit

    def run_interaction_simulator(self, circuit, data):
        simulator = InteractionSimulator(
            circuit.simulator_args, circuit.simulator_choice)
        return simulator.run(data)

    def find_steady_states(self, circuit):
        modeller_steady_state = Deterministic(
            max_time=50, time_step=1)

        circuit.species.copynumbers = self.compute_steady_states(modeller_steady_state,
                                                                 circuit=circuit,
                                                                 use_solver='ivp')

        circuit.result_collector.add_result(circuit.species.copynumbers,
                                            name='steady_state',
                                            category='time_series',
                                            vis_func=modeller_steady_state.plot,
                                            **{'legend_keys': list(circuit.species.data.sample_names),
                                               'out_path': 'steady_state_plot'})
        steady_state_metrics = circuit.result_collector.get_result(
            key='steady_state').metrics
        circuit.species.steady_state_copynums = steady_state_metrics[
            'steady_state']['steady_states']
        return circuit

    def compute_steady_states(self, modeller, circuit: BaseSystem,
                              use_solver='naive'):
        all_copynumbers = circuit.species.copynumbers
        if use_solver == 'naive':
            self.model_circuit(modeller, all_copynumbers)
        elif use_solver == 'ivp':
            y0 = all_copynumbers[:, -1]
            steady_state_result = integrate.solve_ivp(self.get_modelling_func(modeller, circuit),
                                                      (0, modeller.max_time),
                                                      y0=y0)
            if not steady_state_result.success:
                raise ValueError(
                    'Steady state could not be found through solve_ivp.')
            all_copynumbers = steady_state_result.y
        return all_copynumbers

    def model_circuit(self, modeller, copynumbers, circuit: BaseSystem,
                      signal=None, signal_idx=None):
        modelling_func = self.get_modelling_func(modeller, circuit=circuit)
        modelling_func = partial(modelling_func, t=None)
        current_copynumbers = copynumbers[:, -1].flatten()
        copynumbers = np.concatenate((copynumbers, np.zeros(
            (circuit.species.data.size, modeller.max_time-1))
        ), axis=1)
        if signal is not None:
            copynumbers[signal_idx, :] = signal

        copynumbers = self.iterate_modelling_func(copynumbers, current_copynumbers, modelling_func,
                                                  max_time=modeller.max_time, time_step=modeller.time_step,
                                                  signal=signal, signal_idx=signal_idx)
        return copynumbers

    def iterate_modelling_func(self, copynumbers, init_copynumbers,
                               modelling_func, max_time, time_step,
                               signal=None, signal_idx=None):
        current_copynumbers = init_copynumbers
        for tstep in range(0, max_time-1):
            dxdt = modelling_func(
                copynumbers=copynumbers[:, tstep]) * time_step

            current_copynumbers = np.add(dxdt, current_copynumbers).flatten()

            if signal is not None:
                current_copynumbers[signal_idx] = signal[tstep]
            current_copynumbers = zero_out_negs(current_copynumbers)
            copynumbers[:, tstep +
                        1] = current_copynumbers
        return copynumbers

    def simulate_signal(self, circuit, signal: Signal = None):
        if signal is None:
            signal = circuit.signal
        signal_modeller = Deterministic(
            max_time=signal.total_time, time_step=1
        )
        new_copynumbers = self.model_circuit(signal_modeller,
                                             circuit.species.steady_state_copynums,
                                             circuit=circuit,
                                             signal=signal.real_signal,
                                             signal_idx=signal.identities_idx)

        circuit.species.copynumbers = np.concatenate(
            (circuit.species.copynumbers, new_copynumbers[:, 1:]), axis=1)
        circuit.result_collector.add_result(new_copynumbers,
                                            name='signal',
                                            category='time_series',
                                            vis_func=signal_modeller.plot,
                                            **{'legend_keys': list(circuit.species.data.sample_names),
                                               'out_path': 'signal_plot'})
        return circuit

    def wrap_mutations(self, circuit: BaseSystem, methods: dict, include_normal_run=True):
        mutation_dict = flatten_nested_dict(circuit.species.mutations.items())
        for i, (name, mutation) in enumerate(mutation_dict.items()):
            logging.info(f'Running methods on mutation {name}')
            if include_normal_run and i == 0:
                self.apply_to_circuit(circuit, methods)
            subcircuit = circuit.make_subsystem(name, mutation)
            self.result_writer.subdivide_writing(name)
            self.apply_to_circuit(subcircuit, methods)
        self.result_writer.unsubdivide()

    def apply_to_circuit(self, circuit: BaseSystem, methods: dict):
        for method, kwargs in methods.items():
            if hasattr(self, method):
                circuit = getattr(self, method)(circuit, **kwargs)

    def visualise_graph(self, circuit: BaseSystem, mode="pyvis", new_vis=False):
        circuit.refresh_graph()

        self.result_writer.visualise(circuit, mode, new_vis)

    def write_results(self, circuit, new_report=False):
        self.result_writer.write_all(circuit.results, new_report)
