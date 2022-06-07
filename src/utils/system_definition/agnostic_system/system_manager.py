from copy import deepcopy
from typing import Union
from functools import partial
import logging
import os
import numpy as np
from scipy import integrate
from src.srv.io.results.result_writer import ResultWriter

from src.utils.misc.decorators import time_it
from src.utils.misc.numerical import make_dynamic_indexer, np_delete_axes, zero_out_negs
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

    def get_modelling_func(self, modeller: Deterministic, circuit: BaseSystem, exclude_species_by_idx: Union[int, list] = None):
        num_samples = circuit.species.data.size
        exclude_species_by_idx = self.process_exclude_species_by_idx(
            exclude_species_by_idx)

        interactions = circuit.species.interactions
        creation_rates = circuit.species.creation_rates
        degradation_rates = circuit.species.degradation_rates

        if exclude_species_by_idx is not None:
            num_samples -= len(exclude_species_by_idx)
            for exclude in exclude_species_by_idx:
                if exclude is None:
                    continue
                interactions = np_delete_axes(
                    interactions, exclude, axes=[circuit.species.species_axis, circuit.species.time_axis])
                creation_rates = np.delete(
                    creation_rates, exclude, axis=circuit.species.species_axis)
                degradation_rates = np.delete(
                    degradation_rates, exclude, axis=circuit.species.species_axis)

        return partial(modeller.dxdt_RNA, interactions=interactions,
                       creation_rates=creation_rates,
                       degradation_rates=degradation_rates,
                       num_samples=num_samples
                       )

    @staticmethod
    def process_exclude_species_by_idx(exclude_species_by_idx):
        if type(exclude_species_by_idx) == int:
            exclude_species_by_idx = [exclude_species_by_idx]
        return exclude_species_by_idx

    def compute_interaction_strengths(self, circuit: BaseSystem):
        if not circuit.species.loaded_interactions:
            interactions = self.run_interaction_simulator(circuit,
                                                          circuit.species.data.data)
            circuit.species.interactions = interactions.matrix
            circuit.species.interaction_units = interactions.units

            filename_addon = 'interactions'
            self.result_writer.output(
                out_type='csv', out_name=circuit.name, data=circuit.species.interactions_to_df(), overwrite=False,
                new_file=True, filename_addon=filename_addon, subfolder=filename_addon)
        return circuit

    def run_interaction_simulator(self, circuit: BaseSystem, data):
        simulator = InteractionSimulator(
            circuit.simulator_args)
        return simulator.run(data)

    def find_steady_states(self, circuit: BaseSystem):
        modeller_steady_state = Deterministic(
            max_time=50, time_step=1)

        circuit.species.copynumbers = self.compute_steady_states(modeller_steady_state,
                                                                 circuit=circuit,
                                                                 use_solver='ivp')

        circuit.result_collector.add_result(circuit.species.copynumbers,
                                            name='steady_state',
                                            category='time_series',
                                            vis_func=modeller_steady_state.plot,
                                            **{'legend': list(circuit.species.data.sample_names),
                                               'out_type': 'png'})
        steady_state_analytics = circuit.result_collector.get_result(
            key='steady_state').analytics
        circuit.species.steady_state_copynums = steady_state_analytics[
            'steady_state']['steady_states']
        return circuit

    def compute_steady_states(self, modeller, circuit: BaseSystem,
                              use_solver: str = 'naive',
                              exclude_species_by_idx: Union[int, list] = None):
        copynumbers = circuit.species.copynumbers[:, -1]
        if exclude_species_by_idx is not None:
            exclude_species_by_idx = self.process_exclude_species_by_idx(
                exclude_species_by_idx)
            for excluded in exclude_species_by_idx:
                copynumbers = np.delete(
                    copynumbers, excluded, axis=circuit.species.species_axis)
        copynumbers = np.reshape(copynumbers, (np.shape(copynumbers)[0], 1))

        idxs = make_dynamic_indexer({
            circuit.species.species_axis: slice(0, np.shape(copynumbers)[circuit.species.species_axis], 1),
            circuit.species.time_axis: -1})
        if use_solver == 'naive':
            copynumbers = self.model_circuit(
                modeller, copynumbers, circuit=circuit, exclude_species_by_idx=exclude_species_by_idx)
            copynumbers = copynumbers
        elif use_solver == 'ivp':
            y0 = copynumbers[idxs]
            steady_state_result = integrate.solve_ivp(self.get_modelling_func(modeller, circuit,
                                                                              exclude_species_by_idx),
                                                      (0, modeller.max_time),
                                                      y0=y0)
            if not steady_state_result.success:
                raise ValueError(
                    'Steady state could not be found through solve_ivp - possibly because units '
                    f'are in {circuit.species.interaction_units}.')
            copynumbers = steady_state_result.y        
        return copynumbers

    def model_circuit(self, modeller, init_copynumbers: np.ndarray, circuit: BaseSystem,
                      signal: np.ndarray = None, signal_identity_idx: int = None,
                      exclude_species_by_idx: Union[list, int] = None):
        assert np.shape(init_copynumbers)[circuit.species.species_axis] == np.shape(
            init_copynumbers)[circuit.species.species_axis], 'Please only use 1-d '
        logging.info(np.shape(init_copynumbers))
        modelling_func = self.get_modelling_func(
            modeller, circuit, exclude_species_by_idx)
        modelling_func = partial(modelling_func, t=None)
        copynumbers = np.concatenate((init_copynumbers, np.zeros(
            (np.shape(init_copynumbers)[circuit.species.species_axis], modeller.max_time-1))
        ), axis=1)
        if signal is not None:
            if not np.shape(init_copynumbers)[circuit.species.species_axis] == circuit.species.data.size:
                logging.warning('Shape of copynumbers is not consistent with number of species - make sure '
                                f'that the index for the species serving as the signal ({signal_identity_idx}) '
                                'is not misaligned due to exclusion of another species.')
            init_copynumbers[signal_identity_idx] = signal[0]

        logging.info(np.shape(init_copynumbers))
        copynumbers = self.iterate_modelling_func(copynumbers, init_copynumbers, modelling_func,
                                                  max_time=modeller.max_time, time_step=modeller.time_step,
                                                  signal=signal, signal_idx=signal_identity_idx)
        return copynumbers

    def iterate_modelling_func(self, copynumbers, init_copynumbers,
                               modelling_func, max_time, time_step,
                               signal=None, signal_idx: int = None):
        current_copynumbers = init_copynumbers.flatten()
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

    def simulate_signal(self, circuit: BaseSystem, signal: Signal = None, save_numerical_vis_data=False, re_calculate_steadystate=True):
        if signal is None:
            signal = circuit.signal
        modeller_signal = Deterministic(
            max_time=signal.total_time, time_step=1
        )
        if re_calculate_steadystate:
            steady_states = self.compute_steady_states(Deterministic(
                max_time=500, time_step=1),
                circuit=circuit,
                use_solver='naive',
                exclude_species_by_idx=signal.identities_idx)
            steady_states = np.insert(steady_states, signal.identities_idx,
                                      signal.real_signal[0], axis=circuit.species.species_axis)
            logging.info(f'\n\n\n\n\n\n{steady_states}')
        else:
            steady_states = deepcopy(circuit.species.steady_state_copynums)

        logging.info(np.shape(steady_states))
        logging.info(steady_states)

        new_copynumbers = self.model_circuit(modeller_signal,
                                             steady_states,
                                             circuit=circuit,
                                             signal=signal.real_signal,
                                             signal_identity_idx=signal.identities_idx)

        circuit.species.copynumbers = np.concatenate(
            (circuit.species.copynumbers, new_copynumbers[:, 1:]), axis=1)
        circuit.result_collector.add_result(new_copynumbers,
                                            name='signal',
                                            category='time_series',
                                            vis_func=modeller_signal.plot,
                                            save_numerical_vis_data=save_numerical_vis_data,
                                            **{'legend': list(circuit.species.data.sample_names),
                                               'out_type': 'png'})
        return circuit

    @time_it
    def wrap_mutations(self, circuit: BaseSystem, methods: dict, include_normal_run=True,
                       write_to_subsystem=False):
        if write_to_subsystem:
            self.result_writer.subdivide_writing(circuit.name)
        mutation_dict = flatten_nested_dict(circuit.species.mutations.items())
        for i, (name, mutation) in enumerate(mutation_dict.items()):
            # logging.info(f'Running methods on mutation {name}')
            if include_normal_run and i == 0:
                self.apply_to_circuit(circuit, methods)
            subcircuit = circuit.make_subsystem(name, mutation)
            self.result_writer.subdivide_writing(name, safe_dir_change=False)
            self.apply_to_circuit(subcircuit, methods)
            self.result_writer.unsubdivide_last_dir()
        self.result_writer.unsubdivide()

    def apply_to_circuit(self, circuit: BaseSystem, methods: dict):
        for method, kwargs in methods.items():
            if hasattr(self, method):
                circuit = getattr(self, method)(circuit, **kwargs)
            else:
                logging.warning(
                    f'Could not find method @{method} in class {self}')

    def visualise_graph(self, circuit: BaseSystem, mode="pyvis", new_vis=False):
        self.result_writer.visualise_graph(circuit, mode, new_vis)

    def write_results(self, circuit, new_report: bool = False, no_visualisations: bool = False,
                      only_numerical: bool = False):
        self.result_writer.write_all(
            circuit, new_report, no_visualisations=no_visualisations, only_numerical=only_numerical)
