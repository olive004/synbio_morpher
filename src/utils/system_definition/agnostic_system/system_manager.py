from copy import deepcopy
from typing import Union
from functools import partial
import logging
import os
import numpy as np
from scipy import integrate
from src.utils.misc.units import per_mol_to_per_molecules
from src.utils.results.analytics.timeseries import Timeseries
from src.utils.results.result_writer import ResultWriter

from src.utils.misc.decorators import time_it
from src.utils.misc.numerical import make_dynamic_indexer, np_delete_axes, zero_out_negs
from src.utils.misc.type_handling import flatten_nested_dict
from src.utils.signal.inputs import Signal
from src.srv.parameter_prediction.simulator import MIN_INTERACTION_EQCONSTANT, SIMULATOR_UNITS, InteractionSimulator
from src.utils.system_definition.agnostic_system.base_system import BaseSystem
from src.utils.system_definition.agnostic_system.modelling import Deterministic


class SystemManager():

    def __init__(self, circuits) -> None:
        self.circuits = circuits


class CircuitModeller():

    def __init__(self, result_writer=None, config: dict = {}) -> None:
        self.steady_state_solver = config.get("steady_state_solver", 'ivp')

        if result_writer is None:
            self.result_writer = ResultWriter()
        else:
            self.result_writer = result_writer

    def init_circuit(self, circuit: BaseSystem):
        circuit = self.compute_interaction_strengths(circuit)
        circuit = self.find_steady_states(circuit)
        return circuit

    def make_modelling_func(self, modeller: Deterministic, circuit: BaseSystem,
                            exclude_species_by_idx: Union[int, list] = None,
                            fixed_value: Timeseries(None).num_dtype = None,
                            fixed_value_idx: int = None):
        num_samples = circuit.species.data.size
        exclude_species_by_idx = self.process_exclude_species_by_idx(
            exclude_species_by_idx)

        interaction_binding_rates = circuit.species.interactions
        interaction_binding_rates[interaction_binding_rates < MIN_INTERACTION_EQCONSTANT] = 0
        creation_rates = circuit.species.creation_rates
        degradation_rates = circuit.species.degradation_rates

        if exclude_species_by_idx is not None:
            num_samples -= len(exclude_species_by_idx)
            for exclude in exclude_species_by_idx:
                if exclude is None:
                    continue
                interaction_binding_rates = np_delete_axes(
                    interaction_binding_rates, exclude, axes=[circuit.species.species_axis, circuit.species.time_axis])
                creation_rates = np.delete(
                    creation_rates, exclude, axis=circuit.species.species_axis)
                degradation_rates = np.delete(
                    degradation_rates, exclude, axis=circuit.species.species_axis)

        return partial(modeller.dxdt_RNA, full_interactions=interaction_binding_rates,
                       creation_rates=creation_rates.flatten(),
                       degradation_rates=degradation_rates.flatten(),
                       signal=fixed_value,
                       signal_idx=fixed_value_idx,
                       identity_matrix=np.identity(num_samples)
                       )

    @staticmethod
    def process_exclude_species_by_idx(exclude_species_by_idx):
        if type(exclude_species_by_idx) == int:
            exclude_species_by_idx = [exclude_species_by_idx]
        return exclude_species_by_idx

    # @time_it
    def compute_interaction_strengths(self, circuit: BaseSystem):
        if not circuit.species.loaded_interactions:
            interactions = self.run_interaction_simulator(circuit,
                                                          circuit.species.data.data)
            circuit.species.eqconstants = interactions.eqconstants
            circuit.species.binding_rates_dissociation = interactions.binding_rates
            circuit.species.interactions = interactions.calculate_full_coupling_of_rates(
                degradation_rates=circuit.species.degradation_rates.flatten()
            )
            circuit.species.interaction_units = interactions.units

            # TODO: In the InteractionMatrix, put these addons better somehow
            filename_addon_eqconstants = 'eqconstants'
            filename_addon_binding_rates = 'binding_rates'
            filename_addon_coupled_rates = 'interactions'
            for interaction_matrix, filename_addon in zip(
                [circuit.species.eqconstants, circuit.species.binding_rates_dissociation, 
                circuit.species.interactions],
                [filename_addon_eqconstants, filename_addon_binding_rates,
                filename_addon_coupled_rates]
            ):
                self.result_writer.output(
                    out_type='csv', out_name=circuit.name, data=circuit.species.interactions_to_df(
                        interaction_matrix), overwrite=False,
                    new_file=True, filename_addon=filename_addon, subfolder=filename_addon)
        return circuit

    def run_interaction_simulator(self, circuit: BaseSystem, data):
        simulator = InteractionSimulator(circuit.simulator_args)
        return simulator.run(data)

    def find_steady_states(self, circuit: BaseSystem):
        modeller_steady_state = Deterministic(
            max_time=50, time_step=0.1)

        circuit.species.copynumbers = self.compute_steady_states(modeller_steady_state,
                                                                 circuit=circuit,
                                                                 solver_type=self.steady_state_solver)

        circuit.result_collector.add_result(circuit.species.copynumbers,
                                            name='steady_states',
                                            category='time_series',
                                            vis_func=modeller_steady_state.plot,
                                            vis_kwargs={'legend': list(circuit.species.data.sample_names),
                                                        'out_type': 'png'})
        steady_state_analytics = circuit.result_collector.get_result(
            key='steady_states').analytics
        circuit.species.steady_state_copynums = steady_state_analytics['steady_states']
        return circuit

    def compute_steady_states(self, modeller, circuit: BaseSystem,
                              solver_type: str = 'naive',
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

        if solver_type == 'naive':
            copynumbers = self.model_circuit(
                modeller, copynumbers, circuit=circuit, exclude_species_by_idx=exclude_species_by_idx)
            copynumbers = copynumbers

        elif solver_type == 'ivp':

            if circuit.species.interaction_units == SIMULATOR_UNITS['IntaRNA']['energy']:
                logging.warning(f'Interactions in units of {circuit.species.interaction_units} may not be suitable for '
                                'solving with IVP')
            y0 = copynumbers[idxs]
            steady_state_result = integrate.solve_ivp(self.make_modelling_func(modeller, circuit,
                                                                               exclude_species_by_idx),
                                                      (0, modeller.max_time),
                                                      y0=y0)
            if not steady_state_result.success:
                raise ValueError(
                    'Steady state could not be found through solve_ivp - possibly because units '
                    f'are in {circuit.species.interaction_units}. {SIMULATOR_UNITS}')
            copynumbers = steady_state_result.y
        return copynumbers

    def model_circuit(self, modeller, init_copynumbers: np.ndarray, circuit: BaseSystem,
                      signal: np.ndarray = None, signal_identity_idx: int = None,
                      exclude_species_by_idx: Union[list, int] = None):
        assert np.shape(init_copynumbers)[circuit.species.species_axis] == np.shape(
            init_copynumbers)[circuit.species.species_axis], 'Please only use 1-d ' \
            f'initial copynumbers instead of {np.shape(init_copynumbers)}'

        modelling_func = partial(self.make_modelling_func(
            modeller, circuit, exclude_species_by_idx), t=None)
        copynumbers = np.concatenate((init_copynumbers, np.zeros(
            (np.shape(init_copynumbers)[circuit.species.species_axis], modeller.max_time-1))
        ), axis=1)

        if signal is not None:
            if not np.shape(init_copynumbers)[circuit.species.species_axis] == circuit.species.data.size:
                logging.warning('Shape of copynumbers is not consistent with number of species - make sure '
                                f'that the index for the species serving as the signal ({signal_identity_idx}) '
                                'is not misaligned due to exclusion of another species.')
            init_copynumbers[signal_identity_idx] = signal[0]

        copynumbers = self.iterate_modelling_func(copynumbers, init_copynumbers, modelling_func,
                                                  max_time=modeller.max_time,
                                                  signal=signal, signal_idx=signal_identity_idx)
        return copynumbers

    # @time_it
    def iterate_modelling_func(self, copynumbers, init_copynumbers,
                               modelling_func, max_time,
                               signal=None, signal_idx: int = None):
        current_copynumbers = init_copynumbers.flatten()
        if signal is not None:
            for tstep in range(0, max_time-1):
                dxdt = modelling_func(
                    copynumbers=copynumbers[:, tstep])

                current_copynumbers = np.add(
                    dxdt, current_copynumbers).flatten()

                current_copynumbers[signal_idx] = signal[tstep]
                current_copynumbers = zero_out_negs(current_copynumbers)
                copynumbers[:, tstep +
                            1] = zero_out_negs(current_copynumbers)
        else:
            for tstep in range(0, max_time-1):
                dxdt = modelling_func(
                    copynumbers=copynumbers[:, tstep]) * time_step

                current_copynumbers = np.add(
                    dxdt, current_copynumbers).flatten()

                if signal is not None:
                    current_copynumbers[signal_idx] = signal[tstep]
                current_copynumbers = zero_out_negs(current_copynumbers)
                copynumbers[:, tstep +
                            1] = current_copynumbers
        return copynumbers

    # @time_it
    def simulate_signal(self, circuit: BaseSystem, signal: Signal = None, save_numerical_vis_data: bool = False,
                        use_old_steadystates: bool = False, use_solver: str = 'naive'):
        time_step = 1
        if signal is None:
            # circuit.signal.update_time_interval(time_step)
            signal = circuit.signal
        max_time = int(signal.total_time / time_step)

        modeller_signal = Deterministic(
            max_time=max_time, time_step=time_step
        )
        if use_old_steadystates:
            steady_states = deepcopy(circuit.species.steady_state_copynums)
        else:
            steady_states = self.compute_steady_states(Deterministic(
                max_time=50/time_step, time_step=time_step),
                circuit=circuit,
                solver_type=self.steady_state_solver,
                exclude_species_by_idx=signal.identities_idx)
            steady_states = steady_states[make_dynamic_indexer({
                circuit.species.species_axis: slice(np.shape(steady_states)[circuit.species.species_axis]),
                circuit.species.time_axis: -1
            })]
            steady_states = np.insert(steady_states, signal.identities_idx,
                                      signal.real_signal[0], axis=circuit.species.species_axis)

        steady_states = steady_states.reshape(make_dynamic_indexer({
            circuit.species.species_axis: np.shape(steady_states)[circuit.species.species_axis],
            circuit.species.time_axis: 1
        }))
        if use_solver == 'naive':
            new_copynumbers = self.model_circuit(modeller_signal,
                                                 steady_states,
                                                 circuit=circuit,
                                                 signal=signal.real_signal,
                                                 signal_identity_idx=signal.identities_idx)

        elif use_solver == 'ivp':
            logging.warning(
                'Solving steady state with ivp not fully implemented yet.')
            init_copynumbers = steady_states
            new_copynumbers = np.concatenate((init_copynumbers, np.zeros(
                (np.shape(init_copynumbers)[circuit.species.species_axis], modeller_signal.max_time-1))
            ), axis=1)
            for signal_component, time_start, time_end in signal.summarized_signal:
                time_span = time_end - time_start
                y0 = steady_states.flatten()
                steady_state_result = integrate.solve_ivp(
                    self.make_modelling_func(
                        modeller=modeller_signal,
                        circuit=circuit,
                        fixed_value=signal_component,
                        fixed_value_idx=circuit.species.identities.get('input')),
                    (0, modeller_signal.max_time),
                    y0=y0)
                if not steady_state_result.success:
                    raise ValueError(
                        'Steady state could not be found through solve_ivp - possibly because units '
                        f'are in {circuit.species.interaction_units}.')
                expanded_steady_states = np.concatenate(
                    [steady_state_result.y,
                     np.repeat(np.expand_dims(steady_state_result.y[:, -1], axis=circuit.species.time_axis),
                               time_span - np.shape(steady_state_result.y)[1])])
                new_copynumbers[:,
                                time_start:time_end] = expanded_steady_states

        circuit.species.copynumbers = np.concatenate(
            (circuit.species.copynumbers, new_copynumbers[make_dynamic_indexer({
                circuit.species.species_axis: slice(0, np.shape(new_copynumbers)[circuit.species.species_axis]),
                circuit.species.time_axis: slice(1, np.shape(new_copynumbers)[
                                                 circuit.species.species_axis])
            })]), axis=circuit.species.time_axis)
        circuit.result_collector.add_result(new_copynumbers,
                                            name='signal',
                                            category='time_series',
                                            vis_func=modeller_signal.plot,
                                            save_numerical_vis_data=save_numerical_vis_data,
                                            vis_kwargs={'legend': list(circuit.species.data.sample_names),
                                                        'out_type': 'png'},
                                            analytics_kwargs={'signal_idx': signal.identities_idx})
        return circuit

    # @time_it
    def wrap_mutations(self, circuit: BaseSystem, methods: dict, include_normal_run=True,
                       write_to_subsystem=False):
        if write_to_subsystem:
            self.result_writer.subdivide_writing(circuit.name)
        mutation_dict = flatten_nested_dict(circuit.species.mutations.items())
        # logging.info(
        #     f'Running functions {methods} on circuit with {len(mutation_dict)} items.')

        self.result_writer.subdivide_writing('mutations', safe_dir_change=False)
        for i, (name, mutation) in enumerate(mutation_dict.items()):
            # logging.info(f'Running methods on mutation {name} ({i})')
            if include_normal_run and i == 0:
                self.result_writer.unsubdivide_last_dir()
                self.apply_to_circuit(circuit, methods)
                self.result_writer.subdivide_writing('mutations', safe_dir_change=False)
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
