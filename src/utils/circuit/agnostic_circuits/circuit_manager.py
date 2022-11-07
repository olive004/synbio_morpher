from copy import deepcopy
from typing import List, Tuple, Union
from functools import partial
import logging
import numpy as np
import jax
from scipy import integrate
from src.utils.results.analytics.timeseries import Timeseries
from src.utils.results.result_writer import ResultWriter

from src.utils.misc.numerical import make_dynamic_indexer, np_delete_axes, zero_out_negs
from src.utils.misc.type_handling import flatten_nested_dict
from src.utils.results.visualisation import VisODE
from src.utils.signal.inputs import Signal
from src.srv.parameter_prediction.simulator import SIMULATOR_UNITS, InteractionSimulator
from src.utils.circuit.agnostic_circuits.base_circuit import BaseCircuit
from src.utils.modelling.deterministic import Deterministic, simulate_signal_scan
from src.utils.modelling.base import Modeller


TEST_MODE = False


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

    def init_circuit(self, circuit: BaseCircuit):
        circuit = self.compute_interaction_strengths(circuit)
        circuit = self.find_steady_states(circuit)
        return circuit

    def make_modelling_func(self, modelling_func, circuit: BaseCircuit,
                            exclude_species_by_idx: Union[int, list] = None,
                            fixed_value: Timeseries(None).num_dtype = None,
                            fixed_value_idx: int = None):
        num_samples = circuit.species.data.size
        exclude_species_by_idx = self.process_exclude_species_by_idx(
            exclude_species_by_idx)

        interaction_binding_rates = circuit.species.interactions
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

        return partial(modelling_func, full_interactions=interaction_binding_rates,
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
    def compute_interaction_strengths(self, circuit: BaseCircuit):
        if not circuit.species.are_interactions_loaded:
            if not TEST_MODE:
                interactions = self.run_interaction_simulator(circuit,
                                                              circuit.species.data.data)
                circuit.species.eqconstants = interactions.eqconstants
                circuit.species.binding_rates_dissociation = interactions.binding_rates
                circuit.species.interactions = interactions.calculate_full_coupling_of_rates(
                    eqconstants=circuit.species.eqconstants
                )
                circuit.species.interaction_units = interactions.units
            else:
                logging.warning(
                    'RUNNING IN TEST MODE - interaction rates are fake.')
                circuit.species.eqconstants = np.random.rand(
                    circuit.species.size, circuit.species.size)
                circuit.species.binding_rates_dissociation = np.random.rand(
                    circuit.species.size, circuit.species.size)
                circuit.species.interactions = np.random.rand(
                    circuit.species.size, circuit.species.size)
                circuit.species.interaction_units = 'test'

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

    def run_interaction_simulator(self, circuit: BaseCircuit, data):
        simulator = InteractionSimulator(circuit.simulator_args)
        return simulator.run(data)

    def find_steady_states(self, circuit: BaseCircuit):
        modeller_steady_state = Deterministic(
            max_time=50, time_interval=0.1)

        circuit.species.copynumbers = self.compute_steady_states(modeller_steady_state,
                                                                 circuit=circuit,
                                                                 solver_type=self.steady_state_solver)

        circuit.result_collector.add_result(circuit.species.copynumbers,
                                            name='steady_states',
                                            category='time_series',
                                            vis_func=VisODE().plot,
                                            vis_kwargs={'t': np.arange(0, np.shape(circuit.species.copynumbers)[1]) *
                                                        modeller_steady_state.time_interval,
                                                        'legend': list(circuit.species.data.sample_names),
                                                        'out_type': 'svg'})
        steady_state_analytics = circuit.result_collector.get_result(
            key='steady_states').analytics
        circuit.species.steady_state_copynums = steady_state_analytics['steady_states']
        return circuit

    def compute_steady_states(self, modeller: Modeller, circuit: BaseCircuit,
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
            steady_state_result = integrate.solve_ivp(self.make_modelling_func(modeller.dxdt_RNA, circuit,
                                                                               exclude_species_by_idx),
                                                      (0, modeller.max_time),
                                                      y0=y0)
            if not steady_state_result.success:
                raise ValueError(
                    'Steady state could not be found through solve_ivp - possibly because units '
                    f'are in {circuit.species.interaction_units}. {SIMULATOR_UNITS}')
            copynumbers = steady_state_result.y
        return copynumbers

    def model_circuit(self, modeller: Modeller, init_copynumbers: np.ndarray, circuit: BaseCircuit,
                      signal: np.ndarray = None, signal_identity_idx: int = None,
                      exclude_species_by_idx: Union[list, int] = None):
        assert np.shape(init_copynumbers)[circuit.species.species_axis] == np.shape(
            init_copynumbers)[circuit.species.species_axis], 'Please only use 1-d ' \
            f'initial copynumbers instead of {np.shape(init_copynumbers)}'

        modelling_func = partial(self.make_modelling_func(
            modeller.dxdt_RNA, circuit, exclude_species_by_idx), t=None)
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
        """ Loop the modelling function.
        IMPORTANT! Modeller already includes the dt, or the length of the time step taken. """

        current_copynumbers = init_copynumbers.flatten()
        if signal is not None:
            for tstep in range(0, max_time-1):
                dxdt = modelling_func(
                    copynumbers=copynumbers[:, tstep])

                current_copynumbers = np.add(
                    dxdt, current_copynumbers).flatten()

                current_copynumbers[signal_idx] = signal[tstep]
                copynumbers[:, tstep +
                            1] = zero_out_negs(current_copynumbers)
        else:
            for tstep in range(0, max_time-1):
                dxdt = modelling_func(
                    copynumbers=copynumbers[:, tstep])

                current_copynumbers = np.add(
                    dxdt, current_copynumbers).flatten()

                copynumbers[:, tstep +
                            1] = zero_out_negs(current_copynumbers)
        return copynumbers

    # @time_it
    def simulate_signal(self, circuit: BaseCircuit, signal: Signal = None, save_numerical_vis_data: bool = False,
                        use_solver: str = 'naive', ref_circuit: BaseCircuit = None,
                        time_interval=1):
        if signal is None:
            circuit.signal.update_time_interval(time_interval)
            signal = circuit.signal

        modeller_signal = Deterministic(
            max_time=signal.total_time, time_interval=time_interval
        )
        if circuit.result_collector.get_result('steady_states'):
            steady_states = deepcopy(circuit.species.steady_state_copynums)
        else:
            steady_states = self.compute_steady_states(Deterministic(
                max_time=50/time_interval, time_interval=time_interval),
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
                        modelling_func=modeller_signal.dxdt_RNA,
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
        elif use_solver == 'jax':
            new_copynumbers = simulate_signal_scan(
                copynumbers=steady_states.flatten(), time=np.arange(modeller_signal.max_time),
                full_interactions=circuit.species.interactions,
                creation_rates=circuit.species.creation_rates.flatten(),
                degradation_rates=circuit.species.degradation_rates.flatten(),
                identity_matrix=np.identity(
                    circuit.species.size),
                signal=signal.real_signal, signal_idx=circuit.species.identities.get(
                    'input'),
                one_step_func=modeller_signal.dxdt_RNA_jnp)
            new_copynumbers = np.array(new_copynumbers[1]).T

        circuit.species.copynumbers = np.concatenate(
            (circuit.species.copynumbers, new_copynumbers[make_dynamic_indexer({
                circuit.species.species_axis: slice(0, np.shape(new_copynumbers)[circuit.species.species_axis]),
                circuit.species.time_axis: slice(1, np.shape(new_copynumbers)[
                                                 circuit.species.species_axis])
            })]), axis=circuit.species.time_axis)
        if ref_circuit is None or ref_circuit == circuit:
            ref_circuit_signal = None
        else:
            ref_circuit_result = ref_circuit.result_collector.get_result(
                'signal')
            ref_circuit_signal = None if ref_circuit_result is None else ref_circuit_result.data

        t = np.arange(0, np.shape(new_copynumbers)[
            1]) * modeller_signal.time_interval
        circuit.result_collector.add_result(new_copynumbers,
                                            name='signal',
                                            category='time_series',
                                            vis_func=VisODE().plot,
                                            save_numerical_vis_data=save_numerical_vis_data,
                                            vis_kwargs={'t': t,
                                                        'legend': list(circuit.species.data.sample_names),
                                                        'out_type': 'svg'},
                                            analytics_kwargs={'signal_idx': signal.identities_idx,
                                                              'ref_circuit_signal': ref_circuit_signal})
        return circuit

    def simulate_signal_batch(self, circuits: List[Tuple[str, BaseCircuit]], circuit_idx: int = 0,
                              save_numerical_vis_data: bool = False,
                              ref_circuit: BaseCircuit = None,
                              time_interval=1, batch=True):
        names = list([n for n, c in circuits])
        circuits = list([c for n, c in circuits])
        if circuit_idx < len(circuits):
            signal = circuits[circuit_idx].signal
            signal.update_time_interval(time_interval)
        else:
            raise ValueError(
                f'The chosen circuit index {circuit_idx} exceeds the circuit list (len {len(circuits)}')

        modeller_signal = Deterministic(
            max_time=signal.total_time, time_interval=time_interval
        )

        # Batch
        t = np.arange(modeller_signal.max_time)
        b_starting_copynumbers = np.array(
            [c.species.steady_state_copynums.flatten() for c in circuits])
        b_interactions = np.array([c.species.interactions for c in circuits])
        # b_creation_rates = np.array(
        #     [c.species.creation_rates.flatten() for c in circuits])
        # b_degradation_rates = np.array(
        #     [c.species.degradation_rates.flatten() for c in circuits])

        b_new_copynumbers = jax.vmap(partial(simulate_signal_scan,
                                             # b_new_copynumbers = partial(simulate_signal_scan,
                                             time=t,
                                             creation_rates=circuits[circuit_idx].species.creation_rates.flatten(
                                             ),
                                             degradation_rates=circuits[circuit_idx].species.degradation_rates.flatten(
                                             ),
                                             identity_matrix=np.identity(
                                                 circuits[circuit_idx].species.size),
                                             signal=signal.real_signal, signal_idx=circuits[circuit_idx].species.identities.get(
                                                 'input'),
                                             one_step_func=modeller_signal.dxdt_RNA_jnp))(b_starting_copynumbers, full_interactions=b_interactions)
        b_new_copynumbers = np.array(b_new_copynumbers[1])
        if np.shape(b_new_copynumbers)[1] != circuits[circuit_idx].species.size and np.shape(b_new_copynumbers)[-1] == circuits[circuit_idx].species.size:
            b_new_copynumbers = np.swapaxes(b_new_copynumbers, 1, 2)
        
        # Apply to all circuits
        if ref_circuit is None or ref_circuit == circuit:
            ref_circuit_signal = None
        else:
            ref_circuit_result = ref_circuit.result_collector.get_result(
                'signal')
            ref_circuit_signal = None if ref_circuit_result is None else ref_circuit_result.data
        for i, circuit in enumerate(circuits):
            circuits[i].result_collector.add_result(b_new_copynumbers[i],
                                                    name='signal',
                                                    category='time_series',
                                                    vis_func=VisODE().plot,
                                                    save_numerical_vis_data=save_numerical_vis_data,
                                                    vis_kwargs={'t': t,
                                                                'legend': list(circuit.species.data.sample_names),
                                                                'out_type': 'svg'},
                                                    analytics_kwargs={'signal_idx': signal.identities_idx,
                                                                      'ref_circuit_signal': ref_circuit_signal})
        return list(zip(names, circuits))

    # @time_it
    def wrap_mutations(self, circuit: BaseCircuit, methods: dict, include_normal_run=True,
                       write_to_subsystem=False):
        if write_to_subsystem:
            self.result_writer.subdivide_writing(circuit.name)
        mutation_dict = flatten_nested_dict(circuit.species.mutations.items())
        # logging.info(
        #     f'Running functions {methods} on circuit with {len(mutation_dict)} items.')

        self.result_writer.subdivide_writing(
            'mutations', safe_dir_change=False)
        for i, (name, mutation) in enumerate(mutation_dict.items()):
            # logging.info(f'Running methods on mutation {name} ({i})')
            if include_normal_run and i == 0:
                self.result_writer.unsubdivide_last_dir()
                circuit = self.apply_to_circuit(
                    circuit, methods, ref_circuit=circuit)
                self.result_writer.subdivide_writing(
                    'mutations', safe_dir_change=False)
            subcircuit = circuit.make_subcircuit(name, mutation)
            self.result_writer.subdivide_writing(name, safe_dir_change=False)
            self.apply_to_circuit(subcircuit, methods, ref_circuit=circuit)
            self.result_writer.unsubdivide_last_dir()
        self.result_writer.unsubdivide()

    def batch_mutations(self, circuit: BaseCircuit, methods: dict, include_normal_run=True,
                        write_to_subsystem=False):
        if write_to_subsystem:
            self.result_writer.subdivide_writing(circuit.name)

        mutation_dict = flatten_nested_dict(circuit.species.mutations.items())
        subcircuits = [(name, circuit.make_subcircuit(name, mutation))
                       for name, mutation in mutation_dict.items()]
        if include_normal_run:
            subcircuits.insert(0, ('ref_circuit', circuit))

        for method, kwargs in methods.items():
            if kwargs.get('batch') == True:
                if hasattr(self, method):
                    subcircuits = getattr(self, method)(subcircuits, **kwargs)
                else:
                    logging.warning(
                        f'Could not find method @{method} in class {self}')
            else:
                self.result_writer.subdivide_writing(
                    'mutations', safe_dir_change=False)
                for i, (name, subcircuit) in enumerate(subcircuits):
                    if include_normal_run and i == 0:
                        self.result_writer.unsubdivide_last_dir()
                        circuit = self.apply_to_circuit(
                            subcircuit, {method: kwargs}, ref_circuit=circuit)
                        self.result_writer.subdivide_writing(
                            'mutations', safe_dir_change=False)
                        subcircuit = circuit
                        continue

                    self.result_writer.subdivide_writing(
                        name, safe_dir_change=False)
                    subcircuit = self.apply_to_circuit(
                        subcircuit, {method: kwargs}, ref_circuit=circuit)
                    self.result_writer.unsubdivide_last_dir()
                    subcircuits[i] = (name, subcircuit)
                self.result_writer.unsubdivide_last_dir()

        self.result_writer.unsubdivide()

    def apply_to_circuit(self, circuit: BaseCircuit, _methods: dict, ref_circuit: BaseCircuit):
        methods = deepcopy(_methods)
        for method, kwargs in methods.items():
            if hasattr(self, method):
                if 'ref_circuit' in kwargs.keys():
                    kwargs.update({'ref_circuit': ref_circuit})
                circuit = getattr(self, method)(circuit, **kwargs)
            else:
                logging.warning(
                    f'Could not find method @{method} in class {self}')
        return circuit

    def visualise_graph(self, circuit: BaseCircuit, mode="pyvis", new_vis=False):
        self.result_writer.visualise_graph(circuit, mode, new_vis)

    def write_results(self, circuit, new_report: bool = False, no_visualisations: bool = False,
                      only_numerical: bool = False, no_numerical: bool = False):
        self.result_writer.write_all(
            circuit, new_report, no_visualisations=no_visualisations, only_numerical=only_numerical,
            no_numerical=no_numerical)
        return circuit
