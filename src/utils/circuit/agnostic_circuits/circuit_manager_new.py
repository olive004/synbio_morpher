from copy import deepcopy
from typing import List, Tuple
from functools import partial
import logging
import numpy as np
import jax
from scipy import integrate
from bioreaction.model.data_containers import Species
from bioreaction.simulation.simfuncs.basic_de import bioreaction_sim

from src.utils.results.analytics.timeseries import Timeseries
from src.utils.results.result_writer import ResultWriter
from src.utils.circuit.agnostic_circuits.circuit_new import interactions_to_df
from src.srv.parameter_prediction.simulator import SIMULATOR_UNITS
from src.srv.parameter_prediction.interactions import MolecularInteractions, InteractionData, InteractionSimulator
from src.utils.misc.numerical import make_dynamic_indexer, invert_onehot, zero_out_negs
from src.utils.misc.type_handling import flatten_nested_dict, flatten_listlike, get_unique
from src.utils.misc.helper import vanilla_return
from src.utils.results.visualisation import VisODE
from src.utils.signal.signals_new import Signal
from src.utils.circuit.agnostic_circuits.circuit_new import Circuit
from src.utils.modelling.deterministic import Deterministic, simulate_signal_scan, bioreaction_sim_full
from src.utils.modelling.base import Modeller


TEST_MODE = True


class CircuitModeller():

    def __init__(self, result_writer=None, config: dict = {}) -> None:
        self.steady_state_solver = config.get("steady_state_solver", 'ivp')
        self.simulator_args = config['interaction_simulator']
        self.simulation = config.get('simulation', {})
        self.result_writer = ResultWriter() if result_writer is None else result_writer

    def init_circuit(self, circuit: Circuit):
        circuit = self.compute_interaction_strengths(circuit)
        circuit = self.find_steady_states(circuit)
        return circuit

    def make_modelling_func(self, modelling_func, circuit: Circuit,
                            fixed_value = None,
                            fixed_value_idx: int = None):

        return partial(modelling_func, full_interactions=circuit.interactions.coupled_binding_rates,
                       creation_rates=circuit.qreactions.reactions.forward_rates.flatten(),
                       degradation_rates=circuit.qreactions.reactions.reverse_rates.flatten(),
                       signal=fixed_value,
                       signal_idx=fixed_value_idx,
                       identity_matrix=np.identity(circuit.circuit_size)
                       )

    def update_species_simulated_rates(self, circuit: Circuit,
                                       interactions: MolecularInteractions) -> Circuit:
        for r in circuit.model.reactions:
            if len(r.input) == 2:
                si = r.input[0]
                sj = r.input[1]
                r.forward_rate = interactions.binding_rates_association[circuit.model.species.index(
                    si), circuit.model.species.index(sj)]
                r.reverse_rate = interactions.binding_rates_dissociation[circuit.model.species.index(
                    si), circuit.model.species.index(sj)]

    # @time_it
    def compute_interaction_strengths(self, circuit: Circuit):
        if circuit.species_state == 'uninitialised':
            if not TEST_MODE:
                interactions = self.run_interaction_simulator(
                    get_unique(flatten_listlike([r.input for r in circuit.model.reactions])))
                circuit = self.update_species_simulated_rates(
                    circuit, interactions.interactions)
                circuit.interactions = interactions.interactions
            else:
                logging.warning(
                    'RUNNING IN TEST MODE - interaction rates are fake.')
                eqconstants = np.random.rand(
                    circuit.circuit_size, circuit.circuit_size)
                binding_rates_dissociation = np.random.rand(
                    circuit.circuit_size, circuit.circuit_size)
                interactions = np.random.rand(
                    circuit.circuit_size, circuit.circuit_size)
                interaction_units = 'test'
                circuit.interactions = MolecularInteractions(
                    coupled_binding_rates=interactions, binding_rates_dissociation=binding_rates_dissociation,
                    eqconstants=eqconstants, units=interaction_units
                )

            filename_addons = ['eqconstants',
                               'binding_rates_dissociation', 'interactions']
            for interaction_matrix, filename_addon in zip(
                [circuit.interactions.eqconstants, circuit.interactions.binding_rates_dissociation,
                 circuit.interactions.coupled_binding_rates], filename_addons
            ):
                self.result_writer.output(
                    out_type='csv', out_name=circuit.name, data=interactions_to_df(
                        interaction_matrix, labels=[s.name for s in circuit.model.species]), overwrite=False,
                    new_file=True, filename_addon=filename_addon, subfolder=filename_addon)
        return circuit

    def run_interaction_simulator(self, species: List[Species]) -> InteractionData:
        data = {s.name: s.physical_data for s in species}
        simulator = InteractionSimulator(self.simulator_args)
        return simulator.run(data)

    def find_steady_states(self, circuit: Circuit):
        modeller_steady_state = Deterministic(
            max_time=50, time_interval=0.1)

        circuit.qreactions.quantities = self.compute_steady_states(modeller_steady_state,
                                                                   circuit=circuit,
                                                                   solver_type=self.steady_state_solver)

        circuit.result_collector.add_result(circuit.qreactions.quantities,
                                            name='steady_states',
                                            category='time_series',
                                            vis_func=VisODE().plot,
                                            vis_kwargs={'t': np.arange(0, circuit.circuit_size) *
                                                        modeller_steady_state.time_interval,
                                                        'legend': [s.name for s in circuit.model.species],
                                                        'out_type': 'svg'})
        return circuit

    def compute_steady_states(self, modeller: Modeller, circuit: Circuit,
                              solver_type: str = 'naive'):
        if solver_type == 'naive':
            copynumbers = self.model_circuit(
                modeller, y0=circuit.qreactions.quantities, circuit=circuit)
            # TODO: reshape to copynumbers[1] or copynumbers[0][1]
            copynumbers = copynumbers
        elif solver_type == 'ivp':
            steady_state_result = integrate.solve_ivp(
                partial(bioreaction_sim, args=None, reactions=circuit.qreactions.reactions, signal=vanilla_return,
                        signal_onehot=np.zeros_like(circuit.signal.onehot), dt=modeller.time_interval),
                (0, modeller.max_time),
                y0=circuit.qreactions.quantities)
            if not steady_state_result.success:
                raise ValueError(
                    'Steady state could not be found through solve_ivp - possibly because units '
                    f'are in {circuit.interactions.units}. {SIMULATOR_UNITS}')
            copynumbers = steady_state_result.y
        return copynumbers

    def model_circuit(self, modeller: Modeller, y0: np.ndarray, circuit: Circuit):
        assert np.shape(y0)[circuit.model.species.time_axis] == 1, 'Please only use 1-d ' \
            f'initial copynumbers instead of {np.shape(y0)}'

        modelling_func = partial(
            bioreaction_sim_full,
            qreactions=circuit.qreactions,
            t0=0, t1=modeller.max_time, dt0=modeller.time_interval,
            signal_onehot=circuit.signal.onehot,
            signal=circuit.signal)

        copynumbers = self.iterate_modelling_func(y0, modelling_func,
                                                  max_time=modeller.max_time,
                                                  signal=circuit.signal)
        return copynumbers

    def iterate_modelling_func(self, copynumbers, init_copynumbers,
                               modelling_func, max_time,
                               signal_f=None,
                               signal_onehot=0):
        """ Loop the modelling function.
        IMPORTANT! Modeller already includes the dt, or the length of the time step taken. """

        current_copynumbers = init_copynumbers.flatten()
        for t in range(0, max_time-1):
            dxdt = modelling_func(
                copynumbers=copynumbers[:, t])
            current_copynumbers = np.add(
                dxdt, current_copynumbers).flatten()
            if signal_f is not None:
                current_copynumbers = current_copynumbers * invert_onehot(signal_onehot) + \
                    signal_onehot * signal_f.func(t)
            copynumbers[:, t +
                        1] = zero_out_negs(current_copynumbers)
        return copynumbers

    def simulate_signal(self, circuit: Circuit, signal: Signal = None, save_numerical_vis_data: bool = False,
                        use_solver: str = 'naive', ref_circuit: Circuit = None,
                        time_interval=1):
        if signal is None:
            circuit.signal.update_time_interval(time_interval)
            signal = circuit.signal

        modeller_signal = Deterministic(
            max_time=signal.total_time, time_interval=time_interval
        )
        if circuit.result_collector.get_result('steady_states'):
            steady_states = deepcopy(
                circuit.result_collector.get_result('steady_states'))
        else:
            steady_states = self.compute_steady_states(Deterministic(
                max_time=50/time_interval, time_interval=time_interval),
                circuit=circuit,
                solver_type=self.steady_state_solver)[:, -1]

        steady_states = steady_states.reshape(make_dynamic_indexer({
            circuit.model.species.species_axis: np.shape(steady_states)[circuit.model.species.species_axis],
            circuit.model.species.time_axis: 1
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
                (np.shape(init_copynumbers)[circuit.model.species.species_axis], modeller_signal.max_time-1))
            ), axis=1)
            for signal_component, time_start, time_end in signal.summarized_signal:
                time_span = time_end - time_start
                y0 = steady_states.flatten()
                steady_state_result = integrate.solve_ivp(
                    self.make_modelling_func(
                        modelling_func=modeller_signal.dxdt_RNA,
                        circuit=circuit,
                        fixed_value=signal_component,
                        fixed_value_idx=circuit.model.species.identities.get('input')),
                    (0, modeller_signal.max_time),
                    y0=y0)
                if not steady_state_result.success:
                    raise ValueError(
                        'Steady state could not be found through solve_ivp - possibly because units '
                        f'are in {circuit.model.species.interaction_units}.')
                expanded_steady_states = np.concatenate(
                    [steady_state_result.y,
                     np.repeat(np.expand_dims(steady_state_result.y[:, -1], axis=circuit.model.species.time_axis),
                               time_span - np.shape(steady_state_result.y)[1])])
                new_copynumbers[:,
                                time_start:time_end] = expanded_steady_states
        elif use_solver == 'jax':
            # new_copynumbers = simulate_signal_scan(
            #     copynumbers=steady_states.flatten(), time=np.arange(modeller_signal.max_time),
            #     full_interactions=circuit.species.interactions,
            #     creation_rates=circuit.species.creation_rates.flatten(),
            #     degradation_rates=circuit.species.degradation_rates.flatten(),
            #     identity_matrix=np.identity(
            #         circuit.species.size),
            #     signal=signal.real_signal, signal_idx=circuit.species.identities.get(
            #         'input'),
            #     one_step_func=modeller_signal.dxdt_RNA_jnp)
            # new_copynumbers = np.array(new_copynumbers[1]).T

            new_copynumbers = bioreaction_sim_full(
                circuit.qreactions, t0=0, t1=modeller_signal.max_time, dt0=modeller_signal.time_interval,
                signal=signal, signal_onehot=signal.onehot)

        circuit.model.species.copynumbers = np.concatenate(
            (circuit.model.species.copynumbers, new_copynumbers[make_dynamic_indexer({
                circuit.model.species.species_axis: slice(0, np.shape(new_copynumbers)[circuit.model.species.species_axis]),
                circuit.model.species.time_axis: slice(1, np.shape(new_copynumbers)[
                    circuit.model.species.species_axis])
            })]), axis=circuit.model.species.time_axis)
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
                                                        'legend': list(circuit.model.species.data.sample_names),
                                                        'out_type': 'svg'},
                                            analytics_kwargs={'signal_idx': signal.identities_idx,
                                                              'ref_circuit_signal': ref_circuit_signal})
        return circuit

    def simulate_signal_batch(self, circuits: List[Circuit], max_time,
                              circuit_idx: int = 0,
                              save_numerical_vis_data: bool = False,
                              ref_circuit: Circuit = None,
                              dt=1, as_batch=True):
        names = list([n for n, c in circuits])
        circuits = list([c for n, c in circuits])
        if circuit_idx < len(circuits):
            signal = circuits[circuit_idx].signal
            signal.update_time_interval(dt)
        else:
            raise ValueError(
                f'The chosen circuit index {circuit_idx} exceeds the circuit list (len {len(circuits)}')

        modeller_signal = Deterministic(
            max_time=max_time, time_interval=dt
        )

        # Batch
        t = np.arange(modeller_signal.max_time)
        b_starting_copynumbers = np.array(
            [c.model.species.steady_state_copynums.flatten() for c in circuits])
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
                                                                'legend': [s.name for s in circuit.model.species],
                                                                'out_type': 'svg'},
                                                    analytics_kwargs={'signal_idx': signal.identities_idx,
                                                                      'ref_circuit_signal': ref_circuit_signal})
        return list(zip(names, circuits))

    # @time_it
    def wrap_mutations(self, circuit: Circuit, methods: dict, include_normal_run=True,
                       write_to_subsystem=False):
        if write_to_subsystem:
            self.result_writer.subdivide_writing(circuit.name)
        mutation_dict = flatten_nested_dict(
            circuit.model.species.mutations.items())
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

    def batch_mutations(self, circuit: Circuit, methods: dict, include_normal_run=True,
                        write_to_subsystem=False):
        if write_to_subsystem:
            self.result_writer.subdivide_writing(circuit.name)

        mutation_dict = flatten_nested_dict(
            circuit.mutations.items())
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

    def apply_to_circuit(self, circuit: Circuit, _methods: dict, ref_circuit: Circuit):
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

    def visualise_graph(self, circuit: Circuit, mode="pyvis", new_vis=False):
        self.result_writer.visualise_graph(circuit, mode, new_vis)

    def write_results(self, circuit, new_report: bool = False, no_visualisations: bool = False,
                      only_numerical: bool = False, no_numerical: bool = False):
        self.result_writer.write_all(
            circuit, new_report, no_visualisations=no_visualisations, only_numerical=only_numerical,
            no_numerical=no_numerical)
        return circuit
