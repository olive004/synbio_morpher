from copy import deepcopy
from typing import List, Dict
from functools import partial
import os
import logging
import numpy as np
import jax
from scipy import integrate
from bioreaction.model.data_containers import Species
from bioreaction.simulation.simfuncs.basic_de import bioreaction_sim

from src.utils.results.result_writer import ResultWriter
from src.utils.circuit.agnostic_circuits.circuit_new import interactions_to_df
from src.srv.parameter_prediction.simulator import SIMULATOR_UNITS
from src.srv.parameter_prediction.interactions import MolecularInteractions, InteractionData, InteractionSimulator
from src.utils.misc.numerical import invert_onehot, zero_out_negs
from src.utils.misc.type_handling import flatten_nested_dict, flatten_listlike, get_unique
from src.srv.io.loaders.experiment_loading import INTERACTION_FILE_ADDONS
from src.utils.misc.helper import vanilla_return
from src.utils.results.visualisation import VisODE
from src.utils.signal.signals_new import Signal
from src.utils.circuit.agnostic_circuits.circuit_new import Circuit, update_species_simulated_rates
from src.utils.results.analytics.timeseries import generate_analytics
from src.utils.modelling.deterministic import Deterministic, bioreaction_sim_full, bioreactions_simulate_signal_scan
from src.utils.evolution.mutation import implement_mutation
from src.utils.modelling.base import Modeller


TEST_MODE = True


class CircuitModeller():

    def __init__(self, result_writer=None, config: dict = {}) -> None:
        self.result_writer = ResultWriter() if result_writer is None else result_writer
        self.steady_state_solver = config.get("steady_state_solver", 'ivp')
        self.simulator_args = config['interaction_simulator']
        self.dt = config.get('simulation', {}).get('dt', 1)
        self.t0 = config.get('simulation', {}).get('t0', 0)
        self.t1 = config.get('simulation', {}).get('t1', 10)

    def init_circuit(self, circuit: Circuit):
        circuit = self.compute_interaction_strengths(circuit)
        circuit = self.find_steady_states(circuit)
        return circuit

    def make_modelling_func(self, modelling_func, circuit: Circuit,
                            fixed_value=None,
                            fixed_value_idx: int = None):

        return partial(modelling_func, full_interactions=circuit.interactions.coupled_binding_rates,
                       creation_rates=circuit.qreactions.reactions.forward_rates.flatten(),
                       degradation_rates=circuit.qreactions.reactions.reverse_rates.flatten(),
                       signal=fixed_value,
                       signal_idx=fixed_value_idx,
                       identity_matrix=np.identity(circuit.circuit_size)
                       )

    # @time_it
    def compute_interaction_strengths(self, circuit: Circuit):
        if circuit.interactions_state == 'uninitialised':
            if not TEST_MODE:
                interactions = self.run_interaction_simulator(
                    get_unique(flatten_listlike([r.input for r in circuit.model.reactions])))
                circuit = update_species_simulated_rates(
                    circuit, interactions.interactions)
                circuit.interactions = interactions.interactions
            else:
                logging.warning(
                    'RUNNING IN TEST MODE - interaction rates are fake.')
                random_matrices = np.random.rand(
                    circuit.circuit_size, circuit.circuit_size, 4) * 0.000001
                circuit.interactions = MolecularInteractions(
                    coupled_binding_rates=random_matrices[:, :, 0],
                    binding_rates_association=random_matrices[:, :, 1],
                    binding_rates_dissociation=random_matrices[:, :, 2],
                    eqconstants=random_matrices[:, :, 3], units='test'
                )
                circuit = update_species_simulated_rates(
                    circuit, circuit.interactions)

        filename_addons = INTERACTION_FILE_ADDONS.keys()
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
            max_time=500, time_interval=0.1)

        steady_states = self.compute_steady_states(modeller_steady_state,
                                                   circuit=circuit,
                                                   solver_type=self.steady_state_solver)

        circuit.result_collector.add_result(
            data=steady_states,
            name='steady_states',
            category='time_series',
            vis_func=VisODE().plot,
            vis_kwargs={'t': np.arange(0, np.shape(steady_states)[circuit.time_axis]) *
                        modeller_steady_state.time_interval,
                        'legend': [s.name for s in circuit.model.species],
                        'out_type': 'svg'},
            analytics_kwargs={'labels': [s.name for s in circuit.model.species]})
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
                        signal_onehot=np.zeros_like(circuit.signal.onehot),
                        inverse_onehot=np.ones_like(circuit.signal.onehot)),
                (0, modeller.max_time),
                y0=circuit.qreactions.quantities)
            if not steady_state_result.success:
                raise ValueError(
                    'Steady state could not be found through solve_ivp - possibly because units '
                    f'are in {circuit.interactions.units}. {SIMULATOR_UNITS}')
            copynumbers = steady_state_result.y
        return copynumbers

    def model_circuit(self, y0: np.ndarray, circuit: Circuit):
        assert np.shape(y0)[circuit.time_axis] == 1, 'Please only use 1-d ' \
            f'initial copynumbers instead of {np.shape(y0)}'

        modelling_func = partial(bioreaction_sim, args=None,
                                 reactions=circuit.qreactions.reactions,
                                 signal=circuit.signal.func,
                                 signal_onehot=circuit.signal.onehot,
                                 inverse_onehot=invert_onehot(circuit.signal.onehot))

        copynumbers = self.iterate_modelling_func(y0, modelling_func,
                                                  max_time=self.t1,
                                                  time_interval=self.dt,
                                                  signal_f=circuit.signal.func,
                                                  signal_onehot=circuit.signal.onehot)
        return copynumbers

    def iterate_modelling_func(self, init_copynumbers,
                               modelling_func, max_time,
                               time_interval,
                               signal_f=None,
                               signal_onehot=0):
        """ Loop the modelling function.
        IMPORTANT! Modeller already includes the dt, or the length of the time step taken. """

        time = np.arange(0, max_time-1, time_interval)
        copynumbers = np.concatenate((init_copynumbers, np.zeros(
            (np.shape(init_copynumbers)[0], len(time)))), axis=1)
        current_copynumbers = init_copynumbers.flatten()
        for i, t in enumerate(time):
            dxdt = modelling_func(
                t, copynumbers[:, i])
            current_copynumbers = np.add(
                dxdt, current_copynumbers)
            if signal_f is not None:
                current_copynumbers = current_copynumbers * invert_onehot(signal_onehot) + \
                    signal_onehot * signal_f(t)
            copynumbers[:, i +
                        1] = zero_out_negs(current_copynumbers)
        return copynumbers

    def simulate_signal(self, circuit: Circuit, signal: Signal = None, save_numerical_vis_data: bool = False,
                        solver: str = 'naive', ref_circuit: Circuit = None):
        signal = signal if signal is not None else circuit.signal

        if circuit.result_collector.get_result('steady_states'):
            steady_states = deepcopy(
                circuit.result_collector.get_result('steady_states').analytics['steady_states'])
        else:
            steady_states = self.compute_steady_states(Deterministic(
                max_time=self.t1, time_interval=self.dt),
                circuit=circuit,
                solver_type=self.steady_state_solver)[:, -1]

        if solver == 'naive':
            new_copynumbers = self.model_circuit(steady_states,
                                                 circuit=circuit)
            t = np.arange(0, np.shape(new_copynumbers)[
                1]) * self.t1 / np.shape(new_copynumbers)[1]

        elif solver == 'diffrax':
            solution = bioreaction_sim_full(
                y0=steady_states.flatten() * invert_onehot(signal.onehot),
                qreactions=circuit.qreactions, t0=0, t1=self.t1, dt0=self.dt,
                signal=signal.func, signal_onehot=signal.onehot)
            new_copynumbers = solution.ys[solution.ts < np.inf]
            t = solution.ts[solution.ts < np.inf]
            if np.shape(new_copynumbers)[0] != circuit.circuit_size:
                new_copynumbers = np.rollaxis(new_copynumbers, axis=1)

        if ref_circuit is None or ref_circuit == circuit:
            ref_circuit_signal = None
        else:
            ref_circuit_result = ref_circuit.result_collector.get_result(
                'signal')
            ref_circuit_signal = None if ref_circuit_result is None else ref_circuit_result.data

        circuit.result_collector.add_result(
            data=new_copynumbers,
            name='signal',
            category='time_series',
            time=t,
            vis_func=VisODE().plot,
            save_numerical_vis_data=save_numerical_vis_data,
            vis_kwargs={'t': t,
                        'legend': [s.name for s in circuit.model.species],
                        'out_type': 'svg'},
            analytics_kwargs={'labels': [s.name for s in circuit.model.species],
                              'signal_onehot': signal.onehot,
                              'ref_circuit_signal': ref_circuit_signal})
        return circuit

    def simulate_signal_batch(self, all_circuits: Dict[str, Dict[str, Circuit]],
                              circuit_idx: int = 0,
                              save_numerical_vis_data: bool = False,
                              ref_circuit: Circuit = None,
                              batch=True):
        circuits = [c for s in all_circuits.values() for c in s.values()]
        if circuit_idx < len(circuits):
            signal = circuits[circuit_idx].signal
            signal.update_time_interval(self.dt)
        else:
            raise ValueError(
                f'The chosen circuit index {circuit_idx} exceeds the circuit list (len {len(circuits)}')

        # Batch
        t = np.arange(self.t0, self.t1, self.dt)
        b_steady_states = np.array(
            [c.result_collector.get_result('steady_states').analytics['steady_states'].flatten(
            ) * invert_onehot(signal.onehot) for c in circuits]
        )
        b_forward_rates = np.array(
            [c.qreactions.reactions.forward_rates for c in circuits])
        b_reverse_rates = np.array(
            [c.qreactions.reactions.reverse_rates for c in circuits])

        b_new_copynumbers = jax.vmap(
            partial(bioreactions_simulate_signal_scan,
                    time=t, signal=signal.func, signal_onehot=signal.onehot,
                    inputs=circuits[circuit_idx].qreactions.reactions.inputs,
                    outputs=circuits[circuit_idx].qreactions.reactions.outputs))(
                        copynumbers=b_steady_states, forward_rates=b_forward_rates,
                        reverse_rates=b_reverse_rates)
        # b_new_copynumbers = jax.vmap(partial(simulate_signal_scan,
        #                                      # b_new_copynumbers = partial(simulate_signal_scan,
        #                                      time=t,
        #                                      creation_rates=circuits[circuit_idx].species.creation_rates.flatten(
        #                                      ),
        #                                      degradation_rates=circuits[circuit_idx].species.degradation_rates.flatten(
        #                                      ),
        #                                      identity_matrix=np.identity(
        #                                          circuits[circuit_idx].species.size),
        #                                      signal=signal.real_signal, signal_idx=circuits[circuit_idx].species.identities.get(
        #                                          'input'),
        #                                      one_step_func=modeller_signal.dxdt_RNA_jnp))(b_starting_copynumbers, full_interactions=b_interactions)
        b_new_copynumbers = np.array(b_new_copynumbers[1])
        if np.shape(b_new_copynumbers)[1] != circuits[circuit_idx].circuit_size and np.shape(b_new_copynumbers)[-1] == circuits[circuit_idx].circuit_size:
            b_new_copynumbers = np.swapaxes(b_new_copynumbers, 1, 2)

        # Get analytics batched too
        if ref_circuit is None or ref_circuit == circuit:
            ref_circuit_signal = None
        else:
            ref_circuit_result = ref_circuit.result_collector.get_result(
                'signal')
            ref_circuit_signal = None if ref_circuit_result is None else ref_circuit_result.data.flatten()
        b_analytics = jax.vmap(partial(generate_analytics, time=t, labels=[s.name for s in circuits[circuit_idx].model.species],
                                       signal_onehot=signal.onehot, ref_circuit_data=ref_circuit_signal))(data=b_new_copynumbers)
        b_analytics = [{k: v[i] for k, v in b_analytics.items()}
                       for i in range(len(circuits))]

        # Apply to all circuits
        for i, (circuit, analytics) in enumerate(zip(circuits, b_analytics)):
            circuits[i].result_collector.add_result(
                data=b_new_copynumbers[i],
                name='signal',
                category='time_series',
                vis_func=VisODE().plot,
                save_numerical_vis_data=save_numerical_vis_data,
                analytics=analytics,
                vis_kwargs={'t': t,
                            'legend': [s.name for s in circuit.model.species],
                            'out_type': 'svg'})
        return {top_name: {subname: circuits[len(v)*i + j] for j, subname in enumerate(v.keys())}
                for i, (top_name, v) in enumerate(all_circuits.items())}

    def make_subcircuit(self, circuit: Circuit, mutation_name: str, mutation=None):

        subcircuit = deepcopy(circuit)
        subcircuit.reset_to_initial_state()
        subcircuit.interactions_state = 'uninitialised'
        if mutation is None:
            mutation = circuit.mutations.get(mutation_name)

        subcircuit = implement_mutation(circuit=subcircuit, mutation=mutation)
        return subcircuit

    # @time_it
    def wrap_mutations(self, circuit: Circuit, methods: dict, include_normal_run=True,
                       write_to_subsystem=False):
        if write_to_subsystem:
            self.result_writer.subdivide_writing(circuit.name)
        mutation_dict = flatten_nested_dict(
            circuit.mutations)
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
            subcircuit = self.make_subcircuit(circuit, name, mutation)
            self.result_writer.subdivide_writing(name, safe_dir_change=False)
            self.apply_to_circuit(subcircuit, methods, ref_circuit=circuit)
            self.result_writer.unsubdivide_last_dir()
        self.result_writer.unsubdivide()

    def batch_mutations(self, circuit: Circuit, methods: dict, include_normal_run=True,
                        write_to_subsystem=False):
        subcircuits = {
            circuit.name: {subname: self.make_subcircuit(circuit, subname, mutation)
                           for subname, mutation in flatten_nested_dict(
                circuit.mutations).items()}
        }
        subcircuits[circuit.name]['ref_circuit'] = circuit

        subcircuits = self.run_batch(subcircuits, methods,
                                     include_normal_run, write_to_subsystem)
        return subcircuits

    def batch_circuits(self, circuits: List[Circuit], methods: dict, include_normal_run=True,
                       write_to_subsystem=False) -> List[Circuit]:
        subcircuits = {
            circuit.name: {subname: self.make_subcircuit(circuit, subname, mutation)
                           for subname, mutation in flatten_nested_dict(
                circuit.mutations).items()}
            for circuit in circuits
        }
        for circuit in circuits:
            subcircuits[circuit.name]['ref_circuit'] = circuit

        subcircuits = self.run_batch(
            subcircuits, methods, include_normal_run, write_to_subsystem)
        return subcircuits

    def run_batch(self, subcircuits: Dict[str, Dict[str, Circuit]], methods: dict,
                  include_normal_run: bool = True, write_to_subsystem: bool = False) -> Dict[str, Dict[str, Circuit]]:
        # ref_circuit = None
        for method, kwargs in methods.items():
            if kwargs.get('batch') == True:  # method is batchable
                if hasattr(self, method):
                    subcircuits = getattr(self, method)(subcircuits, **kwargs)
                else:
                    logging.warning(
                        f'Could not find method @{method} in class {self}')
            else:
                for top_name, v in subcircuits.items():
                    for sub_name, subcircuit in v.items():
                        dir_name = top_name if sub_name == 'ref_circuit' or not write_to_subsystem else os.path.join(
                            top_name, 'mutations', sub_name)
                        self.result_writer.subdivide_writing(
                            dir_name, safe_dir_change=True)
                        if not include_normal_run and sub_name == 'ref_circuit':
                            continue
                        subcircuit = self.apply_to_circuit(
                            subcircuit, {method: kwargs}, ref_circuit=subcircuits[top_name]['ref_circuit'])
                        subcircuits[top_name][sub_name] = subcircuit
                self.result_writer.unsubdivide()
        return subcircuits

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
