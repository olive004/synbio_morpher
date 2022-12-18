from copy import deepcopy
from typing import List
from functools import partial
from datetime import datetime
import inspect
import os
# import sys
import logging
import numpy as np

import jax
from scipy import integrate
from bioreaction.model.data_containers import Species
from bioreaction.simulation.simfuncs.basic_de import bioreaction_sim

from src.utils.results.result_writer import ResultWriter
from src.utils.circuit.agnostic_circuits.circuit_new import interactions_to_df
from src.srv.parameter_prediction.simulator import SIMULATOR_UNITS
from src.srv.parameter_prediction.interactions import InteractionData, InteractionSimulator
from src.utils.misc.numerical import invert_onehot, zero_out_negs
from src.utils.misc.type_handling import flatten_nested_dict, flatten_listlike, get_unique
from src.utils.misc.runtime import clear_caches
from src.srv.io.loaders.experiment_loading import INTERACTION_FILE_ADDONS
from src.utils.misc.helper import vanilla_return
from src.utils.results.visualisation import VisODE
from src.utils.signal.signals_new import Signal
from src.utils.circuit.agnostic_circuits.circuit_new import Circuit, update_species_simulated_rates
from src.utils.results.analytics.timeseries import generate_analytics
from src.utils.modelling.deterministic import Deterministic, bioreaction_sim_wrapper, bioreaction_sim_dfx_expanded
from src.utils.evolution.mutation import implement_mutation
from src.utils.modelling.base import Modeller


class CircuitModeller():

    def __init__(self, result_writer=None, config: dict = {}) -> None:
        self.result_writer = ResultWriter() if result_writer is None else result_writer
        self.steady_state_solver = config.get("steady_state_solver", 'ivp')
        self.simulator_args = config['interaction_simulator']
        self.discard_numerical_mutations = config['experiment'].get(
            'no_numerical', False)
        self.dt = config.get('simulation', {}).get('dt', 1)
        self.t0 = config.get('simulation', {}).get('t0', 0)
        self.t1 = config.get('simulation', {}).get('t1', 10)
        
        self.max_circuits = config.get('simulation', {}).get('max_circuits', 10000)  # Maximum number of circuits to hold in memory
        self.test_mode = config.get('experiment', {}).get('test_mode', False)

        # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

        # jax.config.update('jax_platform_name', config.get('simulation', {}).get('device', 'cpu'))
        # logging.warning(f'Using device {config.get("simulation", {}).get("device", "cpu")}')

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
        if circuit.interactions_state == 'uninitialised' and not self.test_mode:
            interactions = self.run_interaction_simulator(
                sorted(get_unique(flatten_listlike([r.input for r in circuit.model.reactions]))))
            circuit = update_species_simulated_rates(
                circuit, interactions.interactions)
            circuit.interactions = interactions.interactions

        filename_addons = sorted(INTERACTION_FILE_ADDONS.keys())
        for interaction_matrix, filename_addon in zip(
            [circuit.interactions.binding_rates_dissociation,
             circuit.interactions.coupled_binding_rates,
             circuit.interactions.eqconstants], filename_addons
        ):
            self.result_writer.output(
                data=interactions_to_df(
                    interaction_matrix,
                    labels=sorted([s.name for s in get_unique(flatten_listlike([r.input for r in circuit.model.reactions]))])),
                out_type='csv', out_name=circuit.name,
                overwrite=False, new_file=True,
                filename_addon=filename_addon, subfolder=filename_addon)
        return circuit

    def run_interaction_simulator(self, species: List[Species]) -> InteractionData:
        data = {s: s.physical_data for s in species}
        simulator = InteractionSimulator(self.simulator_args)
        return simulator.run(data)

    def find_steady_states(self, circuit: Circuit):
        modeller_steady_state = Deterministic(
            max_time=10, time_interval=0.1)

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
                        signal_onehot=np.zeros_like(circuit.signal.onehot)),
                (0, modeller.max_time),
                y0=circuit.qreactions.quantities)
            if not steady_state_result.success:
                raise ValueError(
                    'Steady state could not be found through solve_ivp - possibly because units '
                    f'are in {circuit.interactions.units}. {SIMULATOR_UNITS}')
            copynumbers = steady_state_result.y
        # elif solver_type == 'jax':
        #     partial(bioreaction_sim, args=None, reactions=circuit.qreactions.reactions, signal=vanilla_return,
        #                 signal_onehot=np.zeros_like(circuit.signal.onehot)),
        #         (0, modeller.max_time)
        return copynumbers

    def model_circuit(self, y0: np.ndarray, circuit: Circuit):
        assert np.shape(y0)[circuit.time_axis] == 1, 'Please only use 1-d ' \
            f'initial copynumbers instead of {np.shape(y0)}'

        modelling_func = partial(bioreaction_sim, args=None,
                                 reactions=circuit.qreactions.reactions,
                                 signal=circuit.signal.func,
                                 signal_onehot=circuit.signal.onehot)

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
            solution = bioreaction_sim_wrapper(
                y0=steady_states.flatten() * invert_onehot(signal.onehot),
                qreactions=circuit.qreactions, t0=0, t1=self.t1, dt0=self.dt,
                signal=signal.func, signal_onehot=signal.onehot)
            new_copynumbers = solution.ys[solution.ts < np.inf]
            t = solution.ts[solution.ts < np.inf]
            if np.shape(new_copynumbers)[0] != circuit.circuit_size:
                new_copynumbers = np.rollaxis(new_copynumbers, axis=1)

        if ref_circuit is None or ref_circuit == circuit:
            ref_circuit_data = None
        else:
            ref_circuit_result = ref_circuit.result_collector.get_result(
                'signal')
            ref_circuit_data = None if ref_circuit_result is None else ref_circuit_result.data

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
                              'ref_circuit_data': ref_circuit_data})
        return circuit

    def simulate_signal_batch(self, circuits: List[Circuit],
                              ref_circuit: Circuit,
                              save_numerical_vis_data: bool = False,
                              batch=True):
        signal = ref_circuit.signal
        signal.update_time_interval(self.dt)

        # Batch
        b_steady_states = [None] * len(circuits)
        b_forward_rates = [None] * len(circuits)
        b_reverse_rates = [None] * len(circuits)
        for i, c in enumerate(circuits):
            b_steady_states[i] = c.result_collector.get_result(
                'steady_states').analytics['steady_states'].flatten()
            b_forward_rates[i] = c.qreactions.reactions.forward_rates
            b_reverse_rates[i] = c.qreactions.reactions.reverse_rates
        b_steady_states = np.asarray(b_steady_states)
        b_forward_rates = np.asarray(b_forward_rates)
        b_reverse_rates = np.asarray(b_reverse_rates)

        sim_func = jax.vmap(
            partial(bioreaction_sim_dfx_expanded,
                    t0=self.t0, t1=self.t1, dt0=self.dt,
                    signal=signal.func, signal_onehot=signal.onehot,
                    inputs=ref_circuit.qreactions.reactions.inputs,
                    outputs=ref_circuit.qreactions.reactions.outputs))
        solution = sim_func(
            y0=b_steady_states, forward_rates=b_forward_rates, reverse_rates=b_reverse_rates)

        tf = np.argmax(solution.ts == np.inf)
        b_new_copynumbers = solution.ys[:, :tf, :]
        t = solution.ts[0, :tf]

        if np.shape(b_new_copynumbers)[1] != ref_circuit.circuit_size and np.shape(b_new_copynumbers)[-1] == ref_circuit.circuit_size:
            b_new_copynumbers = np.swapaxes(b_new_copynumbers, 1, 2)

        # Get analytics batched too
        def append_nest_dicts(l: list, i0: int, i1: int, d: dict) -> list:
            for i in range(i0, i1):
                b_analytics_k = {}
                for k, v in d.items():
                    b_analytics_k[k] = v[i]
                l.append(b_analytics_k)
            return l

        ref_circuits = [s for s in circuits if s.subname == 'ref_circuit']
        ref_idxs = [circuits.index(s) for s in ref_circuits]
        b_analytics_l = []

        # First check if the ref_circuit is leading
        if ref_circuit not in circuits:
            ref_idxs.insert(0, None)
        else:
            assert circuits.index(ref_circuit) == 0, f'The reference circuit should be leading or at idx 0, but is at idx {circuits.index(ref_circuit)}'
        ref_idxs2 = [len(circuits)] if len(ref_idxs) < 2 else ref_idxs[1:] + [len(circuits)]
        for ref_idx, ref_idx2 in zip(ref_idxs, ref_idxs2):
            
            if ref_idx is None:
                ref_circuit_result = ref_circuit.result_collector.get_result('signal')
                if ref_circuit_result is None:
                    raise ValueError('Reference circuit was not simulated and was not in this batch.')
                else:
                    ref_idx = 0
                    ref_circuit_data = ref_circuit_result.data
            else:
                ref_circuit_data = b_new_copynumbers[ref_idx]

            analytics_func = jax.vmap(partial(generate_analytics, time=t, labels=[s.name for s in ref_circuit.model.species],
                                        signal_onehot=signal.onehot, ref_circuit_data=ref_circuit_data))
            b_analytics = analytics_func(data=b_new_copynumbers[ref_idx:ref_idx2])
            b_analytics_l = append_nest_dicts(b_analytics_l, ref_idx, ref_idx2, b_analytics)
        assert len(b_analytics_l) == len(circuits), f'There was a mismatch in length of analytics ({len(b_analytics_l)}) and circuits ({len(circuits)})'

        # Save for all circuits
        for i, (circuit, analytics) in enumerate(zip(circuits, b_analytics_l)):
            if self.discard_numerical_mutations and circuit.subname != 'ref_circuit':
                sig_data = None
            else:
                sig_data = b_new_copynumbers[i]
            circuits[i].result_collector.add_result(
                data=sig_data,
                name='signal',
                category='time_series',
                vis_func=VisODE().plot,
                save_numerical_vis_data=save_numerical_vis_data,
                analytics=analytics,
                vis_kwargs={'t': t,
                            'legend': [s.name for s in circuit.model.species],
                            'out_type': 'svg'})
        # return {top_name: {subname: circuits[len(v)*i + j] for j, subname in enumerate(v.keys())}
        #         for i, (top_name, v) in enumerate(.items())}
        clear_caches()
        return circuits

    def make_subcircuit(self, circuit: Circuit, mutation_name: str, mutation=None):

        subcircuit = deepcopy(circuit)
        subcircuit.reset_to_initial_state()
        subcircuit.interactions_state = 'uninitialised'
        if mutation is None:
            mutation = circuit.mutations.get(mutation_name)
        subcircuit.subname = mutation_name

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
        return circuit

    def batch_circuits(self,
                       circuits: List[Circuit],
                       methods: dict,
                       batch_size: int = None,
                       include_normal_run: bool = True,
                       write_to_subsystem=True):
        batch_size = len(circuits) if batch_size is None else batch_size

        num_subcircuits = len(flatten_nested_dict(circuits[0].mutations))
        expected_tot_subcircuits = len(circuits) * (1+num_subcircuits)
        if expected_tot_subcircuits > self.max_circuits:
            viable_circuit_num = int(self.max_circuits / (num_subcircuits+1))
        else:
            viable_circuit_num = len(circuits)

        logging.warning(
            f'\tFrom {len(circuits)} circuits, a total of {expected_tot_subcircuits} mutated circuits will be simulated.')

        start_time = datetime.now()
        for vi in range(0, len(circuits), viable_circuit_num):
            single_batch_time = datetime.now()
            vf = min(vi+viable_circuit_num, len(circuits))
            logging.warning(
                f'\t\tStarting new round of viable circuits ({vi} - {vf} / {len(circuits)})')

            # Preallocate then create subcircuits - otherwise memory leak
            subcircuits = [None] * (viable_circuit_num * (1+num_subcircuits))
            c_idx = 0
            for circuit in circuits[vi: vf]:
                subcircuits[c_idx] = deepcopy(circuit)
                c_idx += 1
                for subname, mutation in flatten_nested_dict(circuit.mutations).items():
                    subcircuits[c_idx] = self.make_subcircuit(
                        circuit, subname, mutation)
                    c_idx += 1
            if c_idx != len(subcircuits):
                subcircuits = list(filter(lambda item: item is not None, subcircuits))

            # Batch
            ref_circuit = subcircuits[0]
            for b in range(0, len(subcircuits), batch_size):
                logging.warning(
                    f'\tBatching {b} - {b+batch_size} circuits (out of {vi/viable_circuit_num*len(subcircuits)} - {vf/viable_circuit_num*len(subcircuits)} (total: {expected_tot_subcircuits})) (Circuits: {vi} - {vf} of {len(circuits)})')
                bf = b+batch_size if b + \
                    batch_size < len(subcircuits) else len(subcircuits)

                b_circuits = subcircuits[b:bf]
                if not b_circuits:
                    continue
                ref_circuit = self.run_batch(
                    b_circuits, methods, leading_ref_circuit=ref_circuit,
                    include_normal_run=include_normal_run,
                    write_to_subsystem=write_to_subsystem)

            single_batch_time = datetime.now() - single_batch_time
            logging.warning(
                f'Single batch: {single_batch_time} \nProjected time: {single_batch_time.total_seconds() * len(circuits)/viable_circuit_num} \nTotal time: {str(datetime.now() - start_time)}')
            del subcircuits
        return circuits

    def run_batch(self,
                  subcircuits: List[Circuit],
                  methods: dict,
                  leading_ref_circuit: Circuit = None,
                  include_normal_run: bool = True,
                  write_to_subsystem: bool = True) -> List[Circuit]:

        for method, kwargs in methods.items():
            ref_circuit = leading_ref_circuit
            logging.warning(
                f'\t\tRunning {len(subcircuits)} Subcircuits - {subcircuits[0].name}: {method}')

            if kwargs.get('batch'):  # method is batchable
                if hasattr(self, method):
                    if 'ref_circuit' in inspect.getfullargspec(getattr(self, method)).args:
                        kwargs.update({'ref_circuit': ref_circuit})
                    subcircuits = getattr(self, method)(subcircuits, **kwargs)
                else:
                    logging.warning(
                        f'Could not find method @{method} in class {self}')
            else:
                for i, subcircuit in enumerate(subcircuits):

                    dir_name = subcircuit.name if subcircuit.subname == 'ref_circuit' or not write_to_subsystem else os.path.join(
                        subcircuit.name, 'mutations', subcircuit.subname)

                    self.result_writer.subdivide_writing(
                        dir_name, safe_dir_change=True)

                    if subcircuit.subname == 'ref_circuit' and subcircuit != ref_circuit:
                        ref_circuit.result_collector.delete_result('signal')
                        ref_circuit = subcircuit
                        if not include_normal_run:
                            continue

                    subcircuit = self.apply_to_circuit(
                        subcircuit, {method: kwargs}, ref_circuit=ref_circuit)
                    subcircuits[i] = subcircuit
                self.result_writer.unsubdivide()
        # Update the leading reference circuit to be the last ref circuti from this batch
        ref_circuits = [c for c in subcircuits if c.subname == 'ref_circuit']
        if ref_circuits:
            leading_ref_circuit = ref_circuits[-1]
        del subcircuits
        return leading_ref_circuit

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

    def write_results(self, circuit: Circuit, new_report: bool = False, no_visualisations: bool = False,
                      only_numerical: bool = False, no_numerical: bool = False):
        self.result_writer.write_all(
            circuit, new_report, no_visualisations=no_visualisations, only_numerical=only_numerical,
            no_numerical=no_numerical)
        return circuit
