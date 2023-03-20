from bioreaction.model.data_tools import construct_model_fromnames
from bioreaction.simulation.simfuncs.basic_de import bioreaction_sim_expanded
from bioreaction.simulation.basic_sim import basic_de_sim, convert_model, BasicSimParams, BasicSimState
from bioreaction.simulation.med_sim import MedSimParams

from datetime import datetime
import diffrax as dfx
from tqdm import tqdm
import jax
import pandas as pd
import scipy
import jax.numpy as jnp
import numpy as np
from copy import deepcopy
from functools import partial
import os
import sys

import matplotlib.pyplot as plt
plt.style.use('bmh')
plt.style.use('seaborn-v0_8')

os.environ["TF_CPP_MIN_LOG_LOVEL"] = "0"
jax.config.update('jax_platform_name', 'cpu')
jax.devices()

if __package__ is None or (__package__ == ''):

    module_path = os.path.abspath(os.path.join('..'))
    sys.path.append(module_path)
    sys.path.append(os.path.abspath(os.path.join('.')))

    __package__ = os.path.basename(module_path)


from src.utils.results.analytics.timeseries import get_precision, get_sensitivity, get_step_response_times
from src.utils.misc.units import per_mol_to_per_molecule
from src.utils.misc.type_handling import flatten_listlike
from src.utils.misc.numerical import make_symmetrical_matrix_from_sequence
from src.utils.modelling.deterministic import bioreaction_sim_dfx_expanded
from tests.shared import CONFIG


config = deepcopy(CONFIG)


def update_model_rates(model, a=None, d=None, ka=None, kd=None):
    for i, r in enumerate(model.reactions):
        if not r.input:  # 0 -> RNA
            if a is not None:
                model.reactions[i].forward_rate = a[model.species.index(
                    r.output[0])]
                model.reactions[i].reverse_rate = 0
        elif not r.output:  # RNA -> 0
            if d is not None:
                model.reactions[i].forward_rate = d[model.species.index(
                    r.input[0])]
                model.reactions[i].reverse_rate = 0
        else:
            if ka is not None:
                model.reactions[i].forward_rate = ka[model.species.index(r.input[0]),
                                                     model.species.index(r.input[1])]
            if kd is not None:
                model.reactions[i].reverse_rate = kd[model.species.index(r.input[0]),
                                                     model.species.index(r.input[1])]
    return model


def define_matrices(num_species, size_interaction_array, num_unique_interactions, num_analytic_types):
    matrix_dimensions = tuple(
        [num_species] + [size_interaction_array]*num_unique_interactions)
    matrix_size = num_species * \
        np.power(size_interaction_array, num_unique_interactions)
    assert matrix_size == np.prod(list(
        matrix_dimensions)), 'Something is off about the intended size of the matrix'

    all_analytic_matrices = []
    for _analytic in range(num_analytic_types):
        all_analytic_matrices.append(np.zeros(
            matrix_dimensions, dtype=np.float32))
    return all_analytic_matrices


def make_reverse_rates(med_model, reverse_rates, interaction_matrices):
    index_translation = []
    for i, r in enumerate(med_model.reactions):
        if r.output and r.input:
            index_translation.append([i, med_model.species.index(r.input[0]),
                                      med_model.species.index(r.input[1])])
    reverse_rates = np.repeat(
        np.expand_dims(reverse_rates, axis=0),
        interaction_matrices.shape[0], axis=0
    )
    for i, k, j in index_translation:
        reverse_rates[:, i] = interaction_matrices[:, k, j]
    return reverse_rates


def clear_gpu():
    backend = jax.lib.xla_bridge.get_backend()
    for buf in backend.live_buffers():
        buf.delete()


# def get_steady_states(starting_state, reverse_rates, sim_model, params):

#     def to_vmap_basic(reverse_rates):
#         sim_model.reverse_rates = reverse_rates
#         return basic_de_sim(starting_state=starting_state, model=sim_model, params=params)

#     steady_state_results = jax.jit(jax.vmap(to_vmap_basic))(reverse_rates)
#     return steady_state_results


def get_loop_simfunc(params, sim_model, saveat):
    def to_vmap_basic(starting_state, reverse_rates):
        return partial(bioreaction_sim_dfx_expanded,
                       t0=params.t_start, t1=params.t_end, dt0=params.delta_t,
                       signal=None, signal_onehot=None,  # signal.reactions_onehot,
                       forward_rates=sim_model.forward_rates,
                       inputs=sim_model.inputs,
                       outputs=sim_model.outputs,
                       solver=dfx.Heun(),
                       saveat=saveat,
                       max_steps=int(np.min(
                           [np.max([16**5, ((params.t_end - params.t_start)/params.delta_t) * 5]), 16**6]))
                       )(y0=starting_state, reverse_rates=reverse_rates)

    return jax.jit(jax.vmap(to_vmap_basic))
    # return partial(scipy.integrate.solve_ivp, fun=to_vmap_basic, t_span=(params.t_start, params.t_end))


def loop_sim(steady_state_results, reverse_rates, sim_func):
    steady_state_results = sim_func(steady_state_results, reverse_rates)
    return steady_state_results


def num_unsteadied(comparison):
    return np.sum(np.abs(comparison) > 0.01)


def get_full_steady_states(starting_state, total_time, reverse_rates, sim_model, params):
    sim_func = get_loop_simfunc(
        params, sim_model=sim_model, saveat=dfx.SaveAt(t1=True))
    ti = 0
    starting_state = np.repeat(np.expand_dims(starting_state.concentrations, axis=0),
                               repeats=reverse_rates.shape[0], axis=0)
    comparison = starting_state
    steady_state_results = loop_sim(
        starting_state, reverse_rates,
        sim_func=sim_func)
    tf = steady_state_results.stats['num_accepted_steps'][0]
    steady_state_results = np.array(steady_state_results.ys[:, tf-1, :])
    ti += params.t_end - params.t_start
    # for ti in range(int(params.t_end), total_time, int(params.t_end)):
    iter_time = datetime.now()
    while (num_unsteadied(comparison - steady_state_results) > 0) and (ti < total_time):
        print('Steady states: ', ti, ' iterations. ', num_unsteadied(
            comparison - steady_state_results), ' left to steady out. ', datetime.now() - iter_time)
        comparison = steady_state_results
        ti += params.t_end - params.t_start
        steady_state_results = loop_sim(
            steady_state_results, reverse_rates, sim_func)
        tf = steady_state_results.stats['num_accepted_steps'][0]
        steady_state_results = np.array(steady_state_results.ys[:, tf-1, :])
    return steady_state_results


def get_final_states(steady_states, reverse_rates, sim_func, t, ti, full_final_states):
    final_states_results = loop_sim(
        steady_states, reverse_rates, sim_func)
    tf = final_states_results.stats['num_accepted_steps'][0]
    t = np.concatenate(
        [t, np.array(final_states_results.ts[0, :tf]) + ti], axis=0)
    final_states_results = np.array(final_states_results.ys[:, :tf, :])
    full_final_states = np.concatenate(
        [full_final_states, final_states_results], axis=1)
    return final_states_results, full_final_states, t


def get_full_final_states(steady_states, reverse_rates, total_time, new_model, params):
    sim_func = get_loop_simfunc(
        params, sim_model=new_model, saveat=dfx.SaveAt(t1=True))
    # sim_func = get_loop_simfunc(params, sim_model=new_model, saveat=dfx.SaveAt(ts=np.linspace(params.t_start, params.t_end, 100)))
    final_states_results = steady_states
    full_final_states = np.expand_dims(steady_states, axis=1)
    t = np.array([0])
    ti = 0

    final_states_results, full_final_states, t = get_final_states(
        steady_states, reverse_rates, sim_func, t, ti, full_final_states)
    iter_time = datetime.now()
    while (num_unsteadied(steady_states - final_states_results[:, -1, :]) > 0) and (ti < total_time):
        print('Final states: ', ti, ' iterations. ', num_unsteadied(steady_states -
              final_states_results[:, -1, :]), ' left to steady out. ', datetime.now() - iter_time)
        steady_states = final_states_results[:, -1, :]
        ti += params.t_end - params.t_start
        final_states_results, full_final_states, t = get_final_states(
            steady_states, reverse_rates, sim_func, t, ti, full_final_states)
    return full_final_states, t


def get_analytics(steady_states, final_states, t, K_eqs, model,
                  species_names, species_types, input_species_idx,
                  starting_circuit_idx):
    batchsize = steady_states.shape[0]

    peaks = jnp.where(steady_states != final_states.max(
        axis=1), final_states.max(axis=1), final_states.min(axis=1))
    precision = np.array(jax.jit(jax.vmap(partial(get_precision, signal_idx=input_species_idx)))(
        starting_states=steady_states,
        steady_states=final_states[:, -1, :]
    ))
    sensitivity = np.array(jax.jit(jax.vmap(partial(get_sensitivity, signal_idx=input_species_idx)))(
        peaks=peaks, starting_states=steady_states
    ))
    clear_gpu()
    resp_vmap = jax.jit(jax.vmap(partial(get_step_response_times,
                                         t=t, signal_time=0.0, signal_idx=input_species_idx)))  # np.expand_dims(t, 0)
    response_times = np.array(resp_vmap(
        deriv=np.gradient(np.swapaxes(final_states, 1, 2), axis=-1),
        data=np.swapaxes(final_states, 1, 2), steady_states=np.expand_dims(
            np.swapaxes(final_states, 1, 2)[:, :, -1], axis=-1)))

    analytics = {
        'precision': precision.flatten(),
        'sensitivity': sensitivity.flatten(),
        'response_times': response_times.flatten()
    }
    analytics_df = pd.DataFrame.from_dict(analytics)
    analytics_df['circuit_name'] = np.repeat(np.expand_dims(
        np.arange(starting_circuit_idx, starting_circuit_idx+batchsize), axis=0), len(model.species), axis=1)[0]
    analytics_df['relative_circ_strength'] = np.repeat(np.expand_dims(
        [K_eqs[i].mean() for i in range(batchsize)], axis=0), len(model.species), axis=1)[0]
    analytics_df['sample_name'] = flatten_listlike(
        [species_names for i in range(batchsize)])
    analytics_df['species_type'] = flatten_listlike(
        [species_types for i in range(batchsize)])
    return analytics_df


def plot_scan(final_states, t_final, bi, species_names_onlyin, num_show_species, num_show_species_i=0):
    max_per_plot = np.min([final_states.shape[0], 49])
    num_plots = np.max([1, int(np.ceil(final_states.shape[0] / max_per_plot))])
    for p in tqdm(range(num_plots)):
        num_rows = int(np.sqrt(max_per_plot))
        plt.figure(figsize=(10, 10))
        for i in range(num_rows**2):
            ii = (p*(num_rows**2))+i
            if ii >= final_states.shape[0]:
                break
            ax = plt.subplot(num_rows, num_rows, i+1)
            plt.plot(t_final, final_states[ii, :,
                     num_show_species_i:num_show_species])
            plt.title(str(bi+ii))
        plt.legend(species_names_onlyin)
        plt.savefig(os.path.join('output', '5_Keqs_exp',
                    f'final_states_{(bi+p)*max_per_plot}-{(bi+p)*max_per_plot+max_per_plot}_{num_show_species_i}{num_show_species}.svg'))
        plt.close()


def adjust_sim_params(reverse_rates, max_kd):
    dt = 0.1

    # params_steady = BasicSimParams(total_time=10.0, delta_t=dt)
    # total_t_steady = 300
    # params_final = BasicSimParams(total_time=10.0, delta_t=dt)
    # total_t_final = 100
    params_steady = MedSimParams(t_start=0, t_end=1000.0, delta_t=dt,
                                 poisson_sim_reactions=None, brownian_sim_reaction=None)
    total_t_steady = 3000000
    params_final = MedSimParams(t_start=0, t_end=1000.0, delta_t=dt,
                                poisson_sim_reactions=None, brownian_sim_reaction=None)
    total_t_final = 3000000

    return params_steady, total_t_steady, params_final, total_t_final


def scan_all_params(kds, K_eqs, b_reverse_rates, model):
    max_kds = kds.max()
    batchsize = int(b_reverse_rates.shape[0] / 2)
    target = 0.3
    a_sig = a * 1
    a_sig[input_species_idx] = a[input_species_idx] * (1 + target)
    model_steady = update_model_rates(model, a=a)
    model_sig = update_model_rates(deepcopy(model), a=a_sig)

    species_names = [s.name for s in model.species]
    species_names_onlyin = species_names[:num_species]
    species_types = ['unbound' if s in model.species[:num_species]
                     else 'bound' for s in model.species]

    analytics_df = pd.DataFrame()

    for bi in range(0, b_reverse_rates.shape[0], batchsize):
        clear_gpu()
        bf = min(bi+batchsize, b_reverse_rates.shape[0])
        reverse_rates = b_reverse_rates[bi:bf]
        starting_state = BasicSimState(concentrations=s0)

        params_steady, total_t_steady, params_final, total_t_final = adjust_sim_params(
            reverse_rates, max_kds)

        steady_states = get_full_steady_states(
            starting_state, total_time=total_t_steady, reverse_rates=reverse_rates, sim_model=convert_model(
                model_steady),
            params=params_steady)
        clear_gpu()
        final_states, t_final = get_full_final_states(
            steady_states, reverse_rates, total_t_final, convert_model(model_sig), params=params_final)
        plot_scan(final_states, t_final, bi, species_names, len(species_names))
        plot_scan(final_states, t_final, bi, species_names_onlyin, num_species)
        plot_scan(final_states, t_final, bi,
                  species_names_onlyin[1:num_species], num_species, 1)

        clear_gpu()
        analytics = get_analytics(
            steady_states, final_states, t_final, K_eqs[bi:bf], model,
            species_names, species_types, input_species_idx, bi)
        analytics_df = pd.concat([analytics_df, analytics], axis=0)
        analytics_df.to_csv(os.path.join('output', '5_Keqs_exp', 'df.csv'))

        np.save(os.path.join('output', '5_Keqs_exp', f't_{bi}.npy'), t_final)
        np.save(os.path.join('output', '5_Keqs_exp', f'final_states_{bi}-{bf}.npy'),
                final_states[bi:bf, :, :])

    return analytics_df


num_species = 3
input_species_idx = 0
output_species_idx = 1

Keq = np.array(
    [[1, 1, 1],
     [1, 1, 1],
     [1, 1, 1]]
)
# From src/utils/common/configs/RNA_circuit/molecular_params.json
a = np.ones(3) * 0.08333
a[1] = a[1] * 5
a[2] = a[2] * 0.3
d = np.ones(3) * 0.0008333
d[0] = d[0] * 0.5
d[1] = d[1] * 1
d[2] = d[2] * 1
ka = np.ones_like(Keq) * per_mol_to_per_molecule(1000000)
kd = ka/Keq

model = construct_model_fromnames([str(i) for i in range(num_species)])
model.species = model.species[-num_species:] + model.species[:-num_species]
model = update_model_rates(model, a, d, ka, kd)

s0 = np.concatenate(
    [np.array([1.0, 1.0, 1.0]), np.zeros(len(model.species[num_species:]))])
starting_state = BasicSimState(concentrations=s0)


K_eqs_range = np.array([0.01, 0.5, 1.0, 5, 40])
num_Keqs = np.size(K_eqs_range)
num_unique_interactions = np.math.factorial(num_species)


# Set loop vars
analytic_types = ['precision']
all_analytic_matrices = define_matrices(
    num_species, num_Keqs, num_unique_interactions, len(analytic_types))

total_iterations = np.power(num_Keqs, num_unique_interactions)
total_processes = 1
sub_process = 0
num_iterations = int(total_iterations / total_processes)

print('Making ', num_iterations,
      ' parameter combinations for equilibrium constants range: ', K_eqs_range)


def make_keqs():
    K_eqs = np.zeros((num_iterations, num_species, num_species))
    pows = np.power(num_Keqs, np.arange(num_unique_interactions))
    for i in range(num_iterations):
        interaction_strength_choices = np.floor(
            np.mod(i / pows, num_Keqs)).astype(int)
        if interaction_strength_choices[1] < interaction_strength_choices[3]:
            continue
        flat_triangle = K_eqs_range[list(
            interaction_strength_choices)]
        K_eqs[i] = make_symmetrical_matrix_from_sequence(
            flat_triangle, num_species)
    return K_eqs


K_eqs = jax.jit(make_keqs, backend='cpu')()
didx = np.flatnonzero((K_eqs == 0).all((1, 2)))
K_eqs = np.delete(K_eqs, didx, axis=0)

print(K_eqs.shape[0], 'unique combinations to scan.')

kds = ka / K_eqs
sim_model = convert_model(model)
b_reverse_rates = make_reverse_rates(model, sim_model.reverse_rates, kds)


clear_gpu()
model = update_model_rates(model, a=a)
batchsize = len(K_eqs)
analytics_df = scan_all_params(
    kds[:batchsize], K_eqs[:batchsize], b_reverse_rates[:batchsize], model)
analytics_df.to_csv(os.path.join('output', '5_Keqs_exp', 'df.csv'))
