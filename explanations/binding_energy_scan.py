from bioreaction.model.data_tools import construct_model_fromnames
from bioreaction.simulation.simfuncs.basic_de import bioreaction_sim_expanded
from bioreaction.simulation.basic_sim import basic_de_sim, convert_model, BasicSimParams, BasicSimState

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
jax.config.update('jax_platform_name', 'gpu')
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


def get_steady_states(starting_state, reverse_rates, sim_model, params):

    def to_vmap_basic(reverse_rates):
        sim_model.reverse_rates = reverse_rates
        return basic_de_sim(starting_state=starting_state, model=sim_model, params=params)

    steady_state_results = jax.jit(jax.vmap(to_vmap_basic))(reverse_rates)
    return steady_state_results


def loop_sim(steady_state_results, params, reverse_rates, sim_model):

    def to_vmap_basic(starting_state, reverse_rates):
        sim_model.reverse_rates = reverse_rates
        return basic_de_sim(starting_state=BasicSimState(concentrations=starting_state), model=sim_model, params=params)

    steady_state_results = jax.jit(jax.vmap(to_vmap_basic))(
        steady_state_results[0], reverse_rates)
    return steady_state_results


def get_full_steady_states(starting_state, total_time, reverse_rates, sim_model, params):
    # all_steady_states = []
    # for i in range(0, reverse_rates.shape[0]):
    #     rate_max = np.max([reverse_rates[i], sim_model.forward_rates])

    #     steady_state_result = scipy.integrate.solve_ivp(
    #         partial(bioreaction_sim_expanded, args=None,
    #                 inputs=sim_model.inputs, outputs=sim_model.outputs,
    #                 forward_rates=sim_model.forward_rates, reverse_rates=reverse_rates[i]),
    #         (0, params.total_time),
    #         y0=starting_state.concentrations,
    #         method='DOP853')
    #     if not steady_state_result.success:
    #         raise ValueError(
    #             'Steady state could not be found through solve_ivp')
    #     all_steady_states.append(steady_state_result.y)
    # return np.array(all_steady_states)
    v = datetime.now()

    sim_func = jax.jit(jax.vmap(
        partial(bioreaction_sim_dfx_expanded,
                t0=0, t1=params.total_time * 3, dt0=params.delta_t,
                signal=None, signal_onehot=None,  # signal.reactions_onehot,
                forward_rates=sim_model.forward_rates,
                inputs=sim_model.inputs,
                outputs=sim_model.outputs,
                y0=starting_state.concentrations,
                solver=dfx.Heun(),
                max_steps=16**6
                )), backend='cpu')
    steady_state_results = sim_func(reverse_rates=reverse_rates)
    tf = steady_state_results.stats['num_accepted_steps'][0]

    vv = datetime.now() - v
    plt.plot(steady_state_results.ys[0])
    plt.savefig('test.png')
    # steady_state_results = get_steady_states(
    #     starting_state, reverse_rates, sim_model, params)
    # for ti in range(0, total_time, int(params.total_time)):
        # steady_state_results = loop_sim(
        #     steady_state_results, params, reverse_rates, sim_model)
    # for ti in range(0, total_time, int(params.total_time)):
    #     steady_state_results = loop_sim(steady_state_results, params, reverse_rates, sim_model)
    return np.array(steady_state_results.ys[:, :tf, :])


def get_full_final_states(steady_states, reverse_rates, total_time, new_model, params):
    # final_states_results = [steady_states]
    # t_steps = max(int(total_time/params.delta_t/1000), 1)
    # final_states = np.expand_dims(steady_states, axis=1)
    # for ti in range(0, total_time, int(params.total_time)):
    #     final_states_results = loop_sim(
    #         final_states_results, params, reverse_rates, new_model)
    #     t_steps = max(int(final_states_results[1].shape[1]/1000), 1)
    #     final_states = np.array(np.concatenate(
    #         [final_states, final_states_results[1][:, ::t_steps, :]], axis=1))
    # t_final = np.arange(
    #     final_states.shape[1]) * params.delta_t
    # return final_states, t_final

    sim_func = jax.jit(jax.vmap(
        partial(bioreaction_sim_dfx_expanded,
                t0=0, t1=params.total_time * 3, dt0=params.delta_t,
                signal=None, signal_onehot=None,  # signal.reactions_onehot,
                forward_rates=sim_model.forward_rates,
                inputs=sim_model.inputs,
                outputs=sim_model.outputs,
                solver=dfx.Heun(),
                max_steps=16**6
                )), backend='cpu')
    final_states_results = sim_func(y0=steady_states, reverse_rates=reverse_rates)
    tf = final_states_results.stats['num_accepted_steps'][0]
    return np.array(final_states_results.ys[:, :tf, :]), np.array(final_states_results.ts[0, :tf, :])


def get_analytics(steady_states, final_states, t, K_eqs, model, species_names, species_types, starting_circuit_idx):
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
                        signal_idx=input_species_idx, t=np.expand_dims(t, 0), signal_time=0.0)))
    response_times = np.array(resp_vmap(
        data=np.swapaxes(final_states, 1, 2), steady_states=np.expand_dims(steady_states, axis=2),
        deriv=jnp.gradient(np.swapaxes(final_states, 1, 2))[0]))

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


def plot_scan(final_states, t_final, bi, species_names_onlyin, num_show_species):
    num_rows = int(np.sqrt(final_states.shape[0]))
    plt.figure(figsize=(50, 50))
    for i in range(num_rows**2):
        ax = plt.subplot(num_rows, num_rows, i+1)
        plt.plot(t_final[::100], final_states[i, ::100, :num_show_species])
        plt.title(str(bi+i))
    plt.legend(species_names_onlyin)
    plt.savefig(os.path.join('output', '5_Keqs_ka',
                f'final_states_{bi}-{bi+final_states.shape[0]}_{num_show_species}.svg'))
    plt.close()


def adjust_sim_params(reverse_rates, max_kd):
    dt_scale = np.round(reverse_rates.max() / max_kd, 1)
    dt = 0.1 / dt_scale

    # params_steady = BasicSimParams(total_time=10.0, delta_t=dt)
    # total_t_steady = 300
    # params_final = BasicSimParams(total_time=10.0, delta_t=dt)
    # total_t_final = 100
    params_steady = BasicSimParams(total_time=100000.0, delta_t=dt)
    total_t_steady = 3000000
    params_final = BasicSimParams(total_time=100000.0, delta_t=dt)
    total_t_final = 1000000

    return params_steady, total_t_steady, params_final, total_t_final


def scan_all_params(kds, K_eqs, b_reverse_rates, model):
    max_kds = kds.max()
    batchsize = 40
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

    for bi in tqdm(range(0, b_reverse_rates.shape[0], batchsize)):
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

        clear_gpu()
        analytics = get_analytics(
            steady_states, final_states, t_final, K_eqs[bi:bf], model, species_names, species_types, bi)
        analytics_df = pd.concat([analytics_df, analytics], axis=0)
        analytics_df.to_csv(os.path.join('output', '5_Keqs_ka', 'df.csv'))

    return analytics_df


num_species = 3
input_species_idx = 0
output_species_idx = 1

Keq = np.array(
    [[1, 2, 1],
     [2, 1, 0.5],
     [1, 0.5, 2]]
)
# From src/utils/common/configs/RNA_circuit/molecular_params.json
a = np.ones(3) * 0.08333
a[1] = a[1] * 1.5
a[2] = a[2] * 0.8
d = np.ones(3) * 0.0008333
ka = np.ones_like(Keq) * per_mol_to_per_molecule(1000000)
kd = ka/Keq

model = construct_model_fromnames([str(i) for i in range(num_species)])
model.species = model.species[-num_species:] + model.species[:-num_species]
model = update_model_rates(model, a, d, ka, kd)

s0 = np.concatenate(
    [np.array([1.0, 1.0, 1.0]), np.zeros(len(model.species[num_species:]))])
starting_state = BasicSimState(concentrations=s0)


K_eqs_range = np.array([0.1, 0.5, 1.0, 1.5, 9])
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


kds = ka / K_eqs
sim_model = convert_model(model)
b_reverse_rates = make_reverse_rates(model, sim_model.reverse_rates, kds)


clear_gpu()
model = update_model_rates(model, a=a)
batchsize = 3 # len(K_eqs)
analytics_df = scan_all_params(kds[:batchsize], K_eqs[:batchsize], b_reverse_rates[:batchsize], model)
analytics_df.to_csv(os.path.join('output', '5_Keqs_ka', 'df.csv'))
