import logging
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfx
from bioreaction.simulation.simfuncs.basic_de import bioreaction_sim
from bioreaction.model.data_containers import QuantifiedReactions
from bioreaction.misc.misc import invert_onehot
from src.utils.modelling.base import Modeller


class Deterministic(Modeller):
    def __init__(self, max_time=0, time_interval=1) -> None:
        super().__init__(max_time, time_interval)

    def dxdt_RNA(self, t, copynumbers, full_interactions, creation_rates, degradation_rates,
                 signal=None, signal_idx=None, identity_matrix=None):
        """ dx_dt = a + x * I * k_d * x' - x * ∂   for x=[A, B] 
        x: the vector of copy numbers of the samples A, B, C...
        y: the vector of copy numbers of bound (nonfunctional) samples (AA, AB, AC...)
        a: the 'creation' rate, or for RNA samples, the transcription rate
        I: the identity matrix
        k_a: fixed binding rate of association between two RNA molecules (fixed at 1 or 1e6)
        k_d: a (symmetrical) matrix of interaction rates for the dissociation binding rate between each pair 
            of samples - self-interactions are included
        ∂: the 'destruction' rate, or for RNA samples, the (uncoupled) degradation rate
        Data in format [sample, timestep] or [sample,]"""

        if signal_idx is not None:
            copynumbers[signal_idx] = signal

        xI = copynumbers * identity_matrix
        coupling = np.matmul(np.matmul(xI, full_interactions), copynumbers.T)
        dxdt = creation_rates - coupling.flatten() - \
            copynumbers.flatten() * degradation_rates

        return np.multiply(dxdt, self.time_interval)

    def dxdt_RNA_jnp(self, copynumbers, t, full_interactions, creation_rates, degradation_rates,
                     identity_matrix, signal=None, signal_idx=None):

        if signal_idx is not None:
            copynumbers = copynumbers.at[signal_idx].set(signal)

        xI = copynumbers * identity_matrix
        coupling = jnp.matmul(jnp.matmul(xI, full_interactions), copynumbers.T)

        dxdt = creation_rates - coupling - \
            copynumbers * degradation_rates
        new_copynumbers = copynumbers + dxdt

        if signal_idx is not None:
            new_copynumbers = new_copynumbers.at[signal_idx].set(signal)

        return jnp.where(new_copynumbers < 0, 0, new_copynumbers)


def simulate_signal_scan(copynumbers, time, full_interactions, creation_rates, degradation_rates,
                         identity_matrix, signal, signal_idx, one_step_func):
    def to_scan(carry, thingy):
        t, s = thingy
        return one_step_func(carry, t, full_interactions, creation_rates, degradation_rates,
                             identity_matrix, s, signal_idx), carry
    return jax.lax.scan(to_scan, copynumbers, (time, signal))


def bioreaction_sim_full(qreactions: QuantifiedReactions, t0, t1, dt0,
                         signal, signal_onehot: np.ndarray,
                         y0=None,
                         solver=dfx.Tsit5(),
                         saveat=dfx.SaveAt(t0=True, t1=True, steps=True)):
    """ The signal is a function that takes in t 
    WARNING! Diffrax eqsolve will simulate to the end of the time series, 
    then set time to infinity for the remainder until max_steps have been reached. """

    term = dfx.ODETerm(partial(bioreaction_sim, reactions=qreactions.reactions, signal=signal,
                               signal_onehot=signal_onehot, inverse_onehot=invert_onehot(signal_onehot)))
    # y0 = qreactions.quantities if y0 is None else y0

    return dfx.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0,
                           y0=y0, saveat=saveat, max_steps=16**4)
