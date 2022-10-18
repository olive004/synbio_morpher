import logging
import numpy as np
from numba import jit
from numba import cuda
from numba import float32
from src.utils.misc.numerical import SCIENTIFIC
from src.utils.misc.units import per_mol_to_per_molecules


# @jit(nopython=True)
def dxdt_RNA(t, copynumbers, interactions, creation_rates, degradation_rates,
             num_samples, time_interval, signal=None, signal_idx=None):
    """ dx_dt = a - x * I * k * x' - x * ∂   for x=[A, B] 
    Data in format [sample, timestep] or [sample,]"""
    if signal_idx is not None:
        copynumbers[signal_idx] = signal

    xI = copynumbers * np.identity(num_samples)
    interactions_xI = np.matmul(xI, interactions)
    coupling = np.matmul(interactions_xI, copynumbers.T)

    dxdt = creation_rates.flatten() - coupling.flatten() - \
        copynumbers.flatten() * degradation_rates.flatten()

    return np.multiply(dxdt, time_interval)


class Modeller():
    """ Common modeller class. For example deterministic vs. 
    stochastic modelling share some things """
    def __init__(self, max_time=0, time_interval=1) -> None:
        """ Time step is the dt """
        self.max_time = int(max_time / time_interval)
        self.original_max_time = max_time
        self.time_interval = time_interval

    def dxdt_RNA(self):
        pass


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
        
        # full_interactions = np.divide(k_a, (k_d + degradation_rates.flatten()))
        coupling = np.matmul(np.matmul(xI, full_interactions), copynumbers.T)
        # coupling = np.matmul(np.matmul(xI, interactions), copynumbers.T)
        # coupling = np.matmul(np.matmul(xI, np.divide(interactions, SCIENTIFIC['mole'])), copynumbers.T)

        dxdt = creation_rates - coupling.flatten() - \
            copynumbers.flatten() * degradation_rates

        # return dxdt
        return np.multiply(dxdt, self.time_interval)

    def plot(self, data, y=None, out_path='test_plot', new_vis=False, out_type='svg',
             **plot_kwrgs):
        from src.utils.results.visualisation import VisODE
        data = data.T if len(plot_kwrgs.get('legend', [])
                             ) == np.shape(data)[0] else data
        VisODE().plot(data, y, new_vis=new_vis, out_path=out_path, out_type=out_type, **plot_kwrgs)
