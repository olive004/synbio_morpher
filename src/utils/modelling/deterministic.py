import logging
import numpy as np
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
