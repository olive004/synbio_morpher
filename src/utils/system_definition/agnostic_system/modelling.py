from copy import copy
import numpy as np


class Deterministic():
    def __init__(self) -> None:
        pass

    def dxdt_RNA(self, copynumbers, interactions, creation_rates, degradation_rates):
        # dx_dt = a + x * k * x.T - x * ∂   for x=[A, B] 
        
        x = copynumbers
        I = np.identity(len(copynumbers))
        dxdt = x * I * x - x * degradation_rates + creation_rates
        return dxdt

    def plot(self, data, legend_keys, new_vis=False):
        from src.srv.results.visualisation import VisODE
        VisODE().plot(data, legend_keys, new_vis)
