import numpy as np
import logging


class Deterministic():
    def __init__(self, max_time=0, time_step=1) -> None:
        self.max_time = max_time
        self.time_step = time_step

    def dxdt_RNA(self, copynumbers, interactions, creation_rates, degradation_rates):
        """ dx_dt = a - x * I * k * x - x * ∂   for x=[A, B] """
        xI = copynumbers * np.identity(np.shape(copynumbers)[0])
        coupling = np.matmul(np.matmul(xI, interactions), copynumbers)  

        dxdt = creation_rates - coupling - copynumbers * degradation_rates

        return dxdt

    def xt_RNA(self, copynumbers, t, interactions, creation_rates, degradation_rates):
        """ x(t) = a * t - x * I * k * x * t - x * ∂ * t """
        x = copynumbers
        k = interactions
        xt = creation_rates * t - x * np.identity(np.shape(x)[0]) * k * x * t - x * degradation_rates * t
        return xt

    def plot(self, data, y=None, legend_keys=None, save_name='test_plot', new_vis=False):
        from src.srv.results.visualisation import VisODE
        data = data.T if len(legend_keys) == np.shape(data)[0] else data
        VisODE().plot(data, y, legend_keys, new_vis, save_name)
