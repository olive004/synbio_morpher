from copy import copy
import numpy as np


class Deterministic():
    def __init__(self, max_time=0, time_step=1) -> None:
        self.max_time = max_time
        self.time_step = time_step

    def dxdt_RNA(self, copynumbers, interactions, creation_rates, degradation_rates,
                 count_complexes=False):
        # dx_dt = a + x * k * x.T - x * ∂   for x=[A, B]

        xI = copynumbers * np.identity(np.shape(copynumbers)[0])
        coupling = np.matmul(xI * interactions, copynumbers)

        dxdt = - coupling - copynumbers * \
            degradation_rates + creation_rates
        if count_complexes:
            return (dxdt, coupling)
        return dxdt

    def plot(self, data, y=None, legend_keys=None, save_name='test_plot', new_vis=False):
        from src.srv.results.visualisation import VisODE
        data = data.T if len(legend_keys) == np.shape(data)[0] else data
        VisODE().plot(data, y, legend_keys, new_vis, save_name)
