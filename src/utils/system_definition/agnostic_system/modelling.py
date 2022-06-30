import numpy as np


class Deterministic():
    def __init__(self, max_time=0, time_step=1) -> None:
        self.max_time = max_time
        self.time_step = time_step

    def dxdt_RNA(self, t, copynumbers, interactions, creation_rates, degradation_rates,
                 num_samples, signal=None, signal_idx=None):
        """ dx_dt = a - x * I * k * x' - x * ∂   for x=[A, B] 
        Data in format [sample, timestep] or [sample,]"""
        if signal_idx is not None:
            copynumbers[signal_idx] = signal

        xI = copynumbers * np.identity(num_samples)
        coupling = np.matmul(np.matmul(xI, interactions), copynumbers.T)
        
        dxdt = creation_rates.flatten() - coupling.flatten() - \
            copynumbers.flatten() * degradation_rates.flatten()

        return dxdt

    def plot(self, data, y=None, out_path='test_plot', new_vis=False, out_type='png',
    **plot_kwrgs):
        from src.src.utils.results.visualisation import VisODE
        data = data.T if len(plot_kwrgs.get('legend', [])) == np.shape(data)[0] else data
        VisODE().plot(data, y, new_vis, out_path=out_path, out_type=out_type, **plot_kwrgs)
