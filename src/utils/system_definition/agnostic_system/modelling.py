from copy import copy
import logging
import numpy as np


class Deterministic():
    def __init__(self) -> None:
        pass

    def dxdt_RNA(self, copynumbers, interactions, creation_rates, degradation_rates):
        k = 0.1 * interactions
        x = 0.1 * copynumbers
        print(interactions)
        print(copynumbers)
        print(creation_rates)
        print(degradation_rates)
        print(np.matmul(np.matmul(x, k), x.T))
        print(np.matmul(x, k))
        logging.debug(creation_rates + np.matmul(x, k) + x * degradation_rates)
        return creation_rates + np.matmul(np.matmul(x, k), x.T) + x * degradation_rates

    def plot(self, data, legend_keys):
        from src.utils.visualisation.graph_drawer import VisODE
        VisODE().plot(data, legend_keys)
