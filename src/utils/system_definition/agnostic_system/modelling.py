from copy import copy
import logging
import numpy as np


class Deterministic():
    def __init__(self) -> None:
        pass

    def dxdt_RNA(self, copynumbers, interactions, creation_rates, degradation_rates):
        # dx_dt = a + x * k * x.T - x * ∂   for x=[A, B]
        
        k = 0.1 * interactions
        x = 0.1 * copynumbers

        # Create middle term for each species
        coupling = np.zeros(copynumbers.shape)
        for current_species in range(len(copynumbers)):
            coupling[current_species] = x * k[current_species] * x.T

        print(interactions)
        print(coupling)
        # print(np.matmul(np.matmul(x, k), x.T))
        # print(np.matmul(x, k))
        logging.debug(creation_rates + coupling + x * degradation_rates)
        return creation_rates + coupling + x * degradation_rates

    def plot(self, data, legend_keys):
        from src.utils.visualisation.graph_drawer import VisODE
        VisODE().plot(data, legend_keys)
