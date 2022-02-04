from copy import copy
import logging
import numpy as np


class Deterministic():
    def __init__(self) -> None:
        pass

    def dxdt_RNA(self, copynumbers, interactions, creation_rates, degradation_rates):
        # dx_dt = a + x * k * x.T - x * ∂   for x=[A, B]
        
        k = 0.1 * interactions
        x = 0.005 * copynumbers
        alpha = 0.1 * creation_rates

        # Create middle term for each species
        # THERE'S DEFINITELY A MORE EFFICIENT WAY TO DO THIS
        coupling = np.zeros(copynumbers.shape)
        for current_species in range(len(copynumbers)):
            coupling[current_species] = np.sum(x * k[current_species] * x.T)

        print(interactions)
        print(copynumbers)
        print(coupling)
        # print(np.matmul(np.matmul(x, k), x.T))
        # print(np.matmul(x, k))
        logging.debug(alpha + coupling + x * degradation_rates)
        return alpha + coupling + x * degradation_rates

    def plot(self, data, legend_keys):
        from src.utils.visualisation.graph_drawer import VisODE
        VisODE().plot(data, legend_keys)
