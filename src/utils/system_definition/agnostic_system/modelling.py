from copy import copy
import numpy as np


class Deterministic():
    def __init__(self) -> None:
        pass

    def dxdt_RNA(self, copynumbers, interactions, creation_rates, degradation_rates):
        k = interactions
        x = copynumbers
        # print(interactions)
        # print(copynumbers)
        # print(creation_rates)
        # print(degradation_rates)
        print(x.T)
        print(creation_rates + np.matmul(np.matmul(x, k), x.T) + x * degradation_rates)
        return creation_rates + np.matmul(np.matmul(x, k), x.T) + x * degradation_rates

    def plot(self, data):
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(data)
        plt.savefig('test_plot.png')
