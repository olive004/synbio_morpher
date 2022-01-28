import numpy as np


class Deterministic():
    def __init__(self) -> None:
        pass

    def dxdt_RNA(self, copynumbers, interactions, creation_rate, degradation_rate):
        k = interactions
        x = copynumbers
        return creation_rate + np.multiply(x, k, x.T) + x * degradation_rate

    def plot(self, data):
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(data)
        plt.savefig('test_plot.png')
