from copy import copy
import numpy as np


class Deterministic():
    def __init__(self) -> None:
        pass

    def dxdt_RNA(self, copynumbers, interactions, creation_rates, degradation_rates):
        # dx_dt = a + x * k * x.T - x * ∂   for x=[A, B] 
        
        x = copynumbers
        xI = copynumbers * np.identity(len(copynumbers))

        import logging
        FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
        FORMAT = "%(filename)s:%(funcName)s():%(lineno)i: %(message)s %(levelname)s"
        logging.basicConfig(level=logging.INFO, format=FORMAT)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # logger.info(xI * interactions)
        # logger.info(np.matmul(xI * interactions, x))
        dxdt = np.matmul(xI * interactions, x) - x * degradation_rates + creation_rates
        dxdt = - x * degradation_rates + creation_rates
        # logger.info(dxdt)
        return dxdt

    def plot(self, data, legend_keys, new_vis=False):
        from src.srv.results.visualisation import VisODE
        VisODE().plot(data, legend_keys, new_vis)
