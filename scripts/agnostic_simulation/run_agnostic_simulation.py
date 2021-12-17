from fire import Fire
import timeit
import sys

from src.utils.decorators import time_it
from src.utils.agnostic_system.base_system import BaseSystem


@time_it
def main(config_args=None):
    circuit = BaseSystem(config_args)
    circuit.visualise()

if __name__ == "__main__":
    Fire(main)
