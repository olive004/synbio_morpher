from fire import Fire

from src.utils.misc.decorators import time_it
from src.utils.circuit.agnostic_circuits.base_circuit import BaseSystem


@time_it
def main(config_args=None):
    circuit = BaseSystem(config_args)
    circuit.visualise()

if __name__ == "__main__":
    Fire(main)
