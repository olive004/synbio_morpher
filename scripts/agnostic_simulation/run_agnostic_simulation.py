from fire import Fire

from src.utils.misc.decorators import time_it
from src.utils.circuit.agnostic_circuits.base_circuit import BaseCircuit


@time_it
def main(config_args=None):
    circuit = BaseCircuit(config_args)
    circuit.visualise()

if __name__ == "__main__":
    Fire(main)
