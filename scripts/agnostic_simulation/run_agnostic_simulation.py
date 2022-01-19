from fire import Fire

from src.utils.decorators import time_it
from src.utils.system_definition.agnostic_system.base_system import BaseSystem


@time_it
def main(config_args=None):
    circuit = BaseSystem(config_args)
    circuit.visualise()

if __name__ == "__main__":
    Fire(main)
