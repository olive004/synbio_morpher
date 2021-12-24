from fire import Fire

from src.utils.decorators import time_it
from src.utils.system_definition.specific_systems import RNASystem


@time_it
def main(config_args=None):

    circuit = RNASystem(config_args)
    circuit.visualise()


if __name__ == "__main__":
    Fire(main)
