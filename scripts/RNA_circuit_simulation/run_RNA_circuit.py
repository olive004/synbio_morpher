from fire import Fire

from src.utils.misc.decorators import time_it
from src.clients.common.setup import compose_kwargs, instantiate_system


@time_it
def main(config_file=None):

    kwargs = compose_kwargs(config_file)
    circuit = instantiate_system(kwargs)
    circuit.visualise()


if __name__ == "__main__":
    Fire(main)
