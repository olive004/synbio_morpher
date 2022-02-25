from fire import Fire
import os

from src.utils.misc.decorators import time_it
from src.clients.common.setup import compose_kwargs, instantiate_system
from src.utils.signal.inputs import Signal


@time_it
def main(config_file=None):

    from src.utils.data.fake_data_generation.nc_sequences import create_toy_circuit
    create_toy_circuit(count=3, slength=25, protocol="template_mix")

    config_file = os.path.join("scripts", "RNA_circuit_simulation", "configs", "toy_RNA.json")
    kwargs = compose_kwargs(config_file)
    circuit = instantiate_system(kwargs)
    signal = Signal(in_magnitude=110, total_time=300)
    circuit.simulate_signal(signal)
    circuit.visualise(new_vis=False)


if __name__ == "__main__":
    Fire(main)
