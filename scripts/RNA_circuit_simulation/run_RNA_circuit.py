from fire import Fire
import os

from src.utils.misc.decorators import time_it
from src.clients.common.setup import compose_kwargs, instantiate_system, construct_signal


@time_it
def main(config_file=None):

    from src.utils.data.fake_data_generation.seq_generator import RNAGenerator
    RNAGenerator(purpose='example_data').generate_circuit(
        count=3, slength=25, protocol="template_mix")

    config_file = os.path.join(
        "scripts", "RNA_circuit_simulation", "configs", "toy_RNA.json")
    kwargs = compose_kwargs(config_file)
    circuit = instantiate_system(kwargs)

    kwargs.get("signal")[
        "identities_idx"] = circuit.species.identities['input']
    signal = construct_signal(kwargs.get("signal"))
    signal.show()

    circuit.simulate_signal(signal)
    circuit.visualise(new_vis=False)


if __name__ == "__main__":
    Fire(main)
