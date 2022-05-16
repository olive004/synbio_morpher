import logging
from src.clients.common.setup import compose_kwargs, instantiate_system, construct_signal


def construct_circuit_from_cfg(extra_configs: dict, config_filepath: str):
    kwargs = compose_kwargs(config_filepath=config_filepath, extra_configs=extra_configs)
    circuit = instantiate_system(kwargs)

    if kwargs.get("signal"):
        kwargs.get("signal")["identities_idx"] = circuit.species.identities['input']
        signal = construct_signal(kwargs.get("signal"))
        circuit.signal = signal
    return circuit
