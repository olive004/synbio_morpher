from src.clients.common.setup import compose_kwargs, instantiate_system, construct_signal


def construct_circuit_from_cfg(extra_configs, config_filepath: str):
    kwargs = compose_kwargs(config_filepath=config_filepath, extra_configs=extra_configs)
#     return construct_circuit(kwargs)


# def construct_circuit_from_dict(extra_configs, config: dict):
#     kwargs = compose_kwargs(config=config, extra_configs=extra_configs)
#     return construct_circuit(kwargs)


# def construct_circuit(kwargs):
    circuit = instantiate_system(kwargs)

    if kwargs.get("signal"):
        kwargs.get("signal")["identities_idx"] = circuit.species.identities['input']
        signal = construct_signal(kwargs.get("signal"))
        circuit.signal = signal
    return circuit