from src.clients.common.setup import compose_kwargs, instantiate_system, construct_signal


def construct_circuit_from_cfg(config_file):
    kwargs = compose_kwargs(config_file)
    circuit = instantiate_system(kwargs)

    kwargs.get("signal")["identities_idx"] = circuit.species.identities['input']
    signal = construct_signal(kwargs.get("signal"))
    circuit.signal = signal
    return circuit
