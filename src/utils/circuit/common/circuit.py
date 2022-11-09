import logging
from src.clients.common.setup import compose_kwargs, instantiate_system, construct_signal
from src.srv.io.manage.sys_interface import make_filename_safely
from src.utils.data.data_format_tools.common import load_json_as_dict


def get_configs(config_file, config_filepath):
    config_filepath = make_filename_safely(config_filepath)
    if config_file is None and config_filepath:
        config_file = load_json_as_dict(config_filepath)
    elif config_file and config_filepath:
        raise ValueError(
            'Both a config and a config filepath were defined - only use one config option.')
    elif config_file is None and config_filepath is None:
        raise ValueError('Config file or path needed as input to function.')
    return config_file


# @time_it
def construct_circuit_from_cfg(extra_configs: dict, config_filepath: str = None, config_file: dict = None):

    config_file = get_configs(config_file, config_filepath)
    kwargs = compose_kwargs(config_filepath=config_filepath,
                            extra_configs=extra_configs, config_file=config_file)
    circuit = instantiate_system(kwargs)

    if kwargs.get("signal"):
        kwargs.get("signal")[
            "identities_idx"] = circuit.species.identities['input']
        signal = construct_signal(kwargs.get("signal"))
        circuit.signal = signal
    return circuit
