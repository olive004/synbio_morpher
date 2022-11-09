import logging
from src.srv.io.manage.sys_interface import make_filename_safely
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.srv.io.manage.data_manager import DataManager
from src.utils.misc.io import isolate_filename
from src.utils.misc.type_handling import cast_all_values_as_list
from src.utils.signal.configs import get_signal_type, parse_sig_args
from src.utils.signal.signals import Signal
from src.utils.circuit.config import parse_cfg_args
from src.utils.circuit.setup import get_system_type


ESSENTIAL_KWARGS = [
    # "interactions",
    "interaction_simulator",
    "molecular_params",
    # "mutations",
    "system_type",
]


def compose_kwargs(extra_configs: dict = None, config_filepath: str = None, config_file: dict = None) -> dict:
    """ Extra configs like data paths can be supplied here, eg. for circuits that were dynamically generated. """
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
    config_file = get_configs(config_file, config_filepath)

    if extra_configs is not None:
        for kwarg, config in extra_configs.items():
            if kwarg != 'experiment': # and config_file.get(kwarg):
                config_file[kwarg] = config
        # config_file.update(extra_configs)
    data_manager = DataManager(filepath=make_filename_safely(config_file.get("data_path", None)),
                               identities=config_file.get("identities", {}),
                               data=config_file.get("data", None),
                               sample_names=config_file.get("sample_names", None))

    if type(config_file.get("molecular_params")) == dict:
        raise ValueError(
            f'The moelcular parameters {config_file.get("molecular_params")} supplied for the circuit should link to a file')
    kwargs = {
        "data": data_manager.data,
        "data_path": data_manager.source,
        "identities": data_manager.data.identities,
        # For pre-loading interactions
        "interactions": config_file.get("interactions", {}),
        "interaction_simulator": config_file.get("interaction_simulator", {}),
        "molecular_params": load_json_as_dict(config_file.get("molecular_params")),
        "mutations": cast_all_values_as_list(config_file.get("mutations", {})),
        "name": isolate_filename(data_manager.data.source),
        "signal": load_json_as_dict(config_file.get("signal")),
        "system_type": config_file.get("system_type")
    }
    assert all([e in kwargs for e in ESSENTIAL_KWARGS]), 'Some of the kwargs for composing ' \
        f'a circuit are not listed in essential kwargs {ESSENTIAL_KWARGS}: {dict({e: e in kwargs for e in ESSENTIAL_KWARGS})}'
    return kwargs


def construct_signal(kwargs) -> Signal:
    SignalType = get_signal_type(kwargs.get("signal_type"))
    return SignalType(**parse_sig_args(kwargs))


def instantiate_system(kwargs):
    system_cfg_args = parse_cfg_args(kwargs)
    SystemType = get_system_type(kwargs["system_type"])
    return SystemType(system_cfg_args)
