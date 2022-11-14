import logging
from typing import List
from bioreaction.model.data_containers import BasicModel
from src.utils.data.common import Data
from src.srv.io.manage.sys_interface import make_filename_safely
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.srv.io.manage.data_manager import DataManager
from src.utils.misc.io import isolate_filename
from src.utils.misc.type_handling import cast_all_values_as_list
from src.utils.signal.configs import get_signal_type, parse_sig_args
from src.utils.signal.signals import Signal
from src.utils.circuit.common.config_setup import parse_cfg_args, get_configs
from src.utils.circuit.common.system_setup import get_system_type


ESSENTIAL_KWARGS = [
    # "interactions",
    "interaction_simulator",
    "molecular_params",
    # "mutations",
    "system_type",
]


def expand_model_config(in_config: dict, out_config: dict, sample_names: List[str]) -> dict:
    if 'starting_concentration' not in out_config.keys():
        out_config['starting_concentration'] = {}
        for s in sample_names:
            out_config['starting_concentration'][s] = in_config['molecular_params'].get(
                'starting_copynumbers', 1)
    if in_config.get('interactions', {}).get('interactions_path') or in_config.get('interactions_path'):
        out_config['species_state'] = 'loaded'
        logging.warning(
            '\n\n\n\nNot implemented yet - to load interactions, modify model construciton\n\n\n\n')
    else:
        out_config['species_state'] = 'uninitialised'
    return out_config


def compose_kwargs(internal_configs: dict = None, config: dict = None) -> dict:
    """ Extra configs like data paths can be supplied here, eg. for circuits that were dynamically generated. """

    if internal_configs is not None:
        for kwarg, config in internal_configs.items():
            if kwarg != 'experiment':
                config[kwarg] = config

    data_manager = DataManager(filepath=make_filename_safely(config.get("data_path", None)),
                               identities=config.get("identities", {}),
                               data=config.get("data", None))
    config["molecular_params"] = load_json_as_dict(
        config.get("molecular_params"))
    kwargs = {}
    kwargs = expand_model_config(config, kwargs, data_manager.data.sample_names)

    kwargs.update({
        "data": data_manager.data,
        "data_path": data_manager.source,
        # For pre-loading interactions
        "interactions": config.get("interactions", {}),
        "interaction_simulator": config.get("interaction_simulator", {"name": "IntaRNA"}),
        "molecular_params": config["molecular_params"],
        "mutations": cast_all_values_as_list(config.get("mutations", {})),
        "name": isolate_filename(data_manager.data.source),
        "signal": load_json_as_dict(config.get("signal")),
        "simulation": config.get("simulation", {}),
        "system_type": config.get("system_type")
    })
    assert all([e in kwargs for e in ESSENTIAL_KWARGS]), 'Some of the kwargs for composing ' \
        f'a circuit are not listed in essential kwargs {ESSENTIAL_KWARGS}: {dict({e: e in kwargs for e in ESSENTIAL_KWARGS})}'
    return kwargs


def construct_circuit_from_cfg(extra_configs: dict, config_filepath: str = None, config_file: dict = None):

    config_file = get_configs(config_file, config_filepath)
    kwargs = compose_kwargs(internal_configs=extra_configs, config=config_file)
    circuit = instantiate_system(kwargs)

    if kwargs.get("signal"):
        signal = construct_signal(kwargs, circuit)
        circuit.signal = signal
    return circuit


def construct_signal(kwargs: dict, circuit) -> Signal:
    SignalType = get_signal_type(kwargs["signal"].get("signal_type"))
    return SignalType(**parse_sig_args(kwargs, circuit))


def instantiate_system(kwargs):
    system_cfg_args = parse_cfg_args(kwargs)
    SystemType = get_system_type(kwargs["system_type"])
    return SystemType(system_cfg_args)
