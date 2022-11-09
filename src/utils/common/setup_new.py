import logging
from bioreaction.model.data_tools import construct_model_fromnames
from src.srv.io.manage.sys_interface import make_filename_safely
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.srv.io.manage.data_manager import DataManager
from src.utils.misc.io import isolate_filename
from src.utils.misc.type_handling import cast_all_values_as_list
from src.utils.signal.configs import get_signal_type, parse_sig_args
from src.utils.signal.signals import Signal
from src.utils.circuit.common.config import parse_cfg_args, get_configs
from src.utils.circuit.common.system_setup import get_system_type


ESSENTIAL_KWARGS = [
    # "interactions",
    "interaction_simulator",
    "molecular_params",
    # "mutations",
    "system_type",
]


def construct_bioreaction_model(sample_names, molecular_params):
    model = construct_model_fromnames(sample_names)
    # Add rates...
    for i in range(len(model.reactions)):
        if model.reactions[i].input == []:
            model.reactions[i].forward_rates = molecular_params.get('creation_rates')
            model.reactions[i].reverse_rates = 0
        if model.reactions[i].output == []:
            model.reactions[i].reverse_rates = molecular_params.get('degradation_rates')
            model.reactions[i].forward_rates = 0
    return model


def compose_kwargs(internal_configs: dict = None, config_file: dict = None) -> dict:
    """ Extra configs like data paths can be supplied here, eg. for circuits that were dynamically generated. """

    if internal_configs is not None:
        for kwarg, config in internal_configs.items():
            if kwarg != 'experiment':
                config_file[kwarg] = config

    data_manager = DataManager(filepath=make_filename_safely(config_file.get("data_path", None)),
                               identities=config_file.get("identities", {}),
                               data=config_file.get("data", None),
                               sample_names=config_file.get("sample_names", None))
    molecular_params = load_json_as_dict(config_file.get("molecular_params"))
 
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
        "model": construct_bioreaction_model(data_manager.data.sample_names, molecular_params),
        "molecular_params": molecular_params,
        "mutations": cast_all_values_as_list(config_file.get("mutations", {})),
        "name": isolate_filename(data_manager.data.source),
        "signal": load_json_as_dict(config_file.get("signal")),
        "system_type": config_file.get("system_type")
    }
    assert all([e in kwargs for e in ESSENTIAL_KWARGS]), 'Some of the kwargs for composing ' \
        f'a circuit are not listed in essential kwargs {ESSENTIAL_KWARGS}: {dict({e: e in kwargs for e in ESSENTIAL_KWARGS})}'
    return kwargs


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



def construct_signal(kwargs) -> Signal:
    SignalType = get_signal_type(kwargs.get("signal_type"))
    return SignalType(**parse_sig_args(kwargs))


def instantiate_system(kwargs):
    system_cfg_args = parse_cfg_args(kwargs)
    SystemType = get_system_type(kwargs["system_type"])
    return SystemType(system_cfg_args)
