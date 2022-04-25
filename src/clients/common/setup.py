from src.utils.data.data_format_tools.common import load_json_as_dict, process_json
from src.srv.io.manage.data_manager import DataManager
from src.utils.misc.io import isolate_filename
from src.utils.misc.type_handling import cast_all_values_as_list
from src.utils.signal.configs import get_signal_type, parse_sig_args
from src.utils.system_definition.config import parse_cfg_args
from src.utils.system_definition.setup import get_system_type


def manage_config(config_file):



def compose_kwargs(config_filepath: str, extra_configs) -> dict:
    config_file = process_json(load_json_as_dict(config_filepath))
    for kwarg, config in extra_configs.items():
        if config_file.get(kwarg):
            config_file[kwarg] = config
    config_file.update(extra_configs)
    data_manager = DataManager(filepath=config_file.get("data"),
                               identities=config_file.get("identities", {}))
    kwargs = {
        "system_type": config_file.get("system_type"),
        "name": isolate_filename(data_manager.data.source),
        "data_path": data_manager.data,
        "identities": data_manager.data.identities,
        "mutations": cast_all_values_as_list(config_file.get("mutations", {})),
        "signal": load_json_as_dict(config_file.get("signal")),
        "molecular_params": load_json_as_dict(config_file.get("molecular_params"))
    }
    return kwargs


def construct_signal(kwargs):
    signal_kwargs = parse_sig_args(kwargs)
    SignalType = get_signal_type(kwargs.get("signal_type"))
    return SignalType(**signal_kwargs)


def instantiate_system(kwargs):

    system_cfg_args = parse_cfg_args(kwargs)
    SystemType = get_system_type(kwargs.get("system_type"))
    return SystemType(system_cfg_args)
