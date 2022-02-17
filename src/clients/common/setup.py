from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.data.loaders.data_manager import DataManager
from src.utils.system_definition.configurations import parse_cfg_args
from src.utils.system_definition.setup import get_system_type


def compose_kwargs(config_filename: str) -> dict:
    config_file = load_json_as_dict(config_filename)
    data_manager = DataManager(config_file.get("data"))
    kwargs = {
        "system_type": config_file.get("system_type"),
        "data": data_manager,
    }
    return kwargs


def instantiate_system(kwargs):

    system_cfg_args = parse_cfg_args(kwargs)
    SystemType = get_system_type(kwargs.get("system_type"))
    return SystemType(system_cfg_args)
