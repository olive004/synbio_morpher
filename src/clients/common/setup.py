from src.utils.data.loaders.data_manager import DataManager
from src.utils.system_definition.configurations import parse_cfg_args
from src.utils.system_definition.setup import get_system_type
from src.utils.system_definition.specific_systems.RNA_system import RNASystem


def compose_kwargs(config_file):
    data_manager = DataManager(config_file.get("data"))
    kwargs = {
        "system_type": config_file.get("system_type"),
        "data": data_manager,
    }
    # mcgb kwargs
    # return {
    #     "expmnt": expmnt,
    #     "data": data_loader,
    #     "model": model,
    #     "report_writer": report_writer,
    #     "progress": progress,
    #     "hall": result_hall,
    #     "desk": result_desk
    # }
    return kwargs


def instantiate_system(kwargs):

    system_cfg_args = parse_cfg_args(kwargs)
    system_type = get_system_type(kwargs.get("system_type"))
    circuit = RNASystem(system_cfg_args)
