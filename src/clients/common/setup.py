from src.utils.system_definition.configurations import parse_cfg_args
from src.utils.system_definition.specific_systems.RNA_system import RNASystem


def compose_kwargs(config_file):

    data_manager = DataManager(config_file.get("data"))

    kwargs = {
        "data_type": config_file.get("type", "RNA")
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

    simulator_cfg = kwargs
    parsed_namespace_args = parse_cfg_args(simulator_cfg)
    circuit = RNASystem(parsed_namespace_args)
