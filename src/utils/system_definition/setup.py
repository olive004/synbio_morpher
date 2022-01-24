from src.utils.system_definition.specific_systems.RNA_system import RNASystem


def get_system_type(sys_type):
    if sys_type == "RNA":
        return RNASystem
    else:
        raise NotImplementedError(
            f"Desired system type {sys_type} not supported.")
