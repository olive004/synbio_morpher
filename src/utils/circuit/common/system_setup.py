from bioreaction.model.data_tools import construct_model_fromnames
from bioreaction.model.data_containers import BasicModel
from src.utils.data.common import Data


def get_system_type(sys_type):
    if sys_type == "RNA":
        from src.utils.circuit.agnostic_circuits.circuit_new import Circuit
        return Circuit
    else:
        raise NotImplementedError(
            f"Desired system type {sys_type} not supported.")


def add_data_to_species(model: BasicModel, data: Data):
    for i, s in enumerate(model.species):
        if s.name in data.data.keys():
            model.species[i].physical_data = data.data[s.name]
    return model


def construct_bioreaction_model(data: Data, molecular_params: dict):
    model = construct_model_fromnames(data.sample_names)
    model = add_data_to_species(model, data)
    for i in range(len(model.reactions)):
        if model.reactions[i].input == []:
            model.reactions[i].forward_rate = molecular_params.get(
                'creation_rate')
            model.reactions[i].reverse_rate = 0
        elif model.reactions[i].output == []:
            model.reactions[i].reverse_rate = molecular_params.get(
                'degradation_rate')
            model.reactions[i].forward_rate = 0
        else:
            model.reactions[i].forward_rate = 0
            model.reactions[i].reverse_rate = 0
    return model
