from bioreaction.model.data_tools import construct_model_fromnames
from src.utils.circuit.specific_circuits.RNA.RNA_circuit import RNACircuit
from src.utils.data.common import Data


def get_system_type(sys_type):
    if sys_type == "RNA":
        return RNACircuit
    else:
        raise NotImplementedError(
            f"Desired system type {sys_type} not supported.")


def add_data_to_species(model, data: Data):
    for s in model.species:
        if s.name in data.data.keys():
            s.physical_data = data.data[s.name]
    return model


def construct_bioreaction_model(data: Data, molecular_params):
    model = construct_model_fromnames(data.sample_names)
    model = add_data_to_species(model, data)
    for i in range(len(model.reactions)):
        if model.reactions[i].input == []:
            model.reactions[i].forward_rate = molecular_params.get(
                'creation_rates')
            model.reactions[i].reverse_rate = 0
        elif model.reactions[i].output == []:
            model.reactions[i].reverse_rate = molecular_params.get(
                'degradation_rates')
            model.reactions[i].forward_rate = 0
        else:
            model.reactions[i].forward_rate = None
            model.reactions[i].reverse_rate = None
    return model
