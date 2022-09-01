from src.utils.circuit.specific_circuits.RNA.RNA_circuit import RNACircuit


def get_system_type(sys_type):
    if sys_type == "RNA":
        return RNACircuit
    else:
        raise NotImplementedError(
            f"Desired system type {sys_type} not supported.")
