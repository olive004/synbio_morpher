# %% [markdown]
# # Modelling a simple RNA circuit
# 
# In this notebook, we will simulate the interactions between RNAs in a circuit and how the dynamics of the circuit can be simulated.

# %%
from src.utils.common.setup_new import construct_circuit_from_cfg
from src.utils.circuit.agnostic_circuits.circuit_manager_new import CircuitModeller

# %%
config = {}
config['data_path'] = 'data/example/toy_mRNA_circuit.fasta'
config = {
    "data_path": "data/example/toy_mRNA_circuit.fasta",
    "experiment": {
        "purpose": "example"
    },
    "molecular_params": "src/utils/common/configs/RNA_circuit/molecular_params.json",
    "system_type": "RNA"
}

circuit = construct_circuit_from_cfg(config_file=config, extra_configs=None)

# %%
circuit.__dict__

# %%



