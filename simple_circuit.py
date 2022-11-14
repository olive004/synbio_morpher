# %% [markdown]
# # Modelling a simple RNA circuit
# 
# In this notebook, we will simulate the interactions between RNAs in a circuit and how the dynamics of the circuit can be simulated.

# %%
from functools import partial
from src.utils.common.setup_new import construct_circuit_from_cfg
from src.utils.circuit.agnostic_circuits.circuit_manager_new import CircuitModeller
from src.srv.io.manage.script_manager import script_preamble

# %%
config = {}
config['data_path'] = 'data/example/toy_mRNA_circuit.fasta'
config = {
    "data_path": "data/example/toy_mRNA_circuit.fasta",
    "experiment": {
        "purpose": "example"
    },
    "interaction_simulator": {
        "name": "IntaRNA",
        "postprocess": True
    },
    "molecular_params": "src/utils/common/configs/RNA_circuit/molecular_params.json",
    "simulation": {
        "dt": 0.1,
        "t0": 0,
        "t1": 100
    },
    "signal": {
        "inputs": ["RNA_0"],
        "outputs": ["RNA_1"],
        "function_name": "step_function",
        "function_kwargs": {
            "impulse_center": 40, 
            "impulse_halfwidth": 5, 
            "target": 0.5
        } 
    },
    "system_type": "RNA"
}

config, data_writer = script_preamble(config=config, data_writer=None, alt_cfg_filepath=None)

# %%

circuit = construct_circuit_from_cfg(config_file=config, extra_configs=None)
circuit.__dict__

modeller = CircuitModeller(result_writer=data_writer, config=config)
circuit = modeller.init_circuit(circuit=circuit)

# %%



# %%
