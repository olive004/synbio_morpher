from run_parameter_grid_analysis import main as parameter_grid_analysis
from src.srv.io.manage.script_manager import Ensembler, script_preamble


def main(config=None, data_writer=None):

    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        'scripts', 'parameter_grid_analysis', 'configs', 'heatmap_cfg.json'))


    v = {"interacting_species": {
            "0": ["RNA_0", "RNA_1"],
            "1": ["RNA_0", "RNA_2"]
        },
        "strengths": {
            "0": {"start_value": 0, "end_value": 1, "step_size": None},
            "1": {"start_value": 0, "end_value": 1, "step_size": None}
        },
        "non_varying_species_interactions": {
            "0": ["RNA_0", "RNA_0"],
            "1": ["RNA_1", "RNA_1"],
            "2": ["RNA_2", "RNA_2"],
            "4": ["RNA_1", "RNA_2"]
        },
        "non_varying_strengths": {
            "0": 0,
            "1": 0,
            "2": 0,
            "4": 0
        }
    }

    v = {
        ("RNA_0", "RNA_0"): {
            "strengths": {"start_value": 0, "end_value": 1, "step_size": None}
        },
        ("RNA_0", "RNA_1"): {
            "non_varying_species_interactions": 0
        },
        ("RNA_0", "RNA_2"): {
            "strengths": {"start_value": 0, "end_value": 1, "step_size": None}
        },
        ("RNA_1", "RNA_1"): {
            "non_varying_species_interactions": 0
        },
        ("RNA_1", "RNA_2"): {
            "non_varying_species_interactions": 0
        },
        ("RNA_2", "RNA_1"): {
            "non_varying_species_interactions": 0
        }
    }

    config["slicing"]["interactions"]

    for species_interaction, interaction_strength_cfg in v.items():
        
