import logging
from fire import Fire
# from scripts.agnostic_simulation.run_agnostic_simulation import main
# from scripts.RNA_circuit_simulation.run_RNA_circuit import main
# from src.utils.data.fake_data_generation.nc_sequences import main
# from scripts.pair_species_mutation.run_pair_species_mutation import main
# from scripts.generate_species_templates.run_generate_species_templates import main
# from scripts.gather_interaction_stats.run_gather_interaction_stats import main
# from scripts.mutation_effect_on_interactions_signal.run_mutation_effect_on_interactions_signal import main
# from scripts.analyse_mutated_templates.run_analyse_mutated_templates import main
# from scripts.intarna_sandbox.run_intarna_sandbox import main
# from scripts.ensemble_mutation_effect_analysis.run_ensemble_mutation_effect_analysis import main
# from scripts.parameter_based_simulation.run_parameter_based_simulation import main
# from scripts.stitch_parameter_grid.run_stitch_parameter_grid import main
# from scripts.parameter_grid_analysis.run_parameter_grid_analysis import main
from scripts.parameter_grid_analysis.run_ensemble_parameter_grid_analysis import main



FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
FORMAT = "%(filename)s:%(funcName)s():%(lineno)i: %(message)s %(levelname)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    Fire(main)



    # \item run_agnostic_simulation
    # \item run_RNA_circuit
    # \item run_pair_species_mutation
    # \item run_generate_species_templates
    # \item run_gather_interaction_stats
    # \item run_mutation_effect_on_interactions_signal
    # \item run_analyse_mutated_templates
    # \item run_intarna_sandbox
    # \item run_ensemble_mutation_effect_analysis
    # \item run_parameter_based_simulation

    # \item agnostic_simulation
    # \item RNA_circuit_simulation
    # \item pair_species_mutation
    # \item generate_species_templates
    # \item gather_interaction_stats
    # \item mutation_effect_on_interactions_signal
    # \item analyse_mutated_templates
    # \item intarna_sandbox
    # \item ensemble_mutation_effect_analysis
    # \item parameter_based_simulation
