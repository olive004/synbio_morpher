import logging
from fire import Fire
# from tests.test_circuits import main
# from tests.test_analytics import main
# from src.scripts.agnostic_simulation.run_agnostic_simulation import main
# from src.scripts.RNA_circuit_simulation.run_RNA_circuit import main
# from src.utils.data.fake_data_generation.nc_sequences import main
# from src.scripts.pair_species_mutation.run_pair_species_mutation import main
# from src.scripts.generate_species_templates.run_generate_species_templates import main
# from src.scripts.generate_seqs_flexible.run_generate_seqs_flexible import main
# from src.scripts.ensemble_generate_circuits.run_ensemble_generate_circuits import main
# from src.scripts.gather_interaction_stats.run_gather_interaction_stats import main
# from src.scripts.mutation_effect_on_interactions_signal.run_mutation_effect_on_interactions_signal import main
# from src.scripts.summarise_simulation.run_summarise_simulation import main
# from src.scripts.analyse_mutated_templates.run_analyse_mutated_templates import main
# from src.scripts.analyse_mutated_templates_loaded_0.run_analyse_mutated_templates_loaded import main
# from src.scripts.analyse_mutated_templates_loaded_0_nosig.run_analyse_mutated_templates_loaded_0_nosig import main
# from src.scripts.analyse_mutated_templates_loaded_1.run_analyse_mutated_templates_loaded_1 import main
# from src.scripts.analyse_mutated_templates_loaded_2.run_analyse_mutated_templates_loaded_2 import main
# from src.scripts.analyse_mutated_templates_loaded_2_nosig.run_analyse_mutated_templates_loaded_2_nosig import main
# from src.scripts.analyse_mutated_templates_loaded_3.run_analyse_mutated_templates_loaded_3 import main
# from src.scripts.analyse_mutated_templates_loaded_4.run_analyse_mutated_templates_loaded_4 import main
# from src.scripts.analyse_mutated_templates_loaded_5.run_analyse_mutated_templates_loaded_5 import main
# from src.scripts.analyse_mutated_templates_loaded_means.run_analyse_mutated_templates_loaded_means import main
from src.scripts.ensemble_mutation_effect_analysis.run_ensemble_mutation_effect_analysis import main
# from src.scripts.ensemble_visualisation.run_ensemble_visualisation import main
# from src.scripts.parameter_based_simulation.run_parameter_based_simulation import main
# from src.scripts.stitch_parameter_grid.run_stitch_parameter_grid import main
# from src.scripts.parameter_grid_analysis.run_parameter_grid_analysis import main
# from src.scripts.parameter_grid_analysis.run_multi_parameter_grid_analysis import main
# from src.scripts.ensemble_parameter_grid_analysis.run_ensemble_parameter_grid_analysis import main


FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
FORMAT = "%(filename)s:%(funcName)s():%(lineno)i: %(message)s %(levelname)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    Fire(main)
