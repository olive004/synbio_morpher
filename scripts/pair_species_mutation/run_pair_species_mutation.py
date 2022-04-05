from functools import partial
import os
from fire import Fire
from scripts.common.circuit import construct_circuit_from_cfg

from src.srv.results.experiments import Experiment
from src.utils.data.fake_data_generation.seq_generator import RNAGenerator
from src.utils.data.manage.writer import DataWriter
from src.utils.evolution.mutation import Evolver


def main():
    # set configs
    config_file = os.path.join(
        "scripts", "pair_species_mutation", "configs", "RNA_pair_species_mutation.json")
    # start_experiment
    data_writer_kwargs = {'purpose': 'pair_species_mutation'}
    data_writer = DataWriter(**data_writer_kwargs)
    protocols = [
        # generate_sequences
        partial(RNAGenerator(data_writer=data_writer).generate_circuit, count=3, slength=25, protocol="template_mix"),
        # generate_mutations
        partial(Evolver(num_mutations=1, data_writer=data_writer).mutate, system=construct_circuit_from_cfg(config_file)),
        # simulate_interactions
        partial(construct_circuit_from_cfg, config_file)
    ]
    experiment = Experiment(config_file, protocols)
    experiment.run_experiment()

    # output_visualisations
    # write_report


if __name__ == "__main__":
    Fire(main)
