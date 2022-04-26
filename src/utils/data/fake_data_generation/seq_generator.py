from functools import partial
import logging
import numpy as np
import random
from Bio.Seq import Seq

from src.srv.io.results.writer import DataWriter
from src.utils.misc.helper import next_wrapper
from src.utils.misc.numerical import generate_mixed_binary
from src.utils.misc.string_handling import ordered_merge, list_to_str


class SeqGenerator():

    SEQ_POOL = {}

    def __init__(self, data_writer: DataWriter) -> None:
        self.stype = None
        self.data_writer = data_writer

    @staticmethod
    def generate_mutated_template(template: str, mutate_prop, mutation_pool):
        mutate_idxs = np.random.randint(
            len(template), size=int(len(template)*mutate_prop))
        template = list(template)
        for idx in mutate_idxs:
            template[idx] = np.random.choice(
                list(mutation_pool.keys()), p=list(mutation_pool.values()))
        template = ''.join(template)
        return template

    @staticmethod
    def generate_str_from_probdict(str_prob_dict: dict, slength) -> str:
        population = list(str_prob_dict.keys())
        probabilities = list(str_prob_dict.values())
        return ''.join(random.choices(population, probabilities, k=slength))


class NucleotideGenerator(SeqGenerator):

    def __init__(self, data_writer) -> None:
        super().__init__(data_writer)

    @staticmethod
    def convert_symbolic_complement(real_seq, symbolic_complement):
        comp_seq = str(Seq(real_seq).complement())
        return list_to_str(ordered_merge(real_seq, comp_seq, symbolic_complement))

    @staticmethod
    def generate_split_template(template: str, count) -> list:
        assert count < len(template), \
            f"Desired sequence length {len(template)} too short to accomodate {count} samples."
        template_complement = str(Seq(template).complement())
        fold = max(int(len(template) / count), 1)

        seqs = []  # could preallocate
        for i in range(0, len(template), fold):
            complement = template_complement[i:(i+fold)]
            new_seq = template[:i] + complement + template[(i+fold):]
            seqs.append(new_seq)
        return seqs[:count]

    def generate_mixed_template(self, template: str, count) -> list:
        # TODO: Generate proper permutations or arpeggiations,
        # rank by mixedness and similarity. Can use list(permutations())
        # and ranking function in future.

        symbolic_complements = generate_mixed_binary(
            len(template), count, zeros_to_ones=False)
        seq_permutations = [None] * count
        for i, symb in enumerate(symbolic_complements):
            seq_permutations[i] = self.convert_symbolic_complement(
                template, symb)
        return seq_permutations

    def generate_circuits(self, iter_count=1, name='toy_mRNA_circuit', **circuit_kwargs):
        circuit_paths = []
        for i in range(iter_count):
            circuit_kwargs['fname'] = name + '_' + str(i)
            circuit_path = self.generate_circuit(**circuit_kwargs)
            circuit_paths.append(circuit_path)
        return circuit_paths

    def generate_circuit(self, count=5, slength=20, protocol="random",
                         name='toy_mRNA_circuit',
                         out_type='fasta',
                         proportion_to_mutate=0, template=None):
        """ Protocol can be 
        'random': Random sequence generated with weighted characters
        'template_mix': A template sequence is interleaved with complementary characters
        'template_mutate': A template sequence is mutated according to weighted characters
        'template_split': Parts of a template sequence are made complementary
        """

        if template is None:
            template = self.generate_str_from_probdict(self.SEQ_POOL, slength)
        # template = 'CGCGCGCGCGCGCGCGCGCGCGCCGCGCG'  # Very strong interactions
        # template = 'CUUCAAUUCCUGAAGAGGCGGUUGG'  # Very weak interactions
        if protocol == "template_mutate":
            seq_generator = partial(self.generate_mutated_template,
                                    template=template,
                                    mutate_prop=proportion_to_mutate,
                                    mutation_pool=self.SEQ_POOL)
        elif protocol == "template_split":
            sequences = self.generate_split_template(template, count)
            seq_generator = partial(next_wrapper, generator=iter(sequences))
        elif protocol == "template_mix":
            sequences = self.generate_mixed_template(template, count)
            seq_generator = partial(next_wrapper, generator=iter(sequences))
        elif protocol == "random":
            seq_generator = partial(self.generate_str_from_probdict,
                                    str_prob_dict=self.SEQ_POOL, slength=slength)
        else:
            seq_generator = partial(self.generate_str_from_probdict,
                                    str_prob_dict=self.SEQ_POOL, slength=slength)
            raise NotImplementedError

        out_path = self.data_writer.output(out_name=name, out_type=out_type,
                                           seq_generator=seq_generator, stype=self.stype, 
                                           count=count, return_path=True,
                                           subfolder='circuits')
        return {'data': out_path}


class RNAGenerator(NucleotideGenerator):

    SEQ_POOL = {
        'A': 0.2,
        'C': 0.3,
        'G': 0.3,
        'U': 0.2
    }

    def __init__(self, data_writer) -> None:
        super().__init__(data_writer)
        self.stype = "RNA"


class DNAGenerator(NucleotideGenerator):

    SEQ_POOL = {
        'A': 0.2,
        'C': 0.3,
        'G': 0.3,
        'T': 0.2
    }

    def __init__(self, data_writer) -> None:
        super().__init__(data_writer)
        self.stype = "DNA"


def main():
    # Testing
    template = 'ACTGTGTGCCCCTAAACCGCGCT'
    count = 10
    seq_permutations = DNAGenerator().generate_mixed_template(template, count)
    assert len(seq_permutations) == count, 'Wrong sequence length.'
    print(seq_permutations)
