from itertools import permutations
import random
import numpy as np
from functools import partial
from typing import List
from Bio.Seq import Seq

from src.utils.misc.helper import next_wrapper
from src.utils.misc.string_handling import ordered_merge
from src.utils.misc.numerical import nPr, generate_mixed_binary


def generate_str_from_dict(str_dict, length):
    return ''.join(random.choice(list(str_dict.keys())) for n in range(length))


def generate_mutated_template(template: str, mutate_prop, mutation_pool):
    mutate_idxs = np.random.randint(
        len(template), size=int(len(template)*mutate_prop))
    template = list(template)
    for idx in mutate_idxs:
        template[idx] = np.random.choice(
            list(mutation_pool.keys()), p=list(mutation_pool.values()))
    template = ''.join(template)
    return template


def rank_by_mixedness(inputs: List[str]):
    return inputs


def get_num_complementary_slots(length, num_combinations, prioritise_high_slot_num=True):
    max_slots = int(length / 2)
    slot_iteration = range(max_slots, 0, -1) if prioritise_high_slot_num \
        else range(max_slots)
    for slots in slot_iteration:
        possible_combos = nPr(length, slots)
        if possible_combos >= num_combinations:
            return slots
    raise ValueError(f'Template of size {length} only has a maximum of {max_slots} '
                     'complementary slots for permuting.')


def convert_symbolic_complement(real_seq, symbolic_complement):
    comp_seq = str(Seq(real_seq).complement())
    return ordered_merge(real_seq, comp_seq, symbolic_complement)


def calculate_pattern_length(length, count):
    # TODO: Smallest pattern length is desirable
    # High orthogonality between sequences as well however
    # The pattern length depends on count
    pattern_length = 
    

def binary_arpeggiator(sequence, count):
    length = len(sequence)
    interval = int(sequence / count)
    arpeggiation = np.arange(0, length, interval)

    all_seqs = []

    



def generate_mixed_template(template: str, count):
    complementarity_level = get_num_complementary_slots(
        len(template), count)

    pattern_length = calculate_pattern_length(len(template), count)

    symbolic_complement = np.concatenate(np.ones(complementarity_level),
                                         np.zeros(len(template) - complementarity_level))
    seq_permutations = list(permutations(symbolic_complement))
    seq_permutations = rank_by_mixedness(seq_permutations)

    for i, seq in enumerate(seq_permutations):
        seq_permutations[i] = convert_symbolic_complement(seq)


def generate_split_template(template: str, count):
    assert count < len(template), \
        f"Desired sequence length {len(template)} too short to accomodate {count} samples."
    template_complement = str(Seq(template).complement())
    fold = max(int(len(template) / count), 1)

    seqs = []  # could preallocate
    for i in range(0, len(template), fold):
        complement = template_complement[i:i+fold]
        new_seq = template[:i] + complement + template[i+fold:]
        seqs.append(new_seq)
    return seqs


def write_seq_file(seq_generator, fname, stype, count):
    with open(fname, 'w') as f:

        for i in range(count):
            seq_name = '>' + stype + '_' + str(i) + '\n'
            f.write(seq_name)
            f.write(seq_generator() + '\n')


def create_toy_circuit(stype='RNA', count=5, slength=20, protocol="random",
                       proportion_to_mutate=0):
    """ Protocol can be 
    'random': Random sequence generated with weighted characters
    'template_mix': A template sequence is interleaved with complementary characters
    'template_mutate': A template sequence is mutated according to weighted characters
    'template_split': Parts of a template sequence are made complementary

    """
    fname = './src/utils/data/example_data/toy_mRNA_circuit.fasta'
    if stype == 'RNA':
        nucleotide_pool = {
            'A': 0.2,
            'C': 0.3,
            'G': 0.3,
            'U': 0.2
        }
    elif stype == 'DNA':
        nucleotide_pool = {
            'A': 0.2,
            'C': 0.3,
            'G': 0.3,
            'T': 0.2
        }
    else:
        raise NotImplementedError
    template = generate_str_from_dict(nucleotide_pool, slength)
    if protocol == "template_mutate":
        seq_generator = partial(generate_mutated_template,
                                template=template,
                                mutate_prop=proportion_to_mutate,
                                mutation_pool=nucleotide_pool)
    elif protocol == "template_split":
        sequences = generate_split_template(template, count)
        seq_generator = partial(next_wrapper, generator=iter(sequences))
    elif protocol == "template_mix":
        sequences = generate_mixed_template(template, count)
        seq_generator = partial(next_wrapper, generator=iter(sequences))
    elif protocol == "random":
        seq_generator = partial(generate_str_from_dict,
                                str_dict=nucleotide_pool, slength=slength)
    else:
        seq_generator = partial(generate_str_from_dict,
                                str_dict=nucleotide_pool, slength=slength)
        raise NotImplementedError
    write_seq_file(seq_generator, fname, stype, count)



def main():
    template = 'ACTGTGTGCCCCTAAACCGCGCT'
    count = 5
    complementarity_level = get_num_complementary_slots(
        len(template), count)
    print(complementarity_level)
    print(len(template) - complementarity_level)
    np.ones(complementarity_level)
    np.zeros(int(len(template) - complementarity_level))
    symbolic_complement = generate_mixed_binary(len(template), complementarity_level, count)
    seq_permutations = list(permutations(symbolic_complement, count))
    print(seq_permutations)