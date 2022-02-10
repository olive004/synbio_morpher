import random
import numpy as np
from functools import partial
from Bio.Seq import Seq

from src.utils.misc.helper import next_wrapper


def generate_str_from_dict(str_dict, length):
    return ''.join(random.choice(list(str_dict.keys())) for n in range(length))


def generate_mutated_template(template: str, mutate_prop, mutation_pool):
    mutate_idxs = np.random.randint(len(template), size=int(len(template)*mutate_prop))
    template = list(template)
    for idx in mutate_idxs:
        template[idx] = np.random.choice(list(mutation_pool.keys()), p=list(mutation_pool.values()))
    template = ''.join(template)
    return template


def generate_split_template(template:str, count):
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
    """ Protocol can be 'random', 'template_mix', or 'template_split'. 

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
    if protocol == "template_mix":
        template = generate_str_from_dict(nucleotide_pool, slength)
        seq_generator = partial(generate_mutated_template,
                                template=template, 
                                mutate_prop=proportion_to_mutate,
                                mutation_pool=nucleotide_pool)
    elif protocol == "template_split":
        template = generate_str_from_dict(nucleotide_pool, slength)
        sequences = generate_split_template(template, count)
        seq_generator = partial(next_wrapper, generator=iter(sequences))
    elif protocol == "random":
        seq_generator = partial(generate_str_from_dict,
                                str_dict=nucleotide_pool, slength=slength)
    else:
        seq_generator = partial(generate_str_from_dict,
                                str_dict=nucleotide_pool, slength=slength)
        raise NotImplementedError
    write_seq_file(seq_generator, fname, stype, count)
