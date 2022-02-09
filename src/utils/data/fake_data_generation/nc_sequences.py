import random
from functools import partial


def generate_nuc_seq(nucleotide_pool, length):
    return ''.join(random.choice(list(nucleotide_pool.keys())) for n in range(length))


def write_seq_file(seq_generator, fname, stype, count, slength):
    with open(fname, 'w') as f:

        for i in range(count):
            seq_name = '>' + stype + '_' + str(i) + '\n'
            f.write(seq_name)
            f.write(seq_generator(slength) + '\n')


def create_toy_circuit(stype='RNA', count=5, slength=20, protocol="random"):
    """ Protocol can be 'random', 'template_mix', or 'template_split'."""
    fname = './src/utils/data/example_data/toy_mRNA_circuit.fasta'
    if stype == 'RNA':
        nucleotide_pool = {
            'A': 0.2,
            'C': 0.3,
            'G': 0.3,
            'U': 0.2
        }
    if stype == 'DNA':
        nucleotide_pool = {
            'A': 0.2,
            'C': 0.3,
            'G': 0.3,
            'T': 0.2
        }
    else:
        raise NotImplementedError
    if protocol == "template_mix":

    if protocol == "random":

        seq_generator = partial(generate_nuc_seq, nucleotide_pool=nucleotide_pool)
    else:
        seq_generator = partial(generate_nuc_seq, nucleotide_pool=nucleotide_pool)
        raise NotImplementedError
    write_seq_file(seq_generator, fname, stype, count, slength)
