import random


def generate_RNA(length=20):
    nucleotide_pool = {
        'A': 0.2,
        'C': 0.3,
        'G': 0.3,
        'U': 0.2
    }
    return ''.join(random.choice(list(nucleotide_pool.keys())) for n in range(length))


def write_seq_file(fname, stype='RNA', count=5, slength=20):
    with open(fname, 'w') as f:
        if stype == 'RNA':
            seq_generator = generate_RNA
        else:
            seq_generator = generate_RNA
            raise NotImplementedError
        for i in range(count):
            seq_name = '>' + stype + '_' + str(i) + '\n'
            f.write(seq_name)
            f.write(seq_generator(slength) + '\n')


def create_toy_circuit(stype='RNA', count=5, slength=20):
    fname = './src/utils/data/example_data/toy_mRNA_circuit.fasta'
    write_seq_file(fname, stype, count, slength)
