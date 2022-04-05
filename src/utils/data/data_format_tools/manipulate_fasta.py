from src.utils.data.data_format_tools.common import verify_file_type
from Bio import SeqIO
from typing import Union, Dict, List


def load_seq_from_FASTA(filename, as_type="list") -> Union[Dict,List]:
    verify_file_type(filename, 'fasta')
    fasta_records = SeqIO.parse(open(filename),'fasta')
    if as_type=="dict":
        sequences = {}
        for fasta_record in fasta_records:
            sequences[fasta_record.id] = str(fasta_record.seq)
        return sequences
    elif as_type=="list":
        sequences = []
        for fasta_record in fasta_records:
            sequences.append(str(fasta_record.seq))
        return sequences
    else:
        raise ValueError(f"Desired type {as_type} not supported.")


def write_fasta_file(seq_generator, fname, stype, count):
    f = open(fname, 'w+')
    for i in range(count):
        seq_name = '>' + stype + '_' + str(i) + '\n'
        f.write(seq_name)
        f.write(seq_generator() + '\n')
    f.close()
    # with open(fname, 'w+') as f:

    #     for i in range(count):
    #         seq_name = '>' + stype + '_' + str(i) + '\n'
    #         f.write(seq_name)
    #         f.write(seq_generator() + '\n')
