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


def write_fasta_file(seq_generator, stype: str, count: int, out_path: str, data=None, byseq=False) -> None:
    if byseq:
        write_fasta_file_byseq()
    else:
        f = open(out_path, 'w+')
        for i in range(count):
            seq_name = '>' + stype + '_' + str(i) + '\n'
            f.write(seq_name)
            f.write(seq_generator() + '\n')
        f.close()

def write_fasta_file_byseq(seqs: list, names: list, out_path: str, data=None) -> None:
    f = open(out_path, 'w+')
    for n, s in zip(names, seqs):
        f.write('>' + n + '\n')
        f.write(s + '\n')
    f.close()
