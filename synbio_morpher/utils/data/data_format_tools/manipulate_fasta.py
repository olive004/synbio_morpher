
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
from synbio_morpher.utils.data.data_format_tools.common import verify_file_type
from Bio import SeqIO
from typing import Union, Dict, List


def load_seq_from_FASTA(filename, as_type="list") -> Union[Dict,List]:
    verify_file_type(filename, 'fasta')
    f = open(filename, 'r')
    fasta_records = SeqIO.parse(f,'fasta')
    if as_type=="dict":
        sequences = {}
        for fasta_record in fasta_records:
            sequences[fasta_record.id] = str(fasta_record.seq)
    elif as_type=="list":
        sequences = []
        for fasta_record in fasta_records:
            sequences.append(str(fasta_record.seq))
    else:
        raise ValueError(f"Desired type {as_type} not supported.")
    f.close()
    return sequences


def write_fasta_file(out_path: str, seq_generator = None, stype: str = None, count: int = None, data=None, byseq=False) -> None:
    if byseq:
        if type(data) == list:
            write_fasta_file_byseq(seqs=data, stype=stype, out_path=out_path)
        elif type(data) == dict:
            write_fasta_file_byseqdict(seqs=data, out_path=out_path)
    else:
        f = open(out_path, 'w+')
        for i in range(count):
            seq_name = '>' + stype + '_' + str(i) + '\n'
            f.write(seq_name)
            f.write(seq_generator() + '\n')
        f.close()


def write_fasta_file_byseq(seqs: list, stype: str, out_path: str) -> None:
    f = open(out_path, 'w+')
    for i, s in enumerate(seqs):
        seq_name = '>' + stype + '_' + str(i) + '\n'
        f.write(seq_name)
        f.write(s + '\n')
    f.close()
    
    
def write_fasta_file_byseqdict(seqs: dict, out_path: str) -> None:
    f = open(out_path, 'w+')
    for n, s in seqs.items():
        seq_name = '>' + n + '\n'
        f.write(seq_name)
        f.write(s + '\n')
    f.close()
