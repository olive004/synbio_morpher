

FORMAT_EXTS = {
    ".fasta": "fasta"
}

def verify_file_type(filepath: str, file_type: str):
    NotImplemented
    pass


def determine_data_format(filepath):
    for extension, ftype in FORMAT_EXTS.items:
        if extension in filepath:
            return ftype
    return None
