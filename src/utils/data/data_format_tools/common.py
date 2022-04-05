import json
import logging
import pandas as pd


FORMAT_EXTS = {
    ".fasta": "fasta"
}

def verify_file_type(filepath: str, file_type: str):
    NotImplemented
    pass


def determine_data_format(filepath):
    for extension, ftype in FORMAT_EXTS.items():
        if extension in filepath:
            return ftype
    return None


def merge_dicts(*dict_objs):
    all_dicts = {}
    for dict_obj in dict_objs:
        if type(dict_obj) == dict:
            all_dicts = {**all_dicts, **dict_obj}
    return all_dicts


def load_json_as_dict(json_pathname):
    try:
        jdict = json.load(open(json_pathname))
    except FileNotFoundError:
        logging.error(f'JSON path {json_pathname} not found')
        SystemExit
    return jdict


def write_csv(data, path_name):
    if type(data) == pd.DataFrame:
        data.to_csv(path_name)
    else:
        raise TypeError(f'Unsupported: cannot output data of type {type(data)} to csv.')
    
