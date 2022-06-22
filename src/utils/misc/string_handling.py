from datetime import datetime
from copy import deepcopy
import os


def add_outtype(filepath, out_type):
    if out_type in filepath:
        return filepath
    return filepath + '.' + out_type


def get_intersecting_string(string_list):
    return string_list[0].intersection(*string_list[1:])


def list_to_str(input_listlike):
    return ''.join(input_listlike)


def make_time_str():
    """Output as 'YEAR_MONTH_DAY_TIME'."""
    now = datetime.now() 
    return now.strftime("%Y_%m_%d_%H%M%S")


def ordered_merge(list1, list2, mask) -> list:
    # its = [iter(l) for l in lists]
    # for m in mask:
    #     yield next(its[i])
    list1 = list1 if type(list1) == list else list(list1)
    merged = deepcopy(list1)
    for i, (n,c,m) in enumerate(zip(list1, list2, mask)):
        merged[i] = c if m else n
    return merged


def remove_file_extension(filename: str) -> str:
    return '.'.join(filename.split('.')[:-1])


def remove_special_py_functions(string_list: list) -> list:
    return [s for s in string_list if '__' not in s]


def sort_by_oridnal_number(string_list: list) -> list:

