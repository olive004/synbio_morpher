from datetime import datetime
from copy import deepcopy
from difflib import SequenceMatcher
from functools import partial
import logging
import os

import numpy as np


def add_outtype(filepath, out_type):
    if out_type in filepath:
        return filepath
    return filepath + '.' + out_type


def get_intersecting_string(string_list):
    for i in range(len(string_list)):
        match = SequenceMatcher(None, string_list[0], string_list[i]).find_longest_match(0, len(string_list[0]), 0, len(string_list[i]))
        base_string = string_list[0][match.a:match.b + match.size]
        if all([base_string in s for s in string_list]):
            return base_string
    raise ValueError(f'No intersecting string could be found in common between {string_list}')


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


def prettify_keys_for_label(key: str):
    key = key.replace('_', ' ')
    key = key.replace('number', 'num').replace('num', 'number')
    key = key.capitalize()
    return key


def prettify_logging_info(loggin):
    str_out = loggin
    if type(loggin) == list:
        logging.info('Printing list more legibly:')
        str_out = '\n'.join(f'{elem}' for elem in loggin)
    return str_out


def remove_file_extension(filename: str) -> str:
    return '.'.join(filename.split('.')[:-1])


def remove_special_py_functions(string_list: list) -> list:
    return [s for s in string_list if '__' not in s]


def remove_element_from_list_by_substring(string_list, exclude):
    return [ x for x in string_list if exclude not in x ]


def sort_by_ordinal_number(string_list: list) -> list:
    base_string = get_intersecting_string(string_list)
    int_equivalents = string_to_num_fromlist(string_list, base_string)
    sorted_strings = deepcopy(string_list)
    for i, int_equiv in enumerate(int_equivalents):
        sorted_strings[int_equiv-1] = string_list[i]
    return sorted_strings


def string_to_num(string_in, base_string):
        num_out = string_in.split(base_string)
        
        if np.char.isdigit(num_out[0]) and np.char.isdigit(num_out[1]):
            raise ValueError(f'Not sure which number to select in string {string_in}')
        elif np.char.isdigit(num_out[0]):
            probable_num = int(num_out[0])
        elif np.char.isdigit(num_out[1]):
            probable_num = int(num_out[1])
        else:
            probable_num = None
        return probable_num


def string_to_num_fromlist(string_list, base_string):
    return list(map(partial(string_to_num, base_string=base_string), string_list))
