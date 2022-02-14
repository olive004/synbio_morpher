from datetime import datetime
from copy import deepcopy


def remove_special_py_functions(string_list: list) -> list:
    return [s for s in string_list if '__' not in s]


def make_time_str():
    """Output as 'YEAR_MONTH_DAY_TIME'."""
    now = datetime.now() 
    return now.strftime("%Y_%m_%d_%H%M%S")

def ordered_merge(list1, list2, mask):
    # its = [iter(l) for l in lists]
    # for m in mask:
    #     yield next(its[i])
    merged = deepcopy(list1)
    for i, (n,c,m) in enumerate(zip(list1, list2, mask)):
        merged[i] = c if m else n
    return merged
