from datetime import datetime


def remove_special_py_functions(string_list: list) -> list:
    return [s for s in string_list if '__' not in s]


def make_time_str():
    """Output as 'YEAR_MONTH_DAY_TIME'."""
    now = datetime.now() 
    return now.strftime("%Y_%m_%d_%H%M%S")
