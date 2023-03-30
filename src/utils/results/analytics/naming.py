from typing import List
import logging
import pandas as pd


DIFF_KEY = '_diff_to_base_circuit'
RATIO_KEY = '_ratio_from_mutation_to_base'


def get_analytics_types_base() -> List[str]:
    return ['fold_change',
            'final_deriv',
            'initial_steady_states',
            'max_amount',
            'min_amount',
            'overshoot',
            'RMSE',
            'steady_states']


def get_signal_dependent_analytics() -> List[str]:
    return ['response_time',
            'precision',
            # 'precision_estimate',
            'sensitivity']  # ,
    # 'sensitivity_estimate']


def get_analytics_types_all() -> List[str]:
    return get_analytics_types() + get_analytics_types_diffs() + get_analytics_types_ratios()


def get_analytics_types() -> List[str]:
    """ The naming here has to be unique and not include small 
    raw values like diff, ratio, max, min. """
    return get_analytics_types_base() + get_signal_dependent_analytics()


def get_diffs(analytics_func=get_analytics_types) -> List[str]:
    return [a + DIFF_KEY for a in analytics_func()]


def get_ratios(analytics_func=get_analytics_types) -> List[str]:
    return [a + RATIO_KEY for a in analytics_func()]


def get_signal_dependent_analytics_all() -> List[str]:
    return get_signal_dependent_analytics() + \
        get_diffs(get_signal_dependent_analytics) + \
        get_ratios(get_signal_dependent_analytics)


def get_base_analytics_all() -> List[str]:
    return get_analytics_types_base() + \
        get_diffs(get_analytics_types_base) + \
        get_ratios(get_analytics_types_base)


def get_analytics_types_diffs() -> List[str]:
    return get_diffs(get_analytics_types)


def get_analytics_types_ratios() -> List[str]:
    return get_ratios(get_analytics_types)


def get_true_names_analytics(candidate_cols: List[str]) -> List[str]:
    true_names = []
    analytics_sig = get_signal_dependent_analytics()
    analytics_base = get_base_analytics_all()

    for c in candidate_cols:
        for s in analytics_sig:
            if s in c:
                if (c.replace(s, '')).startswith('_wrt'):
                    # true_name = c.replace(DIFF_KEY, '').replace(RATIO_KEY, '')
                    true_names.append(c)
        for b in analytics_base:
            if b == c:
                true_names.append(c)
    return true_names


def get_true_interaction_cols(data: pd.DataFrame, interaction_attr, remove_symmetrical=False, num_species=3) -> list:
    if 'sample_name' in data:
        num_species = len(data['sample_name'].unique())
    else:
        logging.warning(f'Assuming that the number of species is {num_species}')
    names = []
    for i in range(num_species):
        for ii in range(num_species):
            idxs = [i, ii]
            if remove_symmetrical:
                idxs = sorted(idxs)
            num_ending = '_' + str(idxs[0]) + '-' + str(idxs[1])
            names.append(interaction_attr + num_ending)
    assert all([n in data.columns for n in names]
               ), f'Interaction info column names were not isolated correctly: {names}'
    return sorted(set([n for n in names if n in data.columns]))
