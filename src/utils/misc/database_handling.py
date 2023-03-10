
from functools import partial
import logging
from src.utils.misc.type_handling import get_unique
import pandas as pd
from typing import List, Union
import operator
from copy import deepcopy
from src.utils.misc.type_handling import flatten_listlike, merge_dicts
from src.utils.misc.numerical import cast_astype
from src.utils.misc.string_handling import get_all_similar


def thresh_func(thresh: bool, range_df, mode,
                outlier_std_threshold_y, selection_conditions,
                sel_col: str, mode_slide_perc=0.05):
    thresh_text = ''
    remove_outliers_y = False
    if thresh:
        remove_outliers_y = True
        if thresh == 'outlier':
            thresh_text = f', outliers >{outlier_std_threshold_y} std removed'
        elif thresh in ['exclude', 'lt', 'gt', 'lt_strict', 'gt_strict']:
            if thresh == 'exclude':
                v = mode
                thresh_text = f', {v} excluded'
                op = operator.ne
            elif thresh == 'lt':
                v = mode
                thresh_text = f', less than {v}'
                op = operator.lt
            elif thresh == 'gt':
                v = mode
                thresh_text = f', greater than {v}'
                op = operator.gt
            elif thresh == 'lt_strict':
                v = mode - range_df*mode_slide_perc
                thresh_text = f', less than {v}'
                op = operator.lt
            elif thresh == 'gt_strict':
                v = mode + range_df*mode_slide_perc
                thresh_text = f', greater than {v}'
                op = operator.gt
            else:
                raise ValueError(f'Threshold type {thresh} not found.')

            if selection_conditions is not None:
                selection_conditions.append(
                    (sel_col, op, mode))
            else:
                selection_conditions = [
                    (sel_col, op, mode)]

    return thresh_text, remove_outliers_y, selection_conditions


def expand_data_by_col(data: pd.DataFrame, columns: Union[str, list], column_for_expanding_coldata: str, idx_for_expanding_coldata: int,
                       find_all_similar_columns=False):
    """ Expand horizontally or vertically based on if a second column name 
    is given for expanding the column data """

    def make_expansion_df(df_lists, col_names: list) -> pd.DataFrame:
        expanded_data = pd.DataFrame(columns=data.columns, dtype=object)
        for c in col_names:
            temp_data = deepcopy(data)
            temp_data.drop(columns, axis=1, inplace=True)
            expanded_df_col = pd.DataFrame.from_dict(
                dict(zip(columns + [column_for_expanding_coldata],
                     [df_lists.loc[column][c] for column in columns] + [c])),
                dtype=object
            )
            temp_data = pd.concat([temp_data, expanded_df_col], axis=1)
            expanded_data = pd.concat(
                [expanded_data, temp_data], axis=0, ignore_index=True)
        return expanded_data

    if type(columns) == str or (type(columns) == list and len(columns) == 1):
        columns = [columns]

    if find_all_similar_columns:
        all_similar_columns = []
        for column in columns:
            similar_cols = get_all_similar(column, data.columns)
            all_similar_columns += similar_cols
        columns = get_unique(all_similar_columns)

    if column_for_expanding_coldata is None:
        # I haven't added the handling here for expanding multiple columns with lists in them simulatenously
        column = columns[0]
        expand_vertically = True
        s = data.apply(lambda x: pd.Series(
            x[column]), axis=1).stack().reset_index(level=1, drop=True)
        s.name = column
        expanded_data = data.drop(column, axis=1).join(s)
    elif any([type(data[c].iloc[0]) == list for c in columns]):
        # Retain the expanding column
        new_expanding_column_name = 'all_' + column_for_expanding_coldata
        if new_expanding_column_name not in list(data.columns):
            data = data.rename(
                columns={column_for_expanding_coldata: new_expanding_column_name})
        # for c in columns:
        #     for r in data[c]:
        #         print(pd.Series(r))
        df_lists = data[columns].unstack().apply(
            partial(flatten_listlike, safe=True)).apply(pd.Series)
        # df_lists = data[columns].unstack().apply(pd.Series)
        col_names = data[new_expanding_column_name].iloc[idx_for_expanding_coldata]
        if type(col_names) != list:
            logging.warning(
                f'The column {column_for_expanding_coldata} chosen for unstacking other columns has one value')
        if len(df_lists.columns) == 1:
            logging.warning(
                f'The current column {columns} may not be expandable with {col_names}')
        df_lists = df_lists.rename(columns=dict(
            zip(df_lists.columns.values, col_names)))

        # Make dataframe with expanded lists in one column labelled in expansion column
        expanded_data = make_expansion_df(df_lists, col_names)
    else:
        expanded_data = data

    expanded_data.reset_index(drop=True, inplace=True)
    return expanded_data


def expand_df_cols_lists(df: pd.DataFrame, col_of_lists: str, col_list_len: str, include_cols: List[str]) -> pd.DataFrame:
    """ Basically the same function as above but more efficient... """

    if type(df[col_of_lists].iloc[0]) == str:
        # df[col_of_lists] = df[[col_of_lists]].apply(str)
        df = df[df[col_of_lists] != '[]']
        df[col_of_lists] = df[[col_of_lists]].apply(
            partial(cast_astype, dtypes=int))

    # Expand the lists within columnns
    df2 = pd.DataFrame()
    for m in df[col_list_len].unique():
        df_mut = df[df[col_list_len] == m]
        df_lists = pd.DataFrame(
            df_mut[col_of_lists].tolist(), index=df_mut.index)
        df_mut = pd.concat([df_mut, df_lists], axis=1)
        df_mut = df_mut.melt(id_vars=[col_list_len] + include_cols, value_vars=df_lists.columns).drop(
            columns='variable').rename(columns={'value': col_of_lists})
        df2 = pd.concat([df2, df_mut])
    return df2


def select_rows_by_conditional_cols(data: pd.DataFrame, conditions: List[tuple]):
    """ Selection condition should be a tuple of (the column to be used for the selection,
    a condition such as == or !=, and the value for the condition """
    if conditions is None:
        return data
    for column, condition_op, value in conditions:
        selection = condition_op(data[column], value)
        data = data[selection]
    data.reset_index(drop=True, inplace=True)
    return data
