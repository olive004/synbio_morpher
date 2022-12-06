from copy import deepcopy
from functools import partial
from typing import List, Union
import networkx as nx
import numpy as np
import pandas as pd
import re
import logging
from src.utils.misc.type_handling import get_unique
from src.utils.results.writer import DataWriter
from src.utils.misc.string_handling import add_outtype, get_all_similar, make_time_str, prettify_keys_for_label
from src.utils.misc.type_handling import flatten_listlike
from pyvis.network import Network
from src.utils.misc.type_handling import merge_dicts
import seaborn as sns
from matplotlib import pyplot as plt
plt.ioff()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NetworkCustom(Network):
    """ This customisation has since been merged into Networkx officially, but keeping this here for legacy. """

    def from_nx(self, nx_graph, node_size_transf=(lambda x: x), edge_weight_transf=(lambda x: x),
                default_node_size=10, default_edge_weight=1, show_edge_weights=False,
                use_weighted_opacity=True, invert_opacity=True):
        """
        Overriding default pyvis Network() function to allow show_edge_weights
        """
        assert (isinstance(nx_graph, nx.Graph))
        edges = nx_graph.edges(data=True)
        nodes = nx_graph.nodes(data=True)

        edge_weights = [
            w for (u, v, w) in nx_graph.edges.data('weight', default=0)]
        if np.all(edge_weights == 0) or edge_weights == []:
            edge_w_range = (0, 0)
        else:
            edge_w_range = (np.min(edge_weights), np.max(edge_weights))
        opacity_range = (0, 1)
        if invert_opacity:
            opacity_range = (1, 0)

        if len(edges) > 0:
            for e in edges:
                if 'size' not in nodes[e[0]].keys():
                    nodes[e[0]]['size'] = default_node_size
                nodes[e[0]]['size'] = int(
                    node_size_transf(nodes[e[0]]['size']))
                if 'size' not in nodes[e[1]].keys():
                    nodes[e[1]]['size'] = default_node_size
                nodes[e[1]]['size'] = int(
                    node_size_transf(nodes[e[1]]['size']))
                self.add_node(e[0], **nodes[e[0]])
                self.add_node(e[1], **nodes[e[1]])

                if 'weight' not in e[2].keys():
                    e[2]['weight'] = default_edge_weight
                e[2]['weight'] = edge_weight_transf(e[2]['weight'])
                if show_edge_weights:
                    # These are parameters for pyvis Network() not native to Networkx.
                    # See https://rdrr.io/cran/visNetwork/man/visEdges.html
                    # for edge parameters options.
                    e[2]["label"] = e[2]["weight"]
                if use_weighted_opacity:
                    squashed_w = np.interp(
                        e[2]['weight'], edge_w_range, opacity_range)
                    e[2]["value"] = 1
                    e[2]["color"] = {}
                    e[2]["font"] = {}
                    e[2]["color"]["opacity"] = edge_weight_transf(squashed_w)
                    e[2]["font"]["size"] = 5
                self.add_edge(e[0], e[1], **e[2])

        for node in nx.isolates(nx_graph):
            if 'size' not in nodes[node].keys():
                nodes[node]['size'] = default_node_size
            self.add_node(node, **nodes[node])


class Graph():

    def __init__(self, labels: List[str], source_matrix: np.ndarray = None) -> None:
        source_matrix = np.zeros(
            (len(labels), len(labels))) if source_matrix is None else source_matrix
        self._node_labels = labels
        self.graph = self.build_graph(source_matrix)

    def build_graph(self, source_matrix: np.ndarray) -> nx.DiGraph:
        graph = nx.from_numpy_matrix(source_matrix, create_using=nx.DiGraph)
        if self.node_labels is not None:
            graph = nx.relabel_nodes(graph, self.node_labels)
        return graph

    def refresh_graph(self, source_matrix: np.ndarray):
        self.graph = self.build_graph(source_matrix)

    def get_graph_labels(self) -> dict:
        return sorted(self.graph)

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, new_graph):
        assert type(new_graph) == nx.DiGraph, 'Cannot set graph to' + \
            f' type {type(new_graph)}.'
        self._graph = new_graph

    @property
    def node_labels(self):
        if type(self._node_labels) == list:
            return {i: n for i, n in enumerate(self._node_labels)}
        return self._node_labels

    @node_labels.setter
    def node_labels(self, labels: Union[dict, list]):

        current_nodes = self.get_graph_labels()
        if type(labels) == list:
            self._node_labels = dict(zip(current_nodes, labels))
        else:
            labels = len(labels)
            self._node_labels = dict(zip(current_nodes, labels))
        self.graph = nx.relabel_nodes(self.graph, self.node_labels)


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
            # {column: df_lists[c], column_for_expanding_coldata: c})
            # temp_data = temp_data.assign(
            #     **dict(zip(
            #         columns + [column_for_expanding_coldata],
            #         [pd.Series(expanded_df_col[column].values) for column in columns] + [pd.Series(
            #             expanded_df_col[column_for_expanding_coldata].values)]))
            #     # column: pd.Series(expanded_df_col[column].values),
            #     # column_for_expanding_coldata: pd.Series(
            #     #     expanded_df_col[column_for_expanding_coldata].values)
            # )
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
    else:
        # Retain the expanding column
        new_expanding_column_name = 'all_' + column_for_expanding_coldata
        if new_expanding_column_name not in list(data.columns):
            data = data.rename(
                columns={column_for_expanding_coldata: new_expanding_column_name})
        # for c in columns:
        #     for r in data[c]:
        #         print(pd.Series(r))
        df_lists = data[columns].unstack().apply(partial(flatten_listlike, safe=True)).apply(pd.Series)
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

    expanded_data.reset_index(drop=True, inplace=True)
    return expanded_data


def visualise_data(og_data: pd.DataFrame, data_writer: DataWriter = None,
                   cols_x: list = None, cols_y: list = None,
                   normalise_data_x: bool = False,
                   normalise_data_y: bool = False,
                   plot_type: str = None, out_name='test_plot',
                   hue: str = None,
                   preprocessor_func_x=None,
                   postprocessor_func_x=None,
                   postprocessor_func_y=None,
                   use_sns=False,
                   log_axis=[False, False],
                   threshold_value_max=None,
                   bin_count=100,
                   selection_conditions: List[tuple] = None,
                   remove_outliers_y: bool = False,
                   outlier_std_threshold_y=3,
                   complete_colx_by_str=None,
                   complete_coly_by_str=None,
                   exclude_rows_nonempty_in_cols: list = None,
                   exclude_rows_zero_in_cols: list = None,
                   expand_xcoldata_using_col: bool = False,
                   expand_ycoldata_using_col: bool = False,
                   column_name_for_expanding_xcoldata: str = None,
                   column_name_for_expanding_ycoldata: str = None,
                   idx_for_expanding_xcoldata: int = 0,
                   idx_for_expanding_ycoldata: int = 0,
                   plot_cols_on_same_graph=False,
                   misc_histplot_kwargs=None,
                   **plot_kwargs):
    """ Plot type can be any attributes of VisODE() """
    data = deepcopy(og_data)

    hue = hue if hue is not None else column_name_for_expanding_xcoldata

    def process_cols(cols, complete_col_by_str):
        if cols is None:
            return [None]
        if complete_col_by_str is not None:
            for i, col in enumerate(cols):
                end = col.split(complete_col_by_str.split('_wrt')[0])[-1]
                rex = np.asarray([re.search(f"^{complete_col_by_str}.*{end}", c) for c in data.columns])
                if not list(data.columns[np.where(rex != None)[0]]):
                    logging.warning(f'Could not find complete column name for {col} with str {complete_col_by_str}')
                potential_cols = list(data.columns[np.where(rex != None)[0]])
                if not end:
                    cols[i] = min(potential_cols)
                else:
                    cols[i] = potential_cols
        return flatten_listlike(cols, safe=True)
    cols_x = process_cols(cols_x, complete_colx_by_str)
    cols_y = process_cols(cols_y, complete_coly_by_str)

    def process_log_sns(data: pd.DataFrame, column_x: str, column_y: str,
                        plot_kwrgs: dict, log_axis: list, plot_type: str = None):
        exempt_from_log = ['histplot']

        def process_log(data, col, log_option, col_label):
            if col is not None:
                if log_option:
                    data = data[data[col] != 0]
                    data = data.drop(col, axis=1).join(data[col].abs())
                    new_col = r'$Log_{10}$' + ' of ' + \
                        plot_kwrgs.get(col_label, col)
                    if plot_type not in exempt_from_log:
                        log_df = np.log10(data[col])
                        data = data.drop(col, axis=1).join(log_df)
                else:
                    new_col = plot_kwrgs.get(col_label, col)

                data = data.rename(columns={col: new_col})
            return data, new_col

        if column_x is not None:
            data, column_x = process_log(data, column_x, log_axis[0], 'xlabel')
        if column_y is not None:
            data, column_y = process_log(data, column_y, log_axis[1], 'ylabel')
        if column_x is None and column_y is None:
            logging.warning('No columns given as input to histplot.')
        data.reset_index(drop=True, inplace=True)
        return data, column_x, column_y

    def preprocess_data(data: pd.DataFrame, preprocessor_func_x,
                        threshold_value_max,
                        expand_xcoldata_using_col, expand_ycoldata_using_col,
                        column_x: str, column_y: str,
                        normalise_data_x: bool, normalise_data_y: bool,
                        column_name_for_expanding_xcoldata,
                        column_name_for_expanding_ycoldata,
                        exclude_rows_nonempty_in_cols, exclude_rows_zero_in_cols,
                        idx_for_expanding_xcoldata,
                        idx_for_expanding_ycoldata,
                        preprocessor_func_y=None,
                        postprocessor_func_x=None,
                        postprocessor_func_y=None,
                        selection_conditions=None):
        def preprocess(data, column, expand_coldata_using_col, normalise_data,
                       preprocessor_func, column_name_for_expanding_coldata,
                       idx_for_expanding_coldata, remove_outliers=False,
                       outlier_std_threshold=3):
            if column is not None:
                if expand_coldata_using_col:
                    data = expand_data_by_col(
                        data=data, columns=column,
                        column_for_expanding_coldata=column_name_for_expanding_coldata,
                        idx_for_expanding_coldata=idx_for_expanding_coldata)
                if preprocessor_func:
                    data = data.drop(column, axis=1).join(
                        preprocessor_func(data[column]))
                if threshold_value_max:
                    data = data[data[column] <= threshold_value_max]
                if remove_outliers:
                    mean = data[column].mean()
                    std = data[column].std()
                    if std != 0:
                        data = data[data[column] < mean +
                                    np.multiply(outlier_std_threshold, std)]
                        data = data[data[column] > mean -
                                    np.multiply(outlier_std_threshold, std)]
                if normalise_data:
                    df = data[column]
                    s = (df - df.mean()) / df.std()
                    data = data.drop(column, axis=1).join(s)

            return data, column

        def postprocess(data, column, postprocessor_func):
            if postprocessor_func:
                data = data.drop(column, axis=1).join(
                    postprocessor_func(data[column]))
            return data
        data, column_x = preprocess(
            data, column_x, expand_xcoldata_using_col, normalise_data_x,
            preprocessor_func_x, column_name_for_expanding_xcoldata,
            idx_for_expanding_xcoldata)
        data, column_y = preprocess(
            data, column_y, expand_ycoldata_using_col, normalise_data_y,
            None, column_name_for_expanding_ycoldata,
            idx_for_expanding_ycoldata, remove_outliers_y, outlier_std_threshold_y)
        data.reset_index(drop=True, inplace=True)

        for exc_cols, condition in zip(
            [exclude_rows_nonempty_in_cols, exclude_rows_zero_in_cols],
                ['', 0]):
            data = exclude_rows_by_cols(data, exc_cols, condition)

        data = select_rows_by_conditional_cols(data, selection_conditions)
        data = postprocess(data, column_x, postprocessor_func_x)
        data = postprocess(data, column_y, postprocessor_func_y)
        data.reset_index(drop=True, inplace=True)
        return data, column_x, column_y

    def exclude_rows_by_cols(data: pd.DataFrame, cols: list, condition):
        if cols is not None:
            for exc_col in cols:
                data = data[data[exc_col] != condition]
        data.reset_index(drop=True, inplace=True)
        return data

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

    visualiser = VisODE()
    if plot_type == 'plot':
        for col_x in cols_x:
            for col_y in cols_y:
                x, y = data[col_x], data[col_y]
                if preprocessor_func_x:
                    x = preprocessor_func_x(x.values)
                    # y = preprocessor_func_x(y.values)
                data_writer.output(out_type='png', out_name=out_name,
                                   write_func=visualiser.plot,
                                   **merge_dicts({'data': x, 'y': y}, plot_kwargs))
        return

    for col_x in cols_x:
        for col_y in cols_y:
            if use_sns:
                data, col_x, col_y = preprocess_data(
                    data, preprocessor_func_x, threshold_value_max,
                    expand_xcoldata_using_col, expand_ycoldata_using_col,
                    col_x, col_y,
                    normalise_data_x, normalise_data_y,
                    column_name_for_expanding_xcoldata,
                    column_name_for_expanding_ycoldata,
                    exclude_rows_nonempty_in_cols, exclude_rows_zero_in_cols,
                    idx_for_expanding_xcoldata,
                    idx_for_expanding_ycoldata,
                    selection_conditions=selection_conditions,
                    postprocessor_func_x=postprocessor_func_x,
                    postprocessor_func_y=postprocessor_func_y)
                data, col_x, col_y = process_log_sns(
                    data, col_x, col_y, plot_kwargs, log_axis, plot_type)
                if plot_type == 'scatter_plot':
                    write_func = visualiser.sns_scatterplot
                elif plot_type == 'bar_plot':
                    write_func = visualiser.sns_barplot
                elif plot_type == 'violin_plot':
                    write_func = visualiser.sns_violinplot
                elif plot_type == 'line_plot':
                    write_func = visualiser.sns_lineplot
                elif plot_type == 'histplot':
                    write_func = visualiser.sns_histplot
                    plot_kwargs.update({'log_axis': log_axis,
                                        'bin_count': bin_count,
                                        'histplot_kwargs': misc_histplot_kwargs
                                        })
                else:
                    logging.warning(f'Unknown plot type given "{plot_type}"')

                if not data.empty:
                    data_writer.output(out_type='svg', out_name=out_name,
                                       write_func=write_func,
                                       **merge_dicts({'x': col_x, 'y': col_y, 'data': data,
                                                      'hue': hue},
                                                     plot_kwargs))
                else:
                    logging.warning(
                        f'Could not visualise columns {col_x} and {col_y}')


def visualise_graph_pyvis(graph: nx.DiGraph,
                          out_path: str,
                          new_vis=False,
                          out_type='html'):
    import webbrowser
    import os

    if new_vis:
        out_path = f'{out_path}_{make_time_str()}'
    out_path = add_outtype(out_path, out_type)

    interactive_graph = NetworkCustom(
        height=800, width=800, directed=True, notebook=True)
    interactive_graph.from_nx(graph, edge_weight_transf=lambda x: round(x, 4))
    interactive_graph.inherit_edge_colors(True)
    interactive_graph.set_edge_smooth('dynamic')
    try:
        interactive_graph.save_graph(out_path)
    except PermissionError:
        logging.warning(
            f'The graph at {out_path} may not have saved properly.')

    # web_filename = 'file:///' + os.getcwd() + '/' + out_path
    # # webbrowser.open(web_filename, new=1, autoraise=True)


def visualise_graph_pyplot(graph: nx.DiGraph):
    ax1 = plt.subplot(111)
    nx.draw(graph)
    raise NotImplementedError


class VisODE():
    def __init__(self, figsize=10) -> None:
        self.figsize = figsize

    def add_kwrgs(self, plt, **plot_kwrgs):
        def warning_call(object, function, dummy_value, **dummy_kwargs):
            logging.warn(
                f'Could not call function {function} in object {object}.')

        for plot_kwrg, value in plot_kwrgs.items():
            if value is not None:
                if type(value) == dict:
                    # Treat as kwargs
                    getattr(plt, plot_kwrg, partial(
                        warning_call, object=plt, dummy_value=None, function=plot_kwrg))(**value)
                else:
                    getattr(plt, plot_kwrg, partial(
                        warning_call, dummy_value=None, function=plot_kwrg))(value)

    def rename_legend(self, plot, new_title: str, new_labels: list):
        leg = plot.axes.flat[0].get_legend()
        leg.set_title(new_title)
        for t, l in zip(leg.texts, new_labels):
            t.set_text(l)

    def plot(self, data, y=None, new_vis=False, t=None, out_path='test_plot', out_type='svg',
             **plot_kwrgs) -> None:
        data = data.T if len(plot_kwrgs.get('legend', [])
                             ) == np.shape(data)[0] else data
        plt.figure()
        if y is not None:
            plt.plot(data, y)
        elif t is not None:
            # if np.shape(t)[0] != np.shape(data)[0] and len(np.shape(data)) == 2:
            #     logging.warning()
            plt.plot(t, data)
        else:
            plt.plot(data)
        self.add_kwrgs(plt, **plot_kwrgs)
        plt.savefig(out_path)
        plt.close()

    def sns_generic_plot(self, plot_func, out_path: str, x: str, y: str, data: pd.DataFrame,
                         title: str = None,
                         figsize=(7, 5), style="ticks",
                         **plot_kwargs):
        sns.set_context('paper')
        sns.set_theme(style=style)
        sns.set_palette("viridis")
        if title is None:
            title = plot_kwargs.get('title')
        if plot_kwargs.get('hue'):
            data = data.rename(
                columns={plot_kwargs.get('hue'): prettify_keys_for_label(plot_kwargs.get('hue'))})
            plot_kwargs['hue'] = prettify_keys_for_label(plot_kwargs.get('hue'))
            plot_kwargs.update({
                # 'palette': sns.color_palette("husl", len(data[plot_kwargs.get('hue')].unique()))
                # 'palette': sns.color_palette("plasma", len(data[plot_kwargs.get('hue')].unique()))
                'palette': sns.color_palette("viridis", len(data[plot_kwargs.get('hue')].unique()))
            })

        if x is None:
            logging.warn(
                'Make sure a column is specified for visualising with seaborn')
        f, ax = plt.subplots(figsize=figsize)

        plot_func(
            data=data,
            x=x,
            y=y,
            **plot_kwargs
        ).set(title=title)
        # if plot_kwargs.get('hue'):
        #     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
        #                title=plot_kwargs['hue'])
        #             #    title=prettify_keys_for_label(plot_kwargs.get('hue')))
        if plot_kwargs.get('hue'):
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        f.savefig(out_path, bbox_inches='tight')
        plt.close()

    def sns_barplot(self, out_path: str, x: str, y: str, data: pd.DataFrame,
                    title: str = None, xlabel=None, ylabel=None,
                    **plot_kwargs):
        plot_kwargs.update({'errwidth': 1})

        # if plot_kwargs.get('hue'):
        #     plot_kwargs.update({
        #         # 'palette': sns.color_palette("husl", len(data[plot_kwargs.get('hue')].unique()))
        #         # 'palette': sns.color_palette("plasma", len(data[plot_kwargs.get('hue')].unique()))
        #         'palette': sns.color_palette("viridis", len(data[plot_kwargs.get('hue')].unique()))
        #     })

        self.sns_generic_plot(sns.barplot, out_path, x,
                              y, data, title, **plot_kwargs)

    def sns_scatterplot(self, out_path: str, x: str, y: str, data: pd.DataFrame,
                        title: str = None, xlabel=None, ylabel=None,
                        **plot_kwargs):
        import seaborn as sns
        self.sns_generic_plot(sns.scatterplot, out_path,
                              x, y, data, title, **plot_kwargs)

    def sns_violinplot(self, out_path: str, x: str, y: str, data: pd.DataFrame,
                       title: str = None, xlabel=None, ylabel=None,
                       **plot_kwargs):
        import seaborn as sns
        self.sns_generic_plot(sns.violinplot, out_path, x, y, data, title,
                              figsize=(10, 6), **plot_kwargs)

    def sns_lineplot(self, out_path: str, x: str, y: str, data: pd.DataFrame,
                     title: str = None, xlabel=None, ylabel=None,
                     **plot_kwargs):
        import seaborn as sns

        if plot_kwargs.get('hue'):
            plot_kwargs.update({
                # 'palette': sns.color_palette("husl", len(data[plot_kwargs.get('hue')].unique()))
                # 'palette': sns.color_palette("plasma", len(data[plot_kwargs.get('hue')].unique()))
                'palette': sns.color_palette("viridis", len(data[plot_kwargs.get('hue')].unique()))
            })

        self.sns_generic_plot(sns.lineplot, out_path,
                              x, y, data, title, **plot_kwargs)

    # Make visualisations for each analytic chosen
    def heatmap(self, data: pd.DataFrame, out_path: str = None, out_type='png',
                new_vis=False, vmin=None, vmax=None, **plot_kwrgs):
        import seaborn as sns
        sns.set_theme()

        plt.figure(figsize=self.figsize)
        ax = sns.heatmap(data, vmin=vmin, vmax=vmax)
        fig = ax.get_figure()
        self.add_kwrgs(plt, **plot_kwrgs)
        fig.savefig(out_path)
        plt.clf()

    def sns_histplot(self, data: pd.DataFrame, out_path,
                     bin_count=100,
                     x: str = None, y: str = None,
                     log_axis: tuple = (False, False),
                     histplot_kwargs: dict = None,
                     **plot_kwargs):
        """ log_axis: use logarithmic axes in (x-axis, y-axis) """
        import seaborn as sns
        sns.set_context('paper')

        histplot_kwargs = {} if histplot_kwargs is None else histplot_kwargs
        default_kwargs = {
            "bins": bin_count,
            "edgecolor": ".3",
            # element='poly'
            # "element": "step",
            "log_scale": log_axis,
            "fill": True,
            "linewidth": .5,
            "multiple": "dodge",  # "stack", "fill", "dodge"
            "palette": "deep",
            # palette="light:m_r",
            "title": plot_kwargs.get('title'),
            "hue": plot_kwargs.get('hue')
        }
        default_kwargs.update(histplot_kwargs)
        self.sns_generic_plot(sns.histplot, out_path,
                              x, y, data, **default_kwargs)

    def histplot(self, data: pd.DataFrame, out_path,
                 bin_count=100,
                 **plot_kwrgs):
        logging.warning(f'Function "histplot" is not finished.')
        plt.figure()
        plt.hist(data, range=(min(data), max(data)), bins=bin_count)
        self.add_kwrgs(plt, **plot_kwrgs)
        plt.savefig(out_path)
        plt.close()
