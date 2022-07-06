from functools import partial
import networkx as nx
import numpy as np
import pandas as pd
import logging
from src.utils.results.writer import DataWriter
from src.utils.misc.string_handling import add_outtype, make_time_str
from pyvis.network import Network
from src.utils.misc.type_handling import merge_dicts


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
        assert(isinstance(nx_graph, nx.Graph))
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
                    e[2]["font"] = 5
                self.add_edge(e[0], e[1], **e[2])

        for node in nx.isolates(nx_graph):
            if 'size' not in nodes[node].keys():
                nodes[node]['size'] = default_node_size
            self.add_node(node, **nodes[node])


def visualise_data(data: pd.DataFrame, data_writer: DataWriter = None,
                   cols: list = None, plot_type='histplot', out_name='test_plot',
                   preprocessor_func=None, exclude_rows_nonempty_in_cols: list = None,
                   use_sns=False, log_axis=False,
                   **plot_kwrgs):
    """ Plot type can be any attributes of VisODE() """
    if exclude_rows_nonempty_in_cols is not None:
        for exc_col in exclude_rows_nonempty_in_cols:
            data = data[data[exc_col] == '']

    visualiser = VisODE()
    if plot_type == 'histplot':
        for col in cols:
            if preprocessor_func:
                data[col] = preprocessor_func(data[col].values)
            if use_sns:
                plot_kwrgs.update({'column': col, 'data': data})
            else:
                plot_kwrgs.update({'data': data[col]})

            data_writer.output(out_type='png', out_name=out_name,
                               write_func=visualiser.histplot, **merge_dicts({'use_sns': use_sns, "log_axis": log_axis},
                                                                         plot_kwrgs))
    elif plot_type == 'plot':
        try:
            x, y = cols[0], cols[-1]
            if preprocessor_func:
                x, y = preprocessor_func(x.values), preprocessor_func(y.values)
            data_writer.output(out_type='png', out_name=out_name,
                               write_func=visualiser.plot, **merge_dicts({'data': x, 'y': y}, plot_kwrgs))
        except IndexError:
            assert len(
                cols) == 2, 'For visualising a plot from a table, please only provide 2 columns as variables.'
    else:
        logging.warn(f'Could not find visualiser function {plot_type}')


def visualise_graph_pyvis(graph: nx.DiGraph,
                          out_path: str,
                          new_vis=False):
    import webbrowser
    import os

    out_type = 'html'

    if new_vis:
        out_path = f'{out_path}_{make_time_str()}'
    out_path = add_outtype(out_path, out_type)

    interactive_graph = NetworkCustom(
        height=800, width=800, directed=True, notebook=True)
    interactive_graph.from_nx(graph, edge_weight_transf=lambda x: round(x, 4))
    interactive_graph.inherit_edge_colors(True)
    interactive_graph.set_edge_smooth('dynamic')
    interactive_graph.show(out_path)

    web_filename = 'file:///' + os.getcwd() + '/' + out_path
    webbrowser.open(web_filename, new=1, autoraise=True)


def visualise_graph_pyplot(graph: nx.DiGraph):
    import matplotlib.pyplot as plt
    ax1 = plt.subplot(111)
    nx.draw(graph)
    plt.show()
    raise NotImplementedError


class VisODE():
    def __init__(self) -> None:
        pass

    def add_kwrgs(self, plt, **plot_kwrgs):
        def warning_call(object, function, dummy_value):
            logging.warn(
                f'Could not call function {function} in object {object}.')

        for plot_kwrg, value in plot_kwrgs.items():
            if value is not None:
                getattr(plt, plot_kwrg, partial(
                    warning_call, object=plt, function=plot_kwrg))(value)

    def plot(self, data, y=None, new_vis=False, out_path='test_plot', out_type='png',
             **plot_kwrgs) -> None:
        from matplotlib import pyplot as plt
        plt.figure()
        if y is not None:
            plt.plot(data, y)
        else:
            plt.plot(data)
        self.add_kwrgs(plt, **plot_kwrgs)
        plt.savefig(out_path)
        plt.close()

    # Make visualisations for each analytic chosen
    def heatmap(self, data: pd.DataFrame, out_path: str = None, out_type='png', new_vis=False, **plot_kwrgs):
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set_theme()

        plt.figure()
        ax = sns.heatmap(data)
        fig = ax.get_figure()
        self.add_kwrgs(plt, **plot_kwrgs)
        fig.savefig(out_path)
        plt.clf()

    def histplot(self, data, out_path, bin_count=50,
                 use_sns=False, column: str = None, log_axis: tuple = (False, False), **plot_kwrgs):
        from matplotlib import pyplot as plt
        if use_sns:
            import seaborn as sns
            data = data.reset_index()
            if log_axis[0]:
                new_col = 'Log of ' + column
                data = data.rename(columns={column: new_col})
                column = new_col

            if column is None:
                logging.warn(
                    'Make sure a column is specified for visualising with seaborn')
            sns.set_theme(style="ticks")
            f, ax = plt.subplots(figsize=(7, 5))
            sns.despine(f)
            sns.histplot(
                data[data[column] != 0],
                x=column,  # hue="cut",
                bins=bin_count,
                multiple="stack",
                palette="light:m_r",
                edgecolor=".3",
                linewidth=.5,
                log_scale=log_axis,
            )
            f.savefig(out_path)
        else:
            plt.figure()
            plt.hist(data, range=(min(data), max(data)), bins=bin_count)
            self.add_kwrgs(plt, **plot_kwrgs)
            plt.savefig(out_path)
            plt.close()
