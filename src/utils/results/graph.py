import networkx as nx
from typing import List, Union
import numpy as np


from pyvis.network import Network


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
                    if edge_w_range[1] - edge_w_range[0] == 0:
                        if e[2]['weight'] > 1:
                            squashed_w = 1 - 1/e[2]['weight']
                        else:
                            squashed_w = e[2]['weight']
                    else:
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
