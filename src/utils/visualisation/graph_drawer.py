import networkx as nx


def visualise_graph_pyvis(graph: nx.DiGraph,
                          plot_name='test_graph.html'):
    from pyvis.network import Network
    import webbrowser
    import os

    interactive_graph = Network(height=800, width=800, directed=True, notebook=True)
    interactive_graph.from_nx(graph, edge_weight_transf=lambda x: round(x, 4))
    # interactive_graph.show_buttons(filter_=['edges'])
    interactive_graph.inherit_edge_colors(True)
    interactive_graph.set_edge_smooth('dynamic')
    interactive_graph.show(plot_name)

    print("Opening graph in browser...")
    web_filename = 'file:///' + os.getcwd() + '/' + plot_name
    webbrowser.open(web_filename, new=1, autoraise=True)

    print('debug over')

def visualise_graph_pyplot(graph: nx.DiGraph):
    import matplotlib.pyplot as plt
    ax1 = plt.subplot(111)
    nx.draw(graph)
    plt.show()