import networkx as nx


def visualise_graph_pyvis(graph: nx.DiGraph,
                          plot_name='test_graph.html'):
    from pyvis.network import Network
    import webbrowser
    import os

    interactive_graph = Network(height=800, width=800, notebook=True)
    interactive_graph.from_nx(graph)
    interactive_graph.show(plot_name)

    print("Opening graph in browser...")
    web_filename = 'file:///' + os.getcwd() + '/' + plot_name
    webbrowser.open(web_filename, new=1, autoraise=True)


def visualise_graph_pyplot(graph: nx.DiGraph):
    import matplotlib.pyplot as plt
    ax1 = plt.subplot(111)
    nx.draw(graph)
    plt.show()