import networkx as nx


def sort_nodes_by_degree(graph: nx.Graph):
    nodes = list(graph.nodes())
    max_node_idx = max(nodes)
    nodes.sort(key=lambda n: graph.degree(n) - n / (max_node_idx + 1), reverse=True)
    node2idx = {n: i for i, n in enumerate(nodes)}
    sorted_graph = nx.Graph()
    sorted_graph.add_nodes_from(range(len(nodes)))
    for node in nodes:
        neighbors = list(graph.neighbors(node))
        neighbors.sort(key=lambda n: graph.degree(n) - n / (max_node_idx + 1), reverse=True)
        for u in neighbors:
            sorted_graph.add_edge(node2idx[node], node2idx[u])

    return sorted_graph
