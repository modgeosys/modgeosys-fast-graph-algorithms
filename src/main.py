"""Usage example(s)."""

import pickle
import networkx as nx
from pygments import highlight

from modgeosys.graph.cuda.steiner import manhattan_distance, euclidean_distance, approximate_steiner_minimal_tree, is_gpu_available, construct_minimum_spanning_tree, plot_graph_with_highlighted_nodes, \
    partition_nx_graph_by_reachability, create_nx_graph, GRAPH_NODE_COORDS, GRAPH_EDGES, GRAPH_TERMINALS

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Example usage
    # if use_gpu:
    #     nodes = cp.array([(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)], dtype=cp.float32)
    # else:
    #     nodes = np.array([(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)], dtype=np.float32)

    # terminals = [0, 1, 2, 3, 4]
    # edges = [
    #     (0, 1, {'weight': 1}),
    #     (1, 2, {'weight': 2}),
    #     (2, 3, {'weight': 3}),
    #     (3, 4, {'weight': 4}),
    #     (4, 0, {'weight': 5}),
    #     (0, 2, {'weight': 1.5}),
    #     (1, 3, {'weight': 2.5})
    # ]
    # required_currents = {
    #     (0, 1): 15,
    #     (1, 2): 25,
    #     (2, 3): 20,
    #     (3, 4): 10,
    #     (4, 0): 30,
    #     (0, 2): 20,
    #     (1, 3): 15
    # }

    # Cable properties (example)
    # cable_types = [
    #     {'capacity': 10, 'cost': 1},
    #     {'capacity': 20, 'cost': 1.8},
    #     {'capacity': 30, 'cost': 2.5}
    # ]

    with open('/home/kweller/graph.pickle', 'rb') as pickled_sample_larger_graph_file:
        graph = pickle.load(pickled_sample_larger_graph_file)

    use_gpu = False # is_gpu_available()
    distance_function = manhattan_distance

    print('Terminal nodes in original graph: ', len(graph[GRAPH_TERMINALS]))

    nx_graph = create_nx_graph(graph[GRAPH_EDGES], graph[GRAPH_NODE_COORDS])
    # reachable_nodes, unreachable_nodes = partition_nx_graph_by_reachability(nx_graph)
    regular_nodes = [node for node in nx_graph.nodes if node not in graph[GRAPH_TERMINALS]]
    highlighted_nodes = graph[GRAPH_TERMINALS]
    # plot_graph_with_highlighted_nodes(nx_graph, regular_nodes, highlighted_nodes)
    # exit(0)

    minimum_spanning_tree, _, _, _, terminals, _, _, _ = construct_minimum_spanning_tree(graph, distance_function, use_gpu)
    print('Minimum spanning tree (Manhattan distance): ', minimum_spanning_tree.edges(data=True))
    print('Edges in minimum spanning tree (Manhattan distance): ', len(minimum_spanning_tree.edges(data=True)))
    print('Terminal nodes in minimum spanning tree: ', len(terminals))
    # reachable_nodes, unreachable_nodes = partition_nx_graph_by_reachability(minimum_spanning_tree, terminals[0])
    regular_nodes = [node for node in minimum_spanning_tree.nodes if node not in terminals]
    highlighted_nodes = terminals
    # plot_graph_with_highlighted_nodes(minimum_spanning_tree, regular_nodes, highlighted_nodes)
    # exit(0)

    steiner_minimal_tree, _, _, _, terminals, _, _, _ = approximate_steiner_minimal_tree(graph, distance_function, use_gpu)

    print('Steiner minimal tree: ', steiner_minimal_tree.edges(data=True))
    print('Edges in Steiner minimal tree: ', len(steiner_minimal_tree.edges(data=True)))
    print('Terminal nodes in Steiner minimal tree: ', len(terminals))

    # reachable_nodes, unreachable_nodes = partition_nx_graph_by_reachability(steiner_minimal_tree, terminals[0])
    regular_nodes = [node for node in steiner_minimal_tree.nodes if node not in terminals]
    highlighted_nodes = terminals
    plot_graph_with_highlighted_nodes(steiner_minimal_tree, regular_nodes, highlighted_nodes)
