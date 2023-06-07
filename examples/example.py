import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D

from task1 import isomorphism, enumerate_yamada_classes, filter_by_internal_factors, \
    generate_geometric_realizations_for_one_topology, filter_by_environmental_factors

from yamada import SpatialGraph, extract_graph_from_json_file
from yamada.visualization import position_spatial_graphs_in_3D

# Archived examples
# Enumerate the Yamada classes for the complex example takes approximately 70 minutes.
# This code snippet loads the results from a previous run for convenience.
# directory        = os.path.dirname(__file__) + '/sample_topologies/'
# filepath_simple  = directory + "G6/C1/G6C1I0.json"
# filepath_complex = directory + "G10/C4/G10C4I7.json"
# nodes, node_positions, edges = extract_graph_from_json_file(filepath_simple)
# sg_0  = SpatialGraph(nodes=nodes, node_positions=node_positions, edges=edges)
# sgd_0 = sg_0.create_spatial_graph_diagram()
# unique_spatial_topologies = {'ust_0': sgd_0}
# spatial_graphs = [sg_0]


# Define the system architecture

# Simple system architecture
sa = [(0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 0), (2, 5), (3, 5), (4, 5)]

# Complex system architecture
# sa = [(0, 1), (0, 3), (0, 8), (1, 8), (1, 9), (2, 6), (2, 9), (3, 2),
#       (3, 4), (4, 5), (4, 7), (5, 6), (5, 9), (7, 6), (8, 7)]

# Create a networkx graph from the system architecture
sa_graph = nx.MultiGraph()
sa_graph.add_edges_from(sa)

nx.draw(sa_graph, with_labels=True)
plt.show()


# Enumerate the Yamada classes
number_of_crossings = 1
unique_spatial_topologies, number_topologies = enumerate_yamada_classes(sa_graph, number_of_crossings)

# Create near-planar geometric realizations of each UST
sg_inputs = position_spatial_graphs_in_3D(unique_spatial_topologies)

# Plot each unique geometric realization and print it's Yamada polynomial
spatial_graphs = []
for sg_input in sg_inputs:
    sg = SpatialGraph(*sg_input)
    spatial_graphs.append(sg)
    sg.plot()

# Gener
graphs, positions = generate_geometric_realizations_for_one_topology(spatial_graphs[0], num_realizations=5, plot=True)


# TODO Add a function
# Define environmental physics sources
hot_pipe_1    = [((x, -1, -1), "r") for x in np.linspace(-1, 1, 20)]
hot_pipe_2    = [((-1, -1, z), "r") for z in np.linspace(-1, 1, 20)]
medium_pipe_1 = [((1, 1, z), "y") for z in np.linspace(-1, 0, 10)]
cold_pipe_1   = [((x, 1, 1), "b") for x in np.linspace(-1, 1, 20)]
environmental_sources = hot_pipe_1 + hot_pipe_2 + medium_pipe_1 + cold_pipe_1


# TODO Loop thru all graphs
# TODO return list, plot optional
filter_by_environmental_factors(comp_xyz, node_xyz, environmental_sources)

# TODO Loop thru all graphs
# TODO return list, plot optional
important_components = ['V1','V4','V7']
my_neighbors = filter_by_internal_factors(g, pos, important_components, k=3)

print(my_neighbors)
