import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D

from task1 import isomorphism, enumerate_yamada_classes, k_nearest_neighbors

from yamada import SpatialGraph, extract_graph_from_json_file
from yamada.visualization import position_spatial_graphs_in_3D

# Archived examples
# Enumerate the Yamada classes for the complex example takes approximately 70 minutes.
# This code snippet loads the results from a previous run for convenience.
directory        = os.path.dirname(__file__) + '/sample_topologies/'
filepath_simple  = directory + "G6/C1/G6C1I0.json"
filepath_complex = directory + "G10/C4/G10C4I7.json"
nodes, node_positions, edges = extract_graph_from_json_file(filepath_simple)
sg_0  = SpatialGraph(nodes=nodes, node_positions=node_positions, edges=edges)
sgd_0 = sg_0.create_spatial_graph_diagram()
unique_spatial_topologies = {'ust_0': sgd_0}
spatial_graphs = [sg_0]


# Define the system architecture

# Simple system architecture
# sa = [(0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 0), (2, 5), (3, 5), (4, 5)]

# Complex system architecture
# sa = [(0, 1), (0, 3), (0, 8), (1, 8), (1, 9), (2, 6), (2, 9), (3, 2),
#       (3, 4), (4, 5), (4, 7), (5, 6), (5, 9), (7, 6), (8, 7)]

# Create a networkx graph from the system architecture
# sa_graph = nx.MultiGraph()
# sa_graph.add_edges_from(sa)

# nx.draw(sa_graph, with_labels=True)
# plt.show()


# Enumerate the Yamada classes
# number_of_crossings = 1
# unique_spatial_topologies, number_topologies = enumerate_yamada_classes(sa_graph, number_of_crossings)

# Create near-planar geometric realizations of each UST
# sg_inputs = position_spatial_graphs_in_3D(unique_spatial_topologies)

# Plot each unique geometric realization and print it's Yamada polynomial
# spatial_graphs = []
# for sg_input in sg_inputs:
#     sg = SpatialGraph(*sg_input)
#     spatial_graphs.append(sg)
#     sg.plot()
#     sgd = sg.create_spatial_graph_diagram()
#     yp = sgd.normalized_yamada_polynomial()
#     print(yp)



# Generate isomorphisms for the first UST
# sg2 = spatial_graphs[0]
# nodes2 = sg2.nodes
# node_positions2 = sg2.node_positions

# Create a node positions dictionary
pos = {node: np.array(position) for node, position in zip(nodes2, node_positions2)}

# gg = nx.Graph()
# gg.add_nodes_from(sg2.nodes)
# gg.add_edges_from(sg2.edges)
#
# node_xyz = np.array([pos[v] for v in sorted(gg)])
# edge_xyz = np.array([(pos[u], pos[v]) for u, v in gg.edges()])

# graphs = []
# positions = []
# for i in range(10):
#     g, pos = generate_isomorphism(gg, pos, n=7, rotate=True)
#     graphs.append(g)
#     positions.append(pos)


# g2, pos = generate_new_geometric_realization(gg, pos, n=20, rotate=True)
#
# node_xyz = np.array([pos[v] for v in sorted(g2)])
# edge_xyz = np.array([(pos[u], pos[v]) for u, v in g2.edges()])

# node_xyz = np.array([pos[v] for v in sorted(g)])
# edge_xyz = np.array([(pos[u], pos[v]) for u, v in g.edges()])
# sg2_iso = SpatialGraph(nodes=sorted(list(g.nodes)), edges=list(g.edges), node_positions=node_xyz)
# sg2_iso.plot()



#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
#
# ax.scatter(*node_xyz.T, s=100, ec="w")
#
# # Rename graph nodes from ints to strings
# comp_nodes = [node for node in nodes if 'V' in node]
# comp_xyz = np.array([pos[v] for v in comp_nodes])
#
# # Plot the component nodes
# ax.scatter(*comp_xyz.T, s=500, ec="w", c="tab:blue")
#
# for vizedge in edge_xyz:
#     ax.plot(*vizedge.T, color="tab:gray")
#
# plt.show()
#
#








#
#
# # Define external physics sources
# hot_source_1  = [((x, -1, -1), "r") for x in np.linspace(-1, 1, 20)]
# hot_source_2  = [((-1, -1, z), "r") for z in np.linspace(-1, 1, 20)]
# medium_source = [((1, 1, z), "y") for z in np.linspace(-1, 0, 10)]
# cold_source   = [((x, 1, 1), "b") for x in np.linspace(-1, 1, 20)]
#
# ref_points_with_colors = hot_source_1 + hot_source_2 + medium_source + cold_source
#
#
#
#
# plot_nodes_with_colors(comp_xyz, node_xyz, ref_points_with_colors)
#
# important_components = ['V1','V4','V7']
# my_neighbors = k_nearest_neighbors(g, pos, important_components, k=3)
#
# print(my_neighbors)
