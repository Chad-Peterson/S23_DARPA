import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from yamada import SpatialGraph, generate_isomorphism, extract_graph_from_json_file
from yamada.enumeration import enumerate_yamada_classes
from yamada.visualization import position_spatial_graphs_in_3D


# original_graph = [(0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 0), (2, 5), (3, 5), (4, 5)]
original_graph = [(0, 1), (0, 3), (0, 8), (1, 8), (1, 9), (2, 6), (2, 9), (3, 2), (3, 4),
                  (4, 5), (4, 7), (5, 6), (5, 9), (7, 6), (8, 7)]
og = nx.MultiGraph()
og.add_edges_from(original_graph)




# Obtain the local path of this example's directory
directory = os.path.dirname(__file__) + '/sample_topologies/'

# Define the JSON file location
filepath = directory + "G6/C1/G6C1I0.json"
nodes, node_positions, edges = extract_graph_from_json_file(filepath)


# Define the nodes and edges of a cube
# nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
# edges = [('a', 'b'), ('a', 'd'), ('a', 'e'), ('b', 'c'), ('b', 'f'), ('c', 'd'),
#          ('c', 'g'), ('d', 'h'), ('e', 'f'), ('e', 'h'), ('f', 'g'), ('g', 'h')]

# Create a networkx graph
g = nx.Graph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)

# nx.draw(g, with_labels=True)
# plt.show()

# Enumerate the Yamada classes
# plantri_directory   = "./plantri53/"
# number_of_crossings = 2
#
# unique_spatial_topologies, number_topologies = enumerate_yamada_classes(plantri_directory, g, number_of_crossings)
#
# # Create near-planar geometric realizations of each UST
# sg_inputs = position_spatial_graphs_in_3D(unique_spatial_topologies)
#
# spatial_graphs = []
# for sg_input in sg_inputs:
#     sg = SpatialGraph(*sg_input)
#     spatial_graphs.append(sg)
#     sg.plot()
#     sgd = sg.create_spatial_graph_diagram()
#     yp = sgd.normalized_yamada_polynomial()
#     print(yp)

sg1 = SpatialGraph(nodes=nodes, node_positions=node_positions, edges=edges)
spatial_graphs = [sg1]


# Generate isomorphisms for the first UST
sg2 = spatial_graphs[0]
nodes = sg2.nodes
node_positions = sg2.node_positions

# Create a node positions dictionary
pos = {node: np.array(position) for node, position in zip(nodes, node_positions)}

gg = nx.Graph()
gg.add_nodes_from(sg2.nodes)
gg.add_edges_from(sg2.edges)

node_xyz = np.array([pos[v] for v in sorted(gg)])
edge_xyz = np.array([(pos[u], pos[v]) for u, v in gg.edges()])

graphs = []
positions = []
for i in range(10):
    g, pos = generate_isomorphism(gg, pos, n=7, rotate=True)
    graphs.append(g)
    positions.append(pos)


g2, pos = generate_isomorphism(gg, pos, n=7, rotate=True)

node_xyz = np.array([pos[v] for v in sorted(g2)])
edge_xyz = np.array([(pos[u], pos[v]) for u, v in g2.edges()])

# node_xyz = np.array([pos[v] for v in sorted(g)])
# edge_xyz = np.array([(pos[u], pos[v]) for u, v in g.edges()])
# sg2_iso = SpatialGraph(nodes=sorted(list(g.nodes)), edges=list(g.edges), node_positions=node_xyz)
# sg2_iso.plot()




fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(*node_xyz.T, s=100, ec="w")

for vizedge in edge_xyz:
    ax.plot(*vizedge.T, color="tab:gray")

plt.show()








def plot_nodes_with_colors(original_graph_nodes, new_graph, node_positions, ref_points_with_colors):

    # Convert node positions and reference points to numpy arrays
    node_xyz = np.array(node_positions)
    ref_xyz = np.array([p[0] for p in ref_points_with_colors])
    ref_colors = [p[1] for p in ref_points_with_colors]

    # Compute the distances between each node and each reference point
    distances = np.sqrt(((node_xyz[:, np.newaxis, :] - ref_xyz) ** 2).sum(axis=2))

    # Find the index of the nearest reference point for each node
    nearest_ref_indices = np.argmin(distances, axis=1)

    # Plot the nodes with colors based on their nearest reference point
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i, ref_color in enumerate(ref_colors):
        mask = nearest_ref_indices == i
        ax.scatter(*node_xyz[mask].T, s=100, ec="w", c=ref_color)
    for ref_point, ref_color in zip(ref_xyz, ref_colors):
        ax.scatter(*ref_point.T, s=100, ec="w", c=ref_color)

    # Plot the nodes as larger cubes
    ng_nodes = list(new_graph.nodes())
    for i, node in enumerate(ng_nodes):
        if node in original_graph_nodes:
            node_i_xyz = node_positions[i]

            # Plot the node as a cube
            ax.scatter(*node_i_xyz.T, s=500, ec="w", c="k")


    plt.show()


def k_nearest_neighbors(graph, positions, k=3):
    nodes = sorted(graph.nodes())
    node_xyz = np.array([positions[v] for v in nodes])
    dist_matrix = cdist(node_xyz, node_xyz)
    nearest_neighbors = {}
    for i, node in enumerate(nodes):
        distances = dist_matrix[i]
        neighbors = np.argsort(distances)[1:k+1]
        nearest_neighbors[node] = [nodes[n] for n in neighbors]
    return nearest_neighbors


# ref_points_with_colors = [([-1, -1, -1], "r"),
#                           ([-0.9, -1, -1], "r"),
#                           ([-0.8, -1, -1], "r"),
#                           ([-0.7, -1, -1], "r"),
#                           ([-0.6, -1, -1], "r"),
#                           ([-0.5, -1, -1], "r"),
#                           ([1, 1, 1], "y"),
#                           ([1, -1, 1], "b")]

hot_source_1  = [((x, -1, -1), "r") for x in np.linspace(-1, 1, 20)]
hot_source_2  = [((-1, -1, z), "r") for z in np.linspace(-1, 1, 20)]
medium_source = [((1, 1, z), "y") for z in np.linspace(-1, 1, 20)]
cold_source   = [((x, 1, 1), "b") for x in np.linspace(-1, 1, 20)]

ref_points_with_colors = hot_source_1 + hot_source_2 + medium_source + cold_source

# Rename graph nodes from ints to strings
og_nodes = [str(n) for n in og.nodes()]

plot_nodes_with_colors(og_nodes, g2, node_xyz, ref_points_with_colors)
#
#
my_neighbors = k_nearest_neighbors(g2, pos)
#
print(my_neighbors)
#
# print('1')