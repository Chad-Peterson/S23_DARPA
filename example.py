import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from yamada import SpatialGraph, generate_isomorphism
from yamada.enumeration import enumerate_yamada_classes
from yamada.visualization import position_spatial_graphs_in_3D






# Define the nodes and edges of a cube
nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
edges = [('a', 'b'), ('a', 'd'), ('a', 'e'), ('b', 'c'), ('b', 'f'), ('c', 'd'),
         ('c', 'g'), ('d', 'h'), ('e', 'f'), ('e', 'h'), ('f', 'g'), ('g', 'h')]

# Create a networkx graph
g = nx.Graph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)

# nx.draw(g, with_labels=True)
# plt.show()

# Enumerate the Yamada classes
plantri_directory   = "./plantri53/"
number_of_crossings = 2

unique_spatial_topologies, number_topologies = enumerate_yamada_classes(plantri_directory, g, number_of_crossings)

# Create near-planar geometric realizations of each UST
sg_inputs = position_spatial_graphs_in_3D(unique_spatial_topologies)

spatial_graphs = []
for sg_input in sg_inputs:
    sg = SpatialGraph(*sg_input)
    spatial_graphs.append(sg)
    sg.plot()
    sgd = sg.create_spatial_graph_diagram()
    yp = sgd.normalized_yamada_polynomial()
    print(yp)


# Generate isomorphisms for the second UST
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






from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

def plot_nodes_with_colors(node_positions, ref_points_with_colors):
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


ref_points_with_colors = [([-1, -1, -1], "r"), ([1, 1, 1], "y"), ([1, -1, 1], "b")]
plot_nodes_with_colors(node_xyz, ref_points_with_colors)
#
#
my_neighbors = k_nearest_neighbors(g2, pos)
#
print(my_neighbors)
#
# print('1')