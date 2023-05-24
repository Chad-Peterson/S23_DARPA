import numpy as np
import matplotlib.pyplot as plt
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


