"""CLASSIFICATION

This module contains functions for classifying nodes based on their nearest reference point.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist


def plot_nodes_with_colors(comp_positions, node_positions, ref_points_with_colors):

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

    # Compute the distances between each node and each reference point
    distances2 = np.sqrt(((comp_positions[:, np.newaxis, :] - ref_xyz) ** 2).sum(axis=2))

    # Find the index of the nearest reference point for each node
    nearest_ref_indices2 = np.argmin(distances2, axis=1)

    for i, ref_color in enumerate(ref_colors):
        mask2 = nearest_ref_indices2 == i
        ax.scatter(*comp_positions[mask2].T, s=750, ec="w", c=ref_color)


    for ref_point, ref_color in zip(ref_xyz, ref_colors):
        ax.scatter(*ref_point.T, s=100, ec="w", c=ref_color)

    plt.show()

# def plot_nodes_with_colors(node_positions, ref_points_with_colors):
#     """
#     Plots a set of nodes in 3D space, with colors based on their nearest reference point.
#
#     :param node_positions: A list of (x, y, z) tuples representing the positions of the nodes.
#     :type node_positions: list of tuples
#     :param ref_points_with_colors: A list of (x, y, z, color) tuples representing the positions and colors of the reference points.
#     :type ref_points_with_colors: list of tuples
#     :return: None
#     :rtype: None
#     """
#     # Implementation details...
#
#
#     # Convert node positions and reference points to numpy arrays
#     node_xyz = np.array(node_positions)
#     ref_xyz = np.array([p[0] for p in ref_points_with_colors])
#     ref_colors = [p[1] for p in ref_points_with_colors]
#
#     # Compute the distances between each node and each reference point
#     distances = np.sqrt(((node_xyz[:, np.newaxis, :] - ref_xyz) ** 2).sum(axis=2))
#
#     # Find the index of the nearest reference point for each node
#     nearest_ref_indices = np.argmin(distances, axis=1)
#
#     # Plot the nodes with colors based on their nearest reference point
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#     for i, ref_color in enumerate(ref_colors):
#         mask = nearest_ref_indices == i
#         ax.scatter(*node_xyz[mask].T, s=100, ec="w", c=ref_color)
#     for ref_point, ref_color in zip(ref_xyz, ref_colors):
#         ax.scatter(*ref_point.T, s=100, ec="w", c=ref_color)
#     plt.show()


def k_nearest_neighbors(graph, positions, important_nodes, k=3):
    """
    Finds the k nearest neighbors for each important node in the graph.

    :param graph: The graph to find nearest neighbors in.
    :type graph: networkx.Graph
    :param positions: A dictionary mapping node IDs to (x, y, z) tuples representing their positions.
    :type positions: dict
    :param important_nodes: A list of node IDs to find nearest neighbors for.
    :type important_nodes: list
    :param k: The number of nearest neighbors to find for each node. Default is 3.
    :type k: int, optional
    :return: A dictionary mapping each important node to a list of its k nearest neighbors.
    :rtype: dict
    """
    # Implementation details...

    nodes = sorted(graph.nodes())
    node_xyz = np.array([positions[v] for v in nodes])
    dist_matrix = cdist(node_xyz, node_xyz)
    nearest_neighbors = {}
    for i, node in enumerate(nodes):
        distances = dist_matrix[i]
        neighbors = np.argsort(distances)[1:k+1]
        nearest_neighbors[node] = [nodes[n] for n in neighbors]

    # Delete entries from dictionary that are not important nodes
    for node in list(nearest_neighbors.keys()):
        if node not in important_nodes:
            del nearest_neighbors[node]

    return nearest_neighbors
