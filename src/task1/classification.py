"""CLASSIFICATION

This module contains functions for classifying nodes based on their nearest reference point.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist


def filter_by_environmental_factors(all_geometric_realizations, component_radii, environmental_factors, plot=True):
    """
    Plot nodes with colors based on their nearest reference point.

    :param component_positions: The positions of the comparison nodes as a 2D array of shape (N, 3).
    :type component_positions: array_like

    :param node_positions: The positions of the nodes as a 2D array of shape (M, 3).
    :type node_positions: array_like

    :param environmental_factors: A list of tuples, where each tuple contains a reference point as a 1D array of shape (3,)
        and a color as a string. The reference points are used to classify the nodes.
    :type environmental_factors: list of tuple

    :return: None
    :rtype: NoneType
    """

    all_filtered_geometric_realizations = {}

    for spatial_graph, geometric_realizations in all_geometric_realizations.items():

        filtered_geometric_realizations = {}
        unqiue_codes = []
        for geometric_realization in geometric_realizations.values():

            node_positions, edges = geometric_realization
            component_nodes = list(component_radii.keys())
            component_positions = np.array([node_positions[node] for node in component_nodes])

            # Convert node positions and reference points to numpy arrays
            # node_xyz = np.array(node_positions)
            node_xyz = np.array(list(node_positions.values()))
            ref_xyz = np.array([p[0] for p in environmental_factors])
            ref_colors = [p[1] for p in environmental_factors]

            # Compute the distances between each node and each reference point
            distances = np.sqrt(((node_xyz[:, np.newaxis, :] - ref_xyz) ** 2).sum(axis=2))

            # Find the index of the nearest reference point for each node
            closest_env_indices = np.argmin(distances, axis=1)
            closest_env_colors = [ref_colors[i] for i in closest_env_indices]

            component_indices = [list(node_positions.keys()).index(node) for node in component_nodes]

            # Create a list that identifies which environmental factor is closest to each component node
            closest_factors = tuple(closest_env_colors[i] for i in component_indices)

            # Plot the nodes with colors based on their nearest reference point
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            for i, ref_color in enumerate(ref_colors):
                mask = closest_env_indices == i
                ax.scatter(*node_xyz[mask].T, s=50, ec="w", c=ref_color)

            # Compute the distances between each node and each reference point
            distances2 = np.sqrt(((component_positions[:, np.newaxis, :] - ref_xyz) ** 2).sum(axis=2))

            # Find the index of the nearest reference point for each node
            nearest_ref_indices2 = np.argmin(distances2, axis=1)

            for i, ref_color in enumerate(ref_colors):
                mask2 = nearest_ref_indices2 == i

                s = [np.pi * (component_radii[v]) ** 2 * 1000 for v in component_nodes]
                s_arr = np.array(s)
                s_arr = s_arr[mask2]
                s = list(s_arr)

                ax.scatter(*component_positions[mask2].T, s=s, ec="w", c=ref_color)


            for ref_point, ref_color in zip(ref_xyz, ref_colors):
                ax.scatter(*ref_point.T, s=100, ec="w", c=ref_color)

            plt.show()

            if closest_factors not in unqiue_codes:
                unqiue_codes.append(closest_factors)
                filtered_geometric_realizations[closest_factors] = geometric_realization

        all_filtered_geometric_realizations[spatial_graph] = filtered_geometric_realizations

    return all_filtered_geometric_realizations



def filter_by_internal_factors(all_geometric_realizations, component_radii, internal_factors, k=3):
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

    all_filtered_geometric_realizations = {}

    for spatial_graph, geometric_realizations in all_geometric_realizations.items():

        filtered_geometric_realizations = {}
        unqiue_codes = []
        for environmental_code, geometric_realization in geometric_realizations.items():

            node_positions, edges = geometric_realization

            component_nodes = list(component_radii.keys())
            component_positions = np.array([node_positions[node] for node in component_nodes])
            nodes = component_nodes
            node_xyz = component_positions

            # nodes = list(node_positions.keys())
            # node_xyz = np.array([node_positions[v] for v in nodes])
            dist_matrix = cdist(node_xyz, node_xyz)
            nearest_neighbors = {}
            for i, node in enumerate(nodes):
                distances = dist_matrix[i]
                neighbors = np.argsort(distances)[1:k+1]
                nearest_neighbors[node] = [nodes[n] for n in neighbors]

            # Delete entries from dictionary that are not important nodes
            for node in list(nearest_neighbors.keys()):
                if node not in internal_factors:
                    del nearest_neighbors[node]

            internal_code = tuple([tuple(nearest_neighbors[node]) for node in nearest_neighbors.keys()])

            unique_code = tuple([internal_code, environmental_code])

            if unique_code not in unqiue_codes:
                unqiue_codes.append(unique_code)
                filtered_geometric_realizations[unique_code] = geometric_realization

        all_filtered_geometric_realizations[spatial_graph] = filtered_geometric_realizations

    return all_filtered_geometric_realizations
