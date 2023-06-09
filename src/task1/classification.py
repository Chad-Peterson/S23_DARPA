"""CLASSIFICATION

This module contains functions for classifying nodes based on their nearest reference point.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist


def filter_by_environmental_factors(all_geometric_realizations, component_radii, environmental_factors, plot=True):
    """
    A function that filters geometric realizations by unique combinations of environmental factors.

    :param all_geometric_realizations: A dictionary containing the geometric realizations for each unique spatial topology.
    :type all_geometric_realizations: dict

    :param component_radii: A dictionary containing the radii of each component node.
    :type component_radii: dict

    :param environmental_factors: A list of tuples, where each tuple contains a reference point as a 1D array of shape (3,)
        and a color as a string. The reference points are used to classify the nodes.
    :type environmental_factors: list of tuple

    :param plot: A boolean indicating whether to plot the results.
    :type plot: bool

    :return: A dictionary containing the filtered geometric realizations for each unique spatial topology.
    :rtype: dict
    """

    all_filtered_geometric_realizations = {}

    for spatial_graph, geometric_realizations in all_geometric_realizations.items():

        filtered_geometric_realizations = {}
        unique_codes = []
        for geometric_realization in geometric_realizations.values():

            node_positions, edges = geometric_realization
            component_nodes = list(component_radii.keys())
            component_positions = np.array([node_positions[node] for node in component_nodes])

            # Convert node positions and reference points to numpy arrays
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

            if closest_factors not in unique_codes:
                unique_codes.append(closest_factors)
                filtered_geometric_realizations[closest_factors] = geometric_realization

        all_filtered_geometric_realizations[spatial_graph] = filtered_geometric_realizations

    return all_filtered_geometric_realizations



def filter_by_internal_factors(all_geometric_realizations, component_radii, internal_factors, k=3):
    """
    A function that filters the geometric realizations based on the internal factors of the spatial graph.

    :param all_geometric_realizations: A dictionary containing the geometric realizations for each unique spatial topology.
    :type all_geometric_realizations: dict

    :param component_radii: A dictionary containing the radii of each component node.
    :type component_radii: dict

    :param internal_factors: A list of nodes that are considered internal factors.
    :type internal_factors: list

    :param k: The number of nearest neighbors to consider.
    :type k: int

    :return: A dictionary containing the filtered geometric realizations for each unique spatial topology.
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
