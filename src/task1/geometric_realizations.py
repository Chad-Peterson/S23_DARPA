"""GEOMETRIC REALIZATIONS

This module contains functions for generating geometric realizations of graphs.
"""

import networkx as nx

import numpy as np
from numpy import cos, sin
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from yamada import SpatialGraph


def rotate_points(positions: np.ndarray,
                  rotation: np.ndarray = None) -> np.ndarray:
    """
    Rotates a set of points about the first 3D point in the array.

    :param positions: A numpy array of 3D points.
    :param rotation: A numpy array of 3 Euler angles in radians.

    :return: new_positions:
    """

    if rotation is None:
        rotation = 2 * np.pi * np.random.rand(3)

    # Shift the object to origin
    reference_position = positions[0]
    origin_positions = positions - reference_position

    alpha, beta, gamma = rotation

    # Rotation matrix Euler angle convention r = r_z(gamma) @ r_y(beta) @ r_x(alpha)

    r_x = np.array([[1., 0., 0.],
                    [0., cos(alpha), -sin(alpha)],
                    [0., sin(alpha), cos(alpha)]])

    r_y = np.array([[cos(beta), 0., sin(beta)],
                    [0., 1., 0.],
                    [-sin(beta), 0., cos(beta)]])

    r_z = np.array([[cos(gamma), -sin(gamma), 0.],
                    [sin(gamma), cos(gamma), 0.],
                    [0., 0., 1.]])

    r = r_z @ r_y @ r_x

    # Transpose positions from [[x1,y1,z1],[x2... ] to [[x1,x2,x3],[y1,... ]
    rotated_origin_positions = (r @ origin_positions.T).T

    # Shift back from origin
    new_positions = rotated_origin_positions + reference_position

    rotated_node_positions = new_positions

    return rotated_node_positions


def subdivide_edge(g, edge, pos, n=3):
    u, v = edge
    g.remove_edge(u, v)

    nodes = [f"{u}{v}{i}" for i in range(1, n)]

    # Add the initial and final nodes
    nodes.insert(0, u)
    nodes.append(v)

    nx.add_path(g, nodes)

    for i in range(1, n):
        g.add_node(f"{u}{v}{i}")
        pos[f"{u}{v}{i}"] = pos[u] + (pos[v] - pos[u]) * i / n

    return g, pos


def isomorphism(g, pos, n=3, rotate=True):
    """
    Generates an isomorphism of a graph with subdivided edges.

    :param g: The graph to generate a new realization for.
    :type g: networkx.Graph

    :param pos: A dictionary mapping node IDs to (x, y, z) tuples representing their positions.
    :type pos: dict

    :param n: The number of subdivisions to make for each edge. Default is 3.
    :type n: int, optional

    :param rotate: Whether to randomly rotate the positions of the nodes. Default is False.
    :type rotate: bool, optional

    :return: A tuple containing the new graph and a dictionary mapping node IDs to their new positions.
    :rtype: tuple
    """

    g = g.copy()

    for edge in list(g.edges()):
        g, pos = subdivide_edge(g, edge, pos, n=n)

    if rotate:
        pos = rotate_points(np.array(list(pos.values())))
        pos = {k: v for k, v in zip(list(g.nodes), pos)}


    # Set random weights for each edge
    nx.set_edge_attributes(g, {e: np.random.rand() for e in g.edges()}, "weight")

    pos = nx.spring_layout(g, iterations=50, dim=3, pos=pos, weight="weight")

    return g, pos


def generate_geometric_realizations_for_one_topology(spatial_graph, component_radii, num_realizations=5, plot=False):
    """
    Generates geometric realizations for a single topology.

    :param spatial_graph: The topology for which to generate geometric realizations.
    :type topology: SpatialGraph

    :param component_radii: A dict of radii for each component in the topology.
    :type component_radii: dict

    :param num_realizations: The number of geometric realizations to generate.
    :type num_realizations: int

    :param plot: Whether to plot the geometric realizations. Default is False.
    :type plot: bool, optional

    :param output_directory: The directory where the output files will be written.
    :type output_directory: dict
    """

    # Extract relevant information from the spatial graph
    nodes = spatial_graph.nodes
    node_positions = spatial_graph.node_positions

    # Create a node positions dictionary to see the isomorphism
    pos = {node: np.array(position) for node, position in zip(nodes, node_positions)}

    # Create a networkx graph
    g = nx.Graph()
    g.add_nodes_from(spatial_graph.nodes)
    g.add_edges_from(spatial_graph.edges)

    geometric_realizations_dict = {}

    all_node_positions = []
    all_edges = []
    for i in range(num_realizations):

        g_iso, pos_iso = isomorphism(g, pos, n=7, rotate=True)

        geometric_realizations_dict[i] = [pos_iso, list(g_iso.edges)]

        all_node_positions.append(pos_iso)
        all_edges.append(list(g_iso.edges))


    if plot:

        for node_positions, edges in zip(all_node_positions, all_edges):

            nodes = list(node_positions.keys())
            pos = node_positions

            node_xyz = np.array([pos[v] for v in nodes])
            edge_xyz = np.array([(pos[u], pos[v]) for u, v in edges])

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            ax.scatter(*node_xyz.T, s=50, ec="w")

            # Rename graph nodes from ints to strings
            comp_nodes = list(component_radii.keys())
            comp_xyz = np.array([pos[v] for v in comp_nodes])

            s = [np.pi*(component_radii[v])**2 * 1000 for v in comp_nodes]

            # Plot the component nodes
            ax.scatter(*comp_xyz.T, s=s, ec="w", c="tab:blue")

            for vizedge in edge_xyz:
                ax.plot(*vizedge.T, color="tab:gray")

            plt.show()

    return geometric_realizations_dict


def generate_geometric_realizations_for_all_topologies(spatial_graphs, component_radii, num_realizations=5, plot=False):
    """
    Generates geometric realizations for all topologies.
    
    :param spatial_graphs: The topologies for which to generate geometric realizations.
    :type topology: list

    :param component_radii: A dict of radii for each component in the topology.
    :type component_radii: dict

    :param num_realizations: The number of geometric realizations to generate.
    :type num_realizations: int

    :param plot: Whether to plot the geometric realizations. Default is False.
    :type plot: bool, optional

    :return: A dictionary mapping topologies to a list of geometric realizations.
    :rtype: dict
    """

    geometric_realizations= {}

    for spatial_graph in spatial_graphs:

        geometric_realizations_dict = generate_geometric_realizations_for_one_topology(spatial_graph,
                                                                                       component_radii,
                                                                                       num_realizations=num_realizations,
                                                                                       plot=plot)

        geometric_realizations[spatial_graph] = geometric_realizations_dict


    return geometric_realizations