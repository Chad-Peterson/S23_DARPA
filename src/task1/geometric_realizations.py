import networkx as nx

import numpy as np
from numpy import cos, sin


def rotate_points(positions: np.ndarray,
                  rotation: np.ndarray = 2 * np.random.rand(3)) -> np.ndarray:
    """
    Rotates a set of points about the first 3D point in the array.

    :param positions: A numpy array of 3D points.
    :param rotation: A numpy array of 3 Euler angles in radians.

    :return: new_positions:
    """

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


def isomorphism(g, pos, n=3, rotate=False):
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
    # Implementation details...

    g = g.copy()

    for edge in list(g.edges()):
        g, pos = subdivide_edge(g, edge, pos, n=n)



    if rotate:
        pos = rotate_points(np.array(list(pos.values())))
        pos = {k: v for k, v in zip(list(g.nodes), pos)}

    # k = 1000 * 1/len(g.nodes)
    # nx.set_edge_attributes(g, 5, "weight")

    # TODO Make weights of components larger than weights of edges
    # Set random weights for each edge
    nx.set_edge_attributes(g, {e: np.random.rand() for e in g.edges()}, "weight")

    pos = nx.spring_layout(g, iterations=50, dim=3, pos=pos, weight="weight")

    return g, pos


