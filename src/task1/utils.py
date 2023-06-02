import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy import cos, sin
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def plot_nodes_with_colors(node_positions, ref_points_with_colors):
    """
    Plots a set of nodes in 3D space, with colors based on their nearest reference point.

    :param node_positions: A list of (x, y, z) tuples representing the positions of the nodes.
    :type node_positions: list of tuples
    :param ref_points_with_colors: A list of (x, y, z, color) tuples representing the positions and colors of the reference points.
    :type ref_points_with_colors: list of tuples
    :return: None
    :rtype: None
    """
    # Implementation details...


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



def generate_new_geometric_realization(g, pos, n=3, rotate=False):
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


    for edge in list(g.edges()):
        g, pos = subdivide_edge(g, edge, pos, n=n)

    def rotate_points(positions: np.ndarray,
                      rotation:  np.ndarray = 2*np.random.rand(3)) -> np.ndarray:
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

    if rotate:
        pos = rotate_points(np.array(list(pos.values())))
        pos = {k: v for k, v in zip(list(g.nodes), pos)}

    # k = 1000 * 1/len(g.nodes)
    # nx.set_edge_attributes(g, 5, "weight")

    # Set random weights for each edge
    nx.set_edge_attributes(g, {e: np.random.rand() for e in g.edges()}, "weight")

    pos = nx.spring_layout(g, iterations=50, dim=3, pos=pos, weight="weight")



    return g, pos


