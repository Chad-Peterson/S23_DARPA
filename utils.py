import numpy as np
from numpy import sin, cos
import networkx as nx
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


def generate_new_geometric_realization(g, pos, n=3, rotate=False):
    """
    Generates an isomorphism of a graph with subdivided edges.
    """

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


