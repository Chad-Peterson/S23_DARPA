"""ENUMERATION

This module contains functions for enumerating Yamada classes (i.e., unique spatial topologies).
"""


import networkx as nx
import subprocess
import io
import itertools
from yamada.spatial_graph_diagrams import Vertex, Edge, Crossing, SpatialGraphDiagram


def read_edge_code(stream, size):
    """
    A helper function that reads a single byte from a stream and returns the corresponding edge code.

    :param stream: A stream to read from.
    :param size: The number of bytes to read.
    """
    ans = [[]]
    for _ in range(size):
        i = int.from_bytes(stream.read(1), 'big')
        if i < 255:
            ans[-1].append(i)
        else:
            ans.append([])
    return ans


def shadows_via_plantri_by_edge_codes(num_trivalent_vertices, num_crossings):
    """
    A function that enumerates the shadows of an abstract graph with a given number of trivalent vertices and
    crossings. These shadows are in intermediate step toward enumerating Yamada classes.

    Note: This is a wrapper around plantri, which must be installed separately. The plantri executable must be in the
    current working directory.

    :param num_trivalent_vertices: The number of trivalent vertices in the abstract graph.
    :param num_crossings: The number of crossings in the abstract graph.
    :return: A list of shadows, where each shadow is a list of edge codes.
    """

    assert num_trivalent_vertices % 2 == 0

    vertices = num_trivalent_vertices + num_crossings
    edges = (3 * num_trivalent_vertices + 4 * num_crossings) // 2
    faces = 2 - vertices + edges

    cmd = ['plantri53/plantri',
           '-p -d',  # simple planar maps, but return the dual
           '-f4',  # maximum valence in the returned dual is <= 4
           '-c1',  # graph should be 1-connected
           '-m2',  # no valence 1 vertices = no loops in the dual
           '-E',  # return binary edge code format
           '-e%d' % edges,
           '%d' % faces]
    proc = subprocess.run(' '.join(cmd), shell=True, capture_output=True)
    stdout = io.BytesIO(proc.stdout)

    assert stdout.read(13) == b'>>edge_code<<'

    shadows = []
    while True:
        b = stdout.read(1)
        if len(b) == 0:
            break
        size = int.from_bytes(b, 'big')
        assert size != 0
        shadows.append(read_edge_code(stdout, size))

    return shadows


class Shadow:
    """
    Shadow is a class used to construct a spatial graph diagram from edge codes.
    """
    def __init__(self, edge_codes):
        self.edge_codes = edge_codes
        self.vertices = [edges for edges in edge_codes if len(edges) == 3]
        self.crossings = [edges for edges in edge_codes if len(edges) == 4]
        self.num_edges = sum(len(edges) for edges in edge_codes) // 2

    def spatial_graph_diagram(self, signs=None, check=True):
        num_cross = len(self.crossings)
        if signs is None:
            signs = num_cross * [0]
        else:
            assert len(signs) == num_cross

        classes = [Edge(i) for i in range(self.num_edges)]
        for v, edges in enumerate(self.vertices):
            d = len(edges)
            V = Vertex(d, 'V%d' % v)
            classes.append(V)
            for i, e in enumerate(edges):
                E = classes[e]
                e = 0 if E.adjacent[0] is None else 1
                V[i] = E[e]

        for c, edges in enumerate(self.crossings):
            C = Crossing('C%d' % c)
            classes.append(C)
            for i, e in enumerate(edges):
                E = classes[e]
                e = 0 if E.adjacent[0] is None else 1
                C[(i + signs[c]) % 4] = E[e]

        return SpatialGraphDiagram(classes, check=check)


def spatial_graph_diagrams_fixed_crossings(G, crossings):
    """
    A function that enumerates the spatial graph diagrams with a given underlying graph and number of crossings.

    :param G: The underlying graph.
    :param crossings: The number of crossings.
    :return: A generator of spatial graph diagrams.

    """
    assert all(d == 3 for v, d in G.degree)
    assert all(a != b for a, b in G.edges())

    raw_shadows = shadows_via_plantri_by_edge_codes(G.number_of_nodes(), crossings)

    for raw_shadow in raw_shadows:
        shadow = Shadow(raw_shadow)
        diagram = shadow.spatial_graph_diagram(check=False)
        U = diagram.underlying_graph()
        if U is not None:
            if nx.is_isomorphic(G, U):
                if not diagram.has_r6():
                    num_cross = len(shadow.crossings)
                    if num_cross == 0:
                        yield diagram
                    else:
                        for signs in itertools.product((0, 1), repeat=num_cross - 1):
                            signs = (0,) + signs
                            D = shadow.spatial_graph_diagram(signs=signs, check=False)
                            if not D.has_r2():
                                yield D


def enumerate_yamada_classes(G, max_crossings):
    """
    A function that enumerates the Yamada classes of a given underlying graph with a given maximum number of crossings.
    :param G: The underlying graph.
    :param max_crossings: The maximum number of crossings.
    :return: A dictionary mapping Yamada polynomials to spatial graph diagrams and the number of examined shadows.
    """

    examined = 0
    polys = dict()
    for crossings in range(0, max_crossings + 1):
        for D in spatial_graph_diagrams_fixed_crossings(G, crossings):
            p = D.normalized_yamada_polynomial()
            if p not in polys:
                polys[p] = D
            examined += 1
    return polys, examined