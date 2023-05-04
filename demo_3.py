import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from yamada import SpatialGraph, SpatialGraphDiagram


nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h','ab', 'ad', 'ae', 'bc', 'bf', 'cd', 'cg', 'dh', 'ef', 'eh', 'fg', 'gh']

# edges = [(component_a, waypoint_ab), (waypoint_ab, component_b),
#          (component_a, waypoint_ad), (waypoint_ad, component_d),
#          (component_a, waypoint_ae), (waypoint_ae, component_e),
#          (component_b, waypoint_bc), (waypoint_bc, component_c),
#          (component_b, waypoint_bf), (waypoint_bf, component_f),
#          (component_c, waypoint_cd), (waypoint_cd, component_d),
#          (component_c, waypoint_cg), (waypoint_cg, component_g),
#          (component_d, waypoint_dh), (waypoint_dh, component_h),
#          (component_e, waypoint_ef), (waypoint_ef, component_f),
#          (component_e, waypoint_eh), (waypoint_eh, component_h),
#          (component_f, waypoint_fg), (waypoint_fg, component_g),
#          (component_g, waypoint_gh), (waypoint_gh, component_h)]

edges = [('a', 'ab'), ('ab', 'b'),
            ('a', 'ad'), ('ad', 'd'),
            ('a', 'ae'), ('ae', 'e'),
            ('b', 'bc'), ('bc', 'c'),
            ('b', 'bf'), ('bf', 'f'),
            ('c', 'cd'), ('cd', 'd'),
            ('c', 'cg'), ('cg', 'g'),
            ('d', 'dh'), ('dh', 'h'),
            ('e', 'ef'), ('ef', 'f'),
            ('e', 'eh'), ('eh', 'h'),
            ('f', 'fg'), ('fg', 'g'),
            ('g', 'gh'), ('gh', 'h')]

pos = {'a': np.array([0, 0, 0]),
       'b': np.array([1, 0, 0]),
       'c': np.array([1, 1, 0]),
       'd': np.array([0, 1, 0]),
       'e': np.array([0, 0, 1]),
       'f': np.array([1, 0, 1]),
       'g': np.array([1, 1, 1]),
       'h': np.array([0, 1, 1]),
       'ab': np.array([0.5, 0, 0]),
       'ad': np.array([0, 0.5, 0]),
       'ae': np.array([0, 0, 0.5]),
       'bc': np.array([1, 0.5, 0]),
       'bf': np.array([1, 0, 0.5]),
       'cd': np.array([0.5, 1, 0]),
       'cg': np.array([1, 1, 0.5]),
       'dh': np.array([0, 1, 0.5]),
       'ef': np.array([0.5, 0, 1]),
       'eh': np.array([0, 0.5, 1]),
       'fg': np.array([1, 0.5, 1]),
       'gh': np.array([0.5, 1, 1])}

g = nx.Graph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)

pos2d = {k: v[:2] for k, v in pos.items()}
nx.draw(g, pos=pos2d)
plt.show()


node_xyz = np.array([pos[v] for v in sorted(g)])
edge_xyz = np.array([(pos[u], pos[v]) for u, v in g.edges()])

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the nodes - alpha is scaled by "depth" automatically
ax.scatter(*node_xyz.T, s=100, ec="w")


# Plot the edges
for vizedge in edge_xyz:
    ax.plot(*vizedge.T, color="tab:gray")

plt.show()



def subdivide_edge(g, edge, pos, n=3):
    u, v = edge
    g.remove_edge(u, v)

    nodes = [f"{u}{v}{i}" for i in range(n)]

    # Add the initial and final nodes
    nodes.insert(0, u)
    nodes.append(v)

    nx.add_path(g, nodes)



    # for i in range(1, n):
    #     g.add_node(f"{u}{v}{i}")
    #     pos[f"{u}{v}{i}"] = pos[u] + (pos[v] - pos[u]) * i / n
    # g.remove_edge(u, v)
    #
    # for i in range(1, n):
    #     g.add_edge(f"{u}{v}{i-1}", f"{u}{v}{i}")
    # return g, pos
#
#
# def subdivide_edges(g, pos, n=3):
#     for edge in list(g.edges()):
#         g, pos = subdivide_edge(g, edge, pos, n=n)
#     return g, pos
#
# g, pos = subdivide_edges(g, pos, n=3)
#
# node_xyz = [pos[v] for v in sorted(g)]
# edge_xyz = np.array([(pos[u], pos[v]) for u, v in g.edges()])
#
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
#
# # Plot the nodes - alpha is scaled by "depth" automatically
# ax.scatter(*node_xyz.T, s=100, ec="w")
# plt.show()