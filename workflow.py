import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from yamada.enumeration import enumerate_yamada_classes
from yamada.dr





# # Define the nodes and edges of a cube
# nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
# edges = [('a', 'b'), ('a', 'd'), ('a', 'e'), ('b', 'c'), ('b', 'f'), ('c', 'd'), ('c', 'g'), ('d', 'h'), ('e', 'f'),
#          ('e', 'h'), ('f', 'g'), ('g', 'h')]
#
#
# g = nx.Graph()
# g.add_nodes_from(nodes)
# g.add_edges_from(edges)
#
# nx.draw(g, with_labels=True)
# plt.show()
#
# max_crossings = 3
#
# # Enumerate the Yamada classes
# yamada_classes = enumerate_yamada_classes(g, max_crossings)


G = nx.MultiGraph([(1, 2), (1,5),(1,6),(2,3),(2,3), (3,4), (4,5),(4,6),(5,6)])

plantri_directory="./plantri53/"
number_of_crossings = 2
data = enumerate_yamada_classes(plantri_directory,G, number_of_crossings)
print(data)