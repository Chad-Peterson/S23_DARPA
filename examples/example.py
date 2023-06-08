"""EXAMPLE

This module demonstrates how to use the Task 1 package.

By specifying a system architecture and component geometries, the package can be used to
generate a diverse set of layouts to perform geometric optimization on.

The "simple" and "complex" examples correspond to the two examples depicted in the project
presentation.
"""


# %% Import Statements


import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from task1 import enumerate_yamada_classes, filter_by_internal_factors, \
    generate_geometric_realizations_for_one_topology, \
    generate_geometric_realizations_for_all_topologies,\
    filter_by_environmental_factors, write_output

from yamada import SpatialGraph, extract_graph_from_json_file
from yamada.visualization import position_spatial_graphs_in_3D


# %% Archived examples


# Enumerating the Yamada classes for the complex example takes approximately 70 minutes.
# This code snippet loads the results from a previous run for convenience.
# Uncomment the following lines to use the pre-computed results. Make sure to comment out the
# lines of code that enumerate the Yamada classes.

# from yamada import extract_graph_from_json_file
# directory        = os.path.dirname(__file__) + '/sample_topologies/'
# filepath_simple  = directory + "G6/C1/G6C1I0.json"
# filepath_complex = directory + "G10/C4/G10C4I7.json"
# # Specify whether to extract the simple or complex example
# nodes, node_positions, edges = extract_graph_from_json_file(filepath_simple)
# sg_0  = SpatialGraph(nodes=nodes, node_positions=node_positions, edges=edges)
# sgd_0 = sg_0.create_spatial_graph_diagram()
# unique_spatial_topologies = {'ust_0': sgd_0}
# spatial_graphs = [sg_0]


# %% Define the System Architecture

# The system architecture is a NetworkX graph where the nodes represent components and the edges
# represent connections between components. The nodes are labeled with integers starting from 0.

# Currently, components must be either 2- or 3-valent. Please refer to the documentation for
# more information.

# Simple system architecture
sa = [(0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 0), (2, 5), (3, 5), (4, 5)]

# Complex system architecture
# sa = [(0, 1), (0, 3), (0, 8), (1, 8), (1, 9), (2, 6), (2, 9), (3, 2),
#       (3, 4), (4, 5), (4, 7), (5, 6), (5, 9), (7, 6), (8, 7)]

# Create a networkx graph from the system architecture
sa_graph = nx.MultiGraph()
sa_graph.add_edges_from(sa)

# Plot the system architecture
nx.draw(sa_graph, with_labels=True)
plt.show()


# %% Enumerate all Unique Spatial Topologies


number_of_crossings = 1
unique_spatial_topologies, number_topologies = enumerate_yamada_classes(sa_graph, number_of_crossings)


# %% Generate A Near-Planar Geometric Realization of Each Unique Spatial Topology


sg_inputs = position_spatial_graphs_in_3D(unique_spatial_topologies)

# Convert each near-planar geometric realization into a SpatialGraph object
spatial_graphs = []
for sg_input in sg_inputs:
    sg = SpatialGraph(*sg_input)
    spatial_graphs.append(sg)
    # sg.plot()


# %% Generate a Spatially Diverse Set of Geometric Realizations for Each Unique Spatial Topology


geometric_realizations = generate_geometric_realizations_for_all_topologies(spatial_graphs,
                                                                            num_realizations=5,
                                                                            plot=True)


# %% Filter geometric realizations by unique combinations of environmental factors


# Define environmental factors (e.g., a hot pipe from another system)
# Valid environmental factors are defined as a list of tuples, where each tuple is a 3D coordinate and a color
# Each color represents a unique environmental factor (e.g., the 40 points of hot_pipe are grouped as one factor)
# hot_pipe    = [((x, -1, -1), "r") for x in np.linspace(-1, 1, 20)]
# hot_pipe    += [((-1, -1, z), "r") for z in np.linspace(-1, 1, 20)]
# medium_pipe = [((1, 1, z), "y") for z in np.linspace(-1, 0, 10)]
# cold_pipe   = [((x, 1, 1), "b") for x in np.linspace(-1, 1, 20)]
# environmental_sources = hot_pipe + medium_pipe + cold_pipe

# spatial_graph = list(geometric_realizations.keys())[0]
# nodes = spatial_graph.nodes
# pos = list(geometric_realizations.values())[0]
# node_xyz = np.array([pos[v] for v in nodes])

# Rename graph nodes from ints to strings
# comp_nodes = [node for node in nodes if 'V' in node]
# comp_xyz = np.array([pos[v] for v in comp_nodes])


# TODO Loop thru all graphs
# TODO return list, plot optional
# filter_by_environmental_factors(comp_xyz, node_xyz, environmental_sources)


# %% Filter geometric realizations by unique combinations of internal factors

# TODO Loop thru all graphs
# TODO return list, plot optional
# internal_factors = ['V1', 'V4', 'V7']
# my_neighbors = filter_by_internal_factors(g, pos, internal_factors, k=3)
#
# print(my_neighbors)


# %% Write output

# TODO Add write output function
# Writes each geometric realization to an individual JSON file.

# output_directory = os.path.dirname(__file__) + '/output/'
# write_output(geometric_realizations, output_directory)

