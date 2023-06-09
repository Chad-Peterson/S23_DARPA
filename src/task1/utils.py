"""UTILS

This module contains utility functions for task1.
"""

import os
import json



def write_output(data, output_directory):
    """
    Writes data to a file in the output directory.

    :param data: The data to write.
    :param filepath: The name of the file to write to.
    """

    output_dicts = []
    for spatial_graph, sg_geometric_realizations in data.items():
        for geometric_realization in sg_geometric_realizations.values():
            output_dicts.append(geometric_realization)

    output_dicts_new = []
    # Convert all np.ndarrays to lists
    for geometric_realization in output_dicts:
        node_positions, edges = geometric_realization
        node_positions = {node: node_positions[node].tolist() for node in node_positions}
        output_dicts_new.append((node_positions, edges))


    # Write each dictionary to a separate JSON file
    for i, geometric_realization in enumerate(output_dicts_new):
        filename = f"geometric_realization_{i}.json"
        filepath = os.path.join(output_directory, filename)

        final_format = {"3D_positions":geometric_realization}
        with open(filepath, "w") as f:
            json.dump(final_format, f, indent=4)
