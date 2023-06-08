"""UTILS

This module contains utility functions for task1.
"""

import json

def write_output(data, filepath):
    """
    Writes data to a file in the output directory.

    :param data: The data to write.
    :param filepath: The name of the file to write to.
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


# for i, (graph, position) in enumerate(zip(graphs, positions)):
    # Create a directory for the output
    # directory = output_directory + f'graph_{i}/'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    # # Write the graph to a JSON file
    # graph.write_to_json(directory + 'graph.json')
    #
    # # Write the position to a JSON file
    # with open(directory + 'position.json', 'w') as f:
    #     json.dump(position, f, indent=4)