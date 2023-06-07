# DARPA Task 1

## Installation

This project requires Python 3; it has been tested on Python 3.10 for Linux. It may be installed locally using the following command in the root directory of the project:

```bash
pip install .
```

Note that this project requires Plantri to be installed in the working directory (i.e., where the run script is located). For convenience, a copy of Plantri is compiled in the "examples" folder to run the examples.

To install Plantri, follow these steps:

1. Download the source code from the link above and extract the archive.

Plantri can be downloaded from [here](http://users.cecs.anu.edu.au/~bdm/plantri/plantri.html). The source code is available as a `.tar.gz` archive. Download the archive and extract it to a directory of your choice.

2. Compile the source code.

   ```bash
   cd plantri53
   make plantri
   ```

   This will compile the Plantri source code and create an executable called `plantri` in the same directory.


3. Verify that the Plantri command is now available by running the following command:

   ```
   plantri -h
   ```

   This should display the Plantri help message.


## Usage

Usage instructions for this project are documented in interaction Python notebooks (Jupyer Notebooks) within the /examples folder.

## API Reference

The API reference for this project is documented in the /docs folder.

