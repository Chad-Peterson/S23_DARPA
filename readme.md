# DARPA Task 1

## Installation

This project requires Plantri to be installed. Plantri can be downloaded from [here](http://users.cecs.anu.edu.au/~bdm/plantri/plantri.html). Available via xommand terminal..

For Linux:


If you have compiled Plantri from source code on Linux, you may need to add the directory containing the Plantri executable to your system's `PATH` environment variable. 

Here are the steps to add the Plantri directory to your `PATH`:

1. Determine the directory where the Plantri executable is located. This will depend on where you compiled Plantri and where you installed it. For example, if you compiled Plantri in your home directory and installed it in a subdirectory called `plantri`, the executable might be located at `~/plantri/plantri`.

2. Open your shell configuration file. This will depend on which shell you are using. For example, if you are using the Bash shell, you can open the configuration file by running the following command:

   ```
   nano ~/.bashrc
   ```

3. Add the following line to the end of the file, replacing `/path/to/plantri` with the actual path to the Plantri executable:

   ```
   export PATH=$PATH:/path/to/plantri
   ```

4. Save and close the file.

5. Reload your shell configuration by running the following command:

   ```
   source ~/.bashrc
   ```

6. Verify that the Plantri command is now available by running the following command:

   ```
   plantri -h
   ```

   This should display the Plantri help message.

If you are using a different shell or operating system, the steps may be slightly different, but the general idea is the same: you need to add the directory containing the Plantri executable to your system's `PATH` environment variable so that the shell can find it.

## Usage