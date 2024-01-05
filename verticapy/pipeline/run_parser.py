#!/usr/bin/env python3

import subprocess
import os
import argparse

# Get the directory where the current Python script (your Python package script) is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to run_parser.sh
shell_script = os.path.join(script_directory, "run_parser.sh")

# Create an ArgumentParser object to handle command-line arguments
parser = argparse.ArgumentParser(description="Run the parser script")

# Add command-line argument options
parser.add_argument("arg1", help="Argument 1")
parser.add_argument("arg2", help="Argument 2")

# Parse the command-line arguments
args = parser.parse_args()

# Command to execute the shell script with the provided arguments
command = ["/bin/bash", shell_script, args.arg1, args.arg2]

# Run the shell script using subprocess
try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")
