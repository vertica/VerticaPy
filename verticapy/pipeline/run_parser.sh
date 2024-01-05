#!/bin/bash

# Absolute path to your project's root directory
project_root="$HOME/VerticaPy"

# Check if the number of provided arguments is correct
if [ $# -ne 2 ]; then
	    echo "Usage: $0 <connection_file> <recipe_file>"
	        exit 1
fi

# Get the provided connection file and recipe file
connection_file="$(readlink -f "$1")"
recipe_file="$(readlink -f "$2")"

# Change to the directory where your parser.py script is located
cd "$project_root"

# Run the parser.py script with the provided arguments
python3 -m verticapy.pipeline.parser "$connection_file" "$recipe_file"

