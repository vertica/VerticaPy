#!/usr/bin/env python3

import os
import re

# Get the directory where the script is located
script_dir = os.path.dirname(__file__)

# Define the directory path to start the search
start_dir = os.path.join(script_dir, "build", "html")

# Regular expression pattern to find occurrences
pattern_to_find = r'html">verticapy\.'

# Walk through the directory and subdirectories
for root, _, files in os.walk(start_dir):
    for filename in files:
        if filename.endswith(".html"):
            file_path = os.path.join(root, filename)

            # Read the content of the HTML file
            with open(file_path, "r") as file:
                content = file.read()

            # Use regular expression to remove occurrences
            modified_content = re.sub(pattern_to_find, 'html">', content)

            # Write the modified content back to the file
            with open(file_path, "w") as file:
                file.write(modified_content)

            print(f"Modified: {file_path}")
