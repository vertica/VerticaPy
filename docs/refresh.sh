#!/bin/bash

# Uninstall verticapy
echo y | pip uninstall verticapy

# Execute replace_sphinx_dir.py script
python3 replace_sphinx_dir.py

# Change the theme to sphinx inside verticapy/_config/config.py

# Install the package
pip install ../.

# Clean the build directory
make clean

# Build HTML documentation
make html

# Execute remove_pattern.py script
python3 remove_pattern.py

# Fix search directory for top nav bar + logo
python3 fix_links

# Create Manual TOC tree
python3 create_toc_tree.py

# Modify the header links of User Guide to work for Home/User_guide/Api/
python3 notebook_correction.py

# Reverse replace_sphinx_dir.py script
python3 reverse_replace_sphinx_dir.py

# Run Blck
black ../.