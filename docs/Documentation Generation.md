# Doc generation using sphinx

This document will help you create the documentation for entire VerticaPy library automatically using Sphinx.

## Before you start

- You need to clone the entire VerticaPy repo.
- Open verticapy/_config/config.py and change the theme to sphinx (this ensures that the created plots/graphs are transparent which is needed to accessibility)

```
register_option(
    Option(
        "theme",
        "sphinx",
        "",
        in_validator(["light", "dark", "sphinx"]),
    )
)
```
- Open a bash terminal, cd into ``VerticaPy/docs`` and install all requirements.

```
pip install -r requirements
```
- Additionally install pandoc

```
apt install pandoc
```

- Edit the ``replace_sphinx_dir.py`` file to change the replacement directory to the PWD (present working directory) of docs folder

```
replacement_word = "/project/data/VerticaPy/docs"
```

## Generating the doc

Running the ``refresh.sh`` script will generate the documentation on its own. 

If there are issues, you can run the contents of that refresh script line by line:

- Uninstall verticapy if it is already installed
```
echo y | pip uninstall verticapy
```

- Replacing the directory inside the code
```
python3 replace_sphinx_dir.py
```

- Install the updated verticapy with corrected directory paths
```
pip install ../.
```

- CLean the build
make clean

- Create the build (this may take quite some time)
make html

- Removes some patterns that make the links easier to read
python3 remove_pattern.py

- Fixes some links that for notebooks and other nested files
python3 fix_links

- Create a TOC tree for each page
python3 create_toc_tree.py

- Update notebook links (this should not be needed if ``fix_links`` works)
python3 notebook_correction.py 

At this point, your document should be generated. It should be inside the ``build/html`` folder. 

The next steps are only to revert the changes to the python code.

- Reverse the changes to the directory paths
python3 reverse_replace_sphinx_dir.py

- Run black to fix formatting for the python code.
black ../.
