"""
Copyright  (c)  2018-2025 Open Text  or  one  of its
affiliates.  Licensed  under  the   Apache  License,
Version 2.0 (the  "License"); You  may  not use this
file except in compliance with the License.

You may obtain a copy of the License at:
http://www.apache.org/licenses/LICENSE-2.0

Unless  required  by applicable  law or  agreed to in
writing, software  distributed  under the  License is
distributed on an  "AS IS" BASIS,  WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
See the  License for the specific  language governing
permissions and limitations under the License.
"""

#!/usr/bin/env python
import re

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Single-source the version from verticapy/__init__.py so the package
# metadata and verticapy.__version__ can never drift apart. Parsed rather
# than imported, since importing the package at build time would require
# its dependencies to already be installed.
with open("verticapy/__init__.py", "r", encoding="utf-8") as fh:
    version = re.search(
        r"^__version__\s*:\s*str\s*=\s*[\"']([^\"']+)[\"']", fh.read(), re.MULTILINE
    ).group(1)

setuptools.setup(
    name="verticapy",
    version=version,
    author="Badr Ouali",
    author_email="badr.ouali@outlook.fr",
    url="https://github.com/vertica/VerticaPy",
    keywords="vertica python ml data science machine learning statistics database",
    description=(
        "VerticaPy simplifies data exploration, data cleaning, and machine"
        " learning in Vertica."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "graphviz>=0.20.1",
        "ipython>=8.12.0",
        "matplotlib>=3.9.0",
        "numpy>=2.1.0",
        "pandas>=2.2.0",
        "plotly>=5.24.0",
        "scipy>=1.14.0",
        "tqdm>=4.66.0",
        "vertica-highcharts>=0.1.4",
        "vertica-python>=1.4.0",
        "pyyaml>=6.0.1",
        "requests>=2.32.2",
        "urllib3>=2.2.1",
    ],
    extras_require={
        "all": [
            "descartes>=1.1.0",
            "geopandas>=1.0.0",
            "shapely>=2.0.0",
            "pyarrow>=17.0.0",
        ],
    },
    package_data={"": ["*.csv", "*.json", "*.css", "*.html"]},
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Database",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
