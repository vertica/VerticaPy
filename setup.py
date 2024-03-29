"""
Copyright  (c)  2018-2023 Open Text  or  one  of its
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
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="verticapy",
    version="1.0.3",
    author="Badr Ouali",
    author_email="badr.ouali@vertica.com",
    url="https://github.com/vertica/VerticaPy",
    keywords="vertica python ml data science machine learning statistics database",
    description=(
        "VerticaPy simplifies data exploration, data cleaning, and machine"
        " learning in Vertica."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "graphviz>=0.9.0",
        "matplotlib>=3.5.2",
        "numpy>=1.11.0",
        "pandas>=0.23.0",
        "plotly>=5.10.0",
        "scipy>=1.0.0",
        "tqdm>=4.0.0",
        "vertica-highcharts>=0.1.4",
        "vertica-python>=1.2.0",
        "plotly>=5.10.0",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "all": [
            "descartes>=1.0.0",
            "geopandas>=0.8.0",
            "shapely>=1.6.0",
            "pyarrow>=14.0.0",
        ],
    },
    package_data={"": ["*.csv", "*.json", "*.css", "*.html"]},
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Database",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
