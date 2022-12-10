# (c) Copyright [2018-2022] Micro Focus or one of its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="verticapy",
    version="0.12.0",
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
    python_requires=">=3.6",
    install_requires=[
        "matplotlib<=3.5.3",
        "numpy>=1.11.0",
        "pandas>=0.23.0",
        "python-highcharts>=0.4.1",
        "scipy>=1.0.0",
        "tqdm>=4.0.0",
        "vertica-python>=0.11.0",
    ],
    extras_require={
        "all": [
            "descartes>=1.0.0",
            "geopandas>=0.8.0",
            "graphviz>=0.9.0",
            "shapely>=1.6.0",
        ],
    },
    package_data={"": ["*.csv", "*.json"]},
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Database",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
