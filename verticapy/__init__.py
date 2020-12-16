# (c) Copyright [2018-2020] Micro Focus or one of its affiliates.
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
#
# |_     |~) _  _| _  /~\    _ |.
# |_)\/  |_)(_|(_||   \_/|_|(_|||
#    /
#              ____________       ______
#             / __        `\     /     /
#            |  \/         /    /     /
#            |______      /    /     /
#                   |____/    /     /
#          _____________     /     /
#          \           /    /     /
#           \         /    /     /
#            \_______/    /     /
#             ______     /     /
#             \    /    /     /
#              \  /    /     /
#               \/    /     /
#                    /     /
#                   /     /
#                   \    /
#                    \  /
#                     \/
#                    _
# \  / _  __|_. _ _ |_)
#  \/ (/_|  | |(_(_|| \/
#                     /
# VerticaPy is a Python library with scikit-like functionality to use to conduct
# data science projects on data stored in Vertica, taking advantage Vertica’s
# speed and built-in analytics and machine learning features. It supports the
# entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize
# data transformation operations, and offers beautiful graphical options.
#
# VerticaPy aims to solve all of these problems. The idea is simple: instead
# of moving data around for processing, VerticaPy brings the logic to the data.
#
#
__version__ = "0.4.0"
__author__ = "Badr Ouali"
__author_email__ = "badr.ouali@vertica.com"
__description__ = """VerticaPy simplifies data exploration, data cleaning and machine learning in Vertica."""
__url__ = "https://github.com/vertica/verticapy/"
__license__ = "Apache License, Version 2.0"

# vDataFrame
from verticapy.vdataframe import *

# Utilities
from verticapy.utilities import *

# Connect
from verticapy.connections.connect import *

# SQL Functions
import verticapy.stats

# Learn
import verticapy.learn
import verticapy.learn.tsa

verticapy.options = {
    "cache": True,
    "max_rows": 100,
    "max_columns": 50,
    "percent_bar": None,
    "print_info": True,
    "query_on": False,
    "time_on": False,
    "mode": None,
    "random_state": None,
    "colors": [],
}
