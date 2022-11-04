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
# VerticaPy is a Python library with scikit-like functionality for conducting
# data science projects on data stored in Vertica, taking advantage Vertica’s
# speed and built-in analytics and machine learning features. It supports the
# entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize
# data transformation operations, and offers beautiful graphical options.
#
# VerticaPy aims to do all of the above. The idea is simple: instead of moving
# data around for processing, VerticaPy brings the logic to the data.
#
#
__version__ = "0.11.0"
__author__ = "Badr Ouali"
__author_email__ = "badr.ouali@vertica.com"
__description__ = (
    "VerticaPy simplifies data exploration, data cleaning"
    " and machine learning in Vertica."
)
__url__ = "https://github.com/vertica/verticapy/"
__license__ = "Apache License, Version 2.0"

# Logo
from verticapy.logo import *

# vDataFrame
from verticapy.vdataframe import *

# Utilities
from verticapy.utilities import *

# Connect
from verticapy.connect import *

# SQL Functions
import verticapy.stats

# Learn
import verticapy.learn

# Other Modules
import uuid

verticapy.options = {
    "cache": True,
    "colors": [],
    "color_style": "default",
    "connection": {"conn": None, "section": None, "dsn": None},
    "interactive": False,
    "count_on": False,
    "footer_on": True,
    "identifier": str(uuid.uuid1()).replace("-", ""),
    "max_columns": 50,
    "max_rows": 100,
    "mode": None,
    "overwrite_model": True,
    "percent_bar": None,
    "print_info": True,
    "sql_on": False,
    "random_state": None,
    "temp_schema": "public",
    "time_on": False,
    "tqdm": True,
    "vertica_version": None,
}
