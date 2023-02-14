"""
(c)  Copyright  [2018-2023]  OpenText  or one of its
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
"""
VerticaPy  is   a  Python   library   with   scikit-like
functionality  for  conducting   data science   projects
on data stored in Vertica, taking advantage of Vertica’s
speed and built-in   analytics  and   machine   learning
features. It supports   the entire   data  science  life
cycle,  uses  a ‘pipeline’  mechanism to   sequentialize
data  transformation  operations,  and  offers beautiful
graphical options.

VerticaPy aims  to  do  all  of  the above.  The idea is
simple:  instead of moving data around  for  processing, 
VerticaPy brings the logic to the data.
"""
__author__ = "Badr Ouali"
__author_email__ = "badr.ouali@vertica.com"
__description__ = (
    "VerticaPy simplifies data exploration, data cleaning"
    " and machine learning in Vertica."
)
__url__ = "https://github.com/vertica/verticapy/"
__license__ = "Apache License, Version 2.0"

# VerticaPy Modules IMPORT

# Logo
from verticapy._utils._logo import *

# Connect
from verticapy.connect import *

# Config
from verticapy._config.config import *

# vDataFrame
from verticapy.vdataframe import *

# Utilities
from verticapy.sql.parsers.csv import read_csv
from verticapy.sql.parsers.json import read_json
from verticapy.sql.parsers.avro import read_avro
from verticapy.sql.parsers.shp import read_shp
from verticapy.sql.parsers.all import read_file
from verticapy.sql.parsers.pandas import pandas_to_vertica
from verticapy.sql.flex import (
    compute_flextable_keys,
    compute_vmap_keys,
    isflextable,
    isvmap,
)
from verticapy.sql.create import create_schema, create_table, create_verticapy_schema
from verticapy.sql.drop import drop
from verticapy.sql.read import readSQL, to_tablesample, vDataFrameSQL
from verticapy.sql.insert import insert_into, insert_verticapy_schema
from verticapy._help import help_start, vHelp
from verticapy._version import vertica_version
from verticapy.core.tablesample import tablesample
from verticapy.sql.dtypes import get_data_types

# Learn
import verticapy.learn

# Version
from verticapy._version import *
from verticapy._version import __version__
