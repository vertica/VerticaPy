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

from verticapy._version import __version__, vertica_version
from verticapy._config.config import set_option
from verticapy._utils._logo import gen_verticapy_logo_html, gen_verticapy_logo_str
from verticapy._help import help_start

from verticapy.connect.external import set_external_connection
from verticapy.connect.connect import (
    close_connection,
    connect,
    current_connection,
    current_cursor,
    set_connection,
)
from verticapy.connect.write import (
    change_auto_connection,
    delete_connection,
    new_connection,
)
from verticapy.connect.read import available_connections

from verticapy.core.vdataframe import vDataFrame
from verticapy.core.tablesample.base import tablesample
from verticapy.core.str_sql.base import str_sql

from verticapy._version import vertica_version
from verticapy._config.config import set_option
from verticapy._utils._collect import save_to_query_profile

from verticapy.sql.parsers.all import read_file
from verticapy.sql.parsers.avro import read_avro
from verticapy.sql.parsers.csv import read_csv
from verticapy.sql.parsers.json import read_json
from verticapy.sql.parsers.pandas import read_pandas, read_pandas as pandas_to_vertica
from verticapy.sql.parsers.shp import read_shp
from verticapy.sql.create import create_schema, create_table, create_verticapy_schema
from verticapy.sql.drop import drop
from verticapy.sql.dtypes import get_data_types
from verticapy.sql.insert import insert_into
from verticapy.sql.read import readSQL, to_tablesample, vDataFrameSQL
from verticapy.sql.sys import current_session, username
