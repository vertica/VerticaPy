"""
Copyright  (c)  2018-2024 Open Text  or  one  of its
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

__author__: str = "Badr Ouali"
__author_email__: str = "badr.ouali@vertica.com"
__description__: str = (
    "VerticaPy simplifies data exploration, data cleaning"
    " and machine learning in Vertica."
)
__url__: str = "https://github.com/vertica/verticapy/"
__license__: str = "Apache License, Version 2.0"
__version__: str = "1.0.0"

from verticapy._config.config import get_option, set_option
from verticapy._utils._sql._vertica_version import vertica_version
from verticapy._utils._logo import verticapy_logo_html, verticapy_logo_str
from verticapy._help import help_start

from verticapy.connection.connect import (
    close_connection,
    connect,
    current_connection,
    current_cursor,
    set_connection,
)
from verticapy.connection.external import set_external_connection
from verticapy.connection.read import available_connections
from verticapy.connection.write import (
    change_auto_connection,
    delete_connection,
    new_connection,
)

from verticapy.core.parsers.all import read_file
from verticapy.core.parsers.avro import read_avro
from verticapy.core.parsers.csv import read_csv, pcsv
from verticapy.core.parsers.json import read_json, pjson
from verticapy.core.parsers.pandas import read_pandas, read_pandas as pandas_to_vertica
from verticapy.core.parsers.shp import read_shp
from verticapy.core.string_sql.base import StringSQL
from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame, vDataColumn

from verticapy.sql.create import create_schema, create_table
from verticapy.sql.drop import drop
from verticapy.sql.dtypes import get_data_types, vertica_python_dtype
from verticapy.sql.insert import insert_into
from verticapy.sql.sys import (
    current_session,
    username,
    does_table_exist,
    has_privileges,
)

##
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
##
