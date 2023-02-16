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
from verticapy._version import vertica_version
from verticapy._config.config import set_option
from verticapy._utils._collect import save_to_query_profile

from verticapy.sql.parsers.all import read_file
from verticapy.sql.parsers.avro import read_avro
from verticapy.sql.parsers.csv import read_csv, pcsv
from verticapy.sql.parsers.json import read_json, pjson
from verticapy.sql.parsers.pandas import pandas_to_vertica
from verticapy.sql.parsers.shp import read_shp
from verticapy.sql.create import create_schema, create_table, create_verticapy_schema
from verticapy.sql.drop import drop
from verticapy.sql.insert import insert_into, insert_verticapy_schema
from verticapy.sql.read import readSQL, to_tablesample, vDataFrameSQL

from verticapy.sql.dtypes import get_data_types

from verticapy._help import help_start, vHelp

from verticapy.sql.flex import (
    compute_flextable_keys,
    compute_vmap_keys,
    isflextable,
    isvmap,
)

from verticapy.core.tablesample import tablesample
