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
"""
import warnings

warning_message = (
    "Importing from 'verticapy.utilities' is deprecated, "
    "and it will no longer be possible in the next minor release. "
    "Please use 'verticapy' instead to ensure compatibility "
    "with upcoming versions."
)
warnings.warn(warning_message, Warning)

from verticapy._config.config import set_option
from verticapy._utils._sql._collect import save_to_query_profile
from verticapy._utils._sql._vertica_version import vertica_version
from verticapy._help import help_start

from verticapy.core.tablesample.base import TableSample

from verticapy.sql.create import create_schema, create_table
from verticapy.sql.drop import drop
from verticapy.sql.flex import (
    compute_flextable_keys,
    compute_vmap_keys,
    isflextable,
    isvmap,
)
from verticapy.sql.insert import insert_into
from verticapy.core.parsers.all import read_file
from verticapy.core.parsers.avro import read_avro
from verticapy.core.parsers.csv import read_csv, pcsv
from verticapy.core.parsers.json import read_json, pjson
from verticapy.core.parsers.pandas import read_pandas as pandas_to_vertica, read_pandas
from verticapy.core.parsers.shp import read_shp
from verticapy.sql.dtypes import get_data_types
