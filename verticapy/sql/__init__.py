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
from verticapy.jupyter.extensions.sql_magic import load_ipython_extension

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
from verticapy.sql.flex import (
    compute_flextable_keys,
    compute_vmap_keys,
    isflextable,
    isvmap,
)
