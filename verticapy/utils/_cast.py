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
from typing import Union
import datetime
import numpy as np


def to_varchar(
    category: str, column: str = "{}",
):
    map_dict = {
        "vmap": f"MAPTOSTRING({column})",
        "binary": f"TO_HEX({column})",
        "spatial": f"ST_AsText({column})",
    }
    if category in map_dict:
        return map_dict[column]
    return column


def to_dtype_category(expr: type) -> str:
    try:
        category = expr.category()
    except:
        if isinstance(expr, float):
            category = "float"
        elif isinstance(expr, int):
            category = "int"
        elif isinstance(expr, str):
            category = "text"
        elif isinstance(expr, (datetime.date, datetime.datetime)):
            category = "date"
        elif isinstance(expr, (dict, list, np.ndarray)):
            category = "complex"
        else:
            category = "undefined"
    return category


def to_sql_dtype(dtype: Union[type, str]):
    if dtype in (str, "str", "string"):
        dtype = "varchar"
    elif dtype == float:
        dtype = "float"
    elif dtype == int:
        dtype = "integer"
    elif dtype == datetime.datetime:
        dtype = "datetime"
    elif dtype == datetime.date:
        dtype = "date"
    elif dtype == datetime.time:
        dtype = "time"
    elif dtype == datetime.timedelta:
        dtype = "interval"
    elif dtype == datetime.timezone:
        dtype = "timestamptz"
    elif dtype in (np.ndarray, np.array, list):
        dtype = "array"
    elif dtype == dict:
        dtype = "row"
    elif dtype == tuple:
        dtype = "set"
    elif isinstance(dtype, str):
        dtype = dtype.lower().strip()
    return dtype


def to_category(ctype: str = ""):
    ctype = ctype.lower().strip()
    if ctype != "":
        if (ctype[0:5] == "array") or (ctype[0:3] == "row") or (ctype[0:3] == "set"):
            return "complex"
        elif (
            (ctype[0:4] == "date")
            or (ctype[0:4] == "time")
            or (ctype == "smalldatetime")
            or (ctype[0:8] == "interval")
        ):
            return "date"
        elif (
            (ctype[0:3] == "int")
            or (ctype[0:4] == "bool")
            or (ctype in ("tinyint", "smallint", "bigint"))
        ):
            return "int"
        elif (
            (ctype[0:3] == "num")
            or (ctype[0:5] == "float")
            or (ctype[0:7] == "decimal")
            or (ctype == "money")
            or (ctype[0:6] == "double")
            or (ctype[0:4] == "real")
        ):
            return "float"
        elif ctype[0:3] == "geo":
            return "spatial"
        elif ("byte" in ctype) or (ctype == "raw") or ("binary" in ctype):
            return "binary"
        elif "uuid" in ctype:
            return "uuid"
        elif ctype.startswith("vmap"):
            return "vmap"
        else:
            return "text"
    else:
        return "undefined"
