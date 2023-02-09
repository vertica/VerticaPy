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
import datetime
import numpy as np


def python_to_dtype_category(expr):
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
            category = ""
    return category


def python_to_sql_dtype(dtype):
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


def verticapy_agg_name(key: str, method: str = ""):
    key = key.lower()
    if key in ("median", "med"):
        key = "50%"
    elif key in ("approx_median", "approximate_median"):
        key = "approx_50%"
    elif key == "100%":
        key = "max"
    elif key == "0%":
        key = "min"
    elif key == "approximate_count_distinct":
        key = "approx_unique"
    elif key == "approximate_count_distinct":
        key = "approx_unique"
    elif key == "ema":
        key = "exponential_moving_average"
    elif key == "mean":
        key = "avg"
    elif key in ("stddev", "stdev"):
        key = "std"
    elif key == "product":
        key = "prod"
    elif key == "variance":
        key = "var"
    elif key == "kurt":
        key = "kurtosis"
    elif key == "skew":
        key = "skewness"
    elif key in ("top1", "mode"):
        key = "top"
    elif key == "top1_percent":
        key = "top_percent"
    elif "%" == key[-1]:
        start = 7 if len(key) >= 7 and key[0:7] == "approx_" else 0
        if float(key[start:-1]) == int(float(key[start:-1])):
            key = f"{int(float(key[start:-1]))}%"
            if start == 7:
                key = "approx_" + key
    elif key == "row":
        key = "row_number"
    elif key == "first":
        key = "first_value"
    elif key == "last":
        key = "last_value"
    elif key == "next":
        key = "lead"
    elif key in ("prev", "previous"):
        key = "lag"
    if method == "vertica":
        if key == "var":
            key = "variance"
        elif key == "std":
            key = "stddev"
    return key
