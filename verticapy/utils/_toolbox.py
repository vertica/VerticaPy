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
#
# Modules
#
# Standard Python Modules
import shutil, re, sys, warnings, random, itertools, datetime, time, html, os
from collections.abc import Iterable
from typing import Union, Literal

# VerticaPy Modules
import verticapy as vp
from verticapy.errors import *
from verticapy.io.sql._utils._format import quote_ident

# Other Modules
import numpy as np
import pandas as pd

# IPython - Optional
try:
    from IPython.display import HTML, display
except:
    pass

#
#
# Functions to use to simplify the coding.
#
def bin_spatial_to_str(
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


def color_dict(d: dict, idx: int = 0):
    if "color" in d:
        if isinstance(d["color"], str):
            return d["color"]
        else:
            return d["color"][idx % len(d["color"])]
    else:
        from verticapy.plotting._colors import gen_colors

        return gen_colors()[idx % len(gen_colors())]


def executeSQL(
    query: str,
    title: str = "",
    data: list = [],
    method: Literal[
        "cursor", "fetchrow", "fetchall", "fetchfirstelem", "copy"
    ] = "cursor",
    path: str = "",
    print_time_sql: bool = True,
    sql_push_ext: bool = False,
    symbol: str = "$",
):
    from verticapy.sdk.vertica.dblink import replace_external_queries_in_query
    from verticapy.io.sql._utils._format import clean_query, erase_label
    from verticapy.io.sql._utils._display import print_query

    # Cleaning the query
    if sql_push_ext and (symbol in vp.SPECIAL_SYMBOLS):
        query = erase_label(query)
        query = symbol * 3 + query.replace(symbol * 3, "") + symbol * 3

    elif sql_push_ext and (symbol not in vp.SPECIAL_SYMBOLS):
        raise ParameterError(f"Symbol '{symbol}' is not supported.")

    query = replace_external_queries_in_query(query)
    query = clean_query(query)

    cursor = vp.current_cursor()
    if vp.OPTIONS["sql_on"] and print_time_sql:
        print_query(query, title)
    start_time = time.time()
    if data:
        cursor.executemany(query, data)
    elif method == "copy":
        with open(path, "r") as fs:
            cursor.copy(query, fs)
    else:
        cursor.execute(query)
    elapsed_time = time.time() - start_time
    if vp.OPTIONS["time_on"] and print_time_sql:
        print_time(elapsed_time)
    if method == "fetchrow":
        return cursor.fetchone()
    elif method == "fetchfirstelem":
        return cursor.fetchone()[0]
    elif method == "fetchall":
        return cursor.fetchall()
    return cursor


def find_val_in_dict(x: str, d: dict, return_key: bool = False):
    for elem in d:
        if quote_ident(x).lower() == quote_ident(elem).lower():
            if return_key:
                return elem
            return d[elem]
    raise NameError(f'Key "{x}" was not found in {d}.')


def gen_name(L: list):
    return "_".join(
        [
            "".join(ch for ch in str(elem).lower() if ch.isalnum() or ch == "_")
            for elem in L
        ]
    )


def gen_tmp_name(schema: str = "", name: str = ""):
    from verticapy.io.sql.sys import current_session, username

    session_user = f"{current_session()}_{username()}"
    L = session_user.split("_")
    L[0] = "".join(filter(str.isalnum, L[0]))
    L[1] = "".join(filter(str.isalnum, L[1]))
    random_int = random.randint(0, 10e9)
    name = f'"_verticapy_tmp_{name.lower()}_{L[0]}_{L[1]}_{random_int}_"'
    if schema:
        name = f"{quote_ident(schema)}.{name}"
    return name

def get_category_from_vertica_type(ctype: str = ""):
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


def get_final_vertica_type(
    type_name: str, display_size: int = 0, precision: int = 0, scale: int = 0
):
    """
Takes as input the Vertica Python type code and returns its corresponding data type.
    """
    result = type_name
    has_precision_scale = (
        (type_name[0:4].lower() not in ("uuid", "date", "bool"))
        and (type_name[0:5].lower() != "array")
        and (type_name[0:3].lower() not in ("set", "row", "map", "int"))
    )
    if display_size and has_precision_scale:
        result += f"({display_size})"
    elif scale and precision and has_precision_scale:
        result += f"({precision},{scale})"
    return result


def get_match_index(x: str, col_list: list, str_check: bool = True):
    for idx, col in enumerate(col_list):
        if (str_check and quote_ident(x.lower()) == quote_ident(col.lower())) or (
            x == col
        ):
            return idx
    return None


def get_random_function(rand_int=None):
    random_state = vp.OPTIONS["random_state"]
    if isinstance(rand_int, int):
        if isinstance(random_state, int):
            random_func = f"FLOOR({rand_int} * SEEDED_RANDOM({random_state}))"
        else:
            random_func = f"RANDOMINT({rand_int})"
    else:
        if isinstance(random_state, int):
            random_func = f"SEEDED_RANDOM({random_state})"
        else:
            random_func = "RANDOM()"
    return random_func


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def updated_dict(
    d1: dict, d2: dict, color_idx: int = 0,
):
    d = {}
    for elem in d1:
        d[elem] = d1[elem]
    for elem in d2:
        if elem == "color":
            if isinstance(d2["color"], str):
                d["color"] = d2["color"]
            elif color_idx < 0:
                d["color"] = [elem for elem in d2["color"]]
            else:
                d["color"] = d2["color"][color_idx % len(d2["color"])]
        else:
            d[elem] = d2[elem]
    return d
