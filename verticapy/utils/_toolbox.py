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
from verticapy.io.sql.utils._format import quote_ident

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


def erase_label(query: str):
    labels = re.findall(r"\/\*\+LABEL(.*?)\*\/", query)
    for label in labels:
        query = query.replace(f"/*+LABEL{label}*/", "")
    return query


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
    from verticapy.io.sql.utils._format import clean_query

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


def extract_col_dt_from_query(query: str, field: str):
    n, m = len(query), len(field) + 2
    for i in range(n - m):
        current_word = query[i : i + m]
        if current_word.lower() == '"' + field.lower() + '"':
            i = i + m
            total_parenthesis = 0
            k = i + 1
            while ((query[i] != ",") or (total_parenthesis > 0)) and i < n - m:
                i += 1
                if query[i] in ("(", "[", "{"):
                    total_parenthesis += 1
                elif query[i] in (")", "]", "}"):
                    total_parenthesis -= 1
            return (current_word, query[k:i])
    return None


def extract_compression(path: str):
    file_extension = path.split(".")[-1].lower()
    lookup_table = {"gz": "GZIP", "bz": "BZIP", "lz": "LZO", "zs": "ZSTD"}
    if file_extension[0:2] in lookup_table:
        return lookup_table[file_extension[0:2]]
    else:
        return "UNCOMPRESSED"


def find_val_in_dict(x: str, d: dict, return_key: bool = False):
    for elem in d:
        if quote_ident(x).lower() == quote_ident(elem).lower():
            if return_key:
                return elem
            return d[elem]
    raise NameError(f'Key "{x}" was not found in {d}.')


def flat_dict(d: dict) -> str:
    # converts dictionary to string with a specific format
    res = []
    for key in d:
        q = '"' if isinstance(d[key], str) else ""
        res += [f"{key}={q}{d[key]}{q}"]
    res = ", ".join(res)
    if res:
        res = f", {res}"
    return res


def gen_name(L: list):
    return "_".join(
        [
            "".join(ch for ch in str(elem).lower() if ch.isalnum() or ch == "_")
            for elem in L
        ]
    )


def gen_tmp_name(schema: str = "", name: str = ""):
    session_user = get_session()
    L = session_user.split("_")
    L[0] = "".join(filter(str.isalnum, L[0]))
    L[1] = "".join(filter(str.isalnum, L[1]))
    random_int = random.randint(0, 10e9)
    name = f'"_verticapy_tmp_{name.lower()}_{L[0]}_{L[1]}_{random_int}_"'
    if schema:
        name = f"{quote_ident(schema)}.{name}"
    return name


def get_category_from_python_type(expr):
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


def get_first_file(path: str, ext: str):
    dirname = os.path.dirname(path)
    files = os.listdir(dirname)
    for f in files:
        file_ext = f.split(".")[-1]
        if file_ext == ext:
            return dirname + "/" + f
    return None


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


def get_narrow_tablesample(t, use_number_as_category: bool = False):
    result = []
    t = t.values
    if use_number_as_category:
        categories_alpha = t["index"]
        categories_beta = [elem for elem in t]
        del categories_beta[0]
        bijection_categories = {}
        for idx, elem in enumerate(categories_alpha):
            bijection_categories[elem] = idx
        for idx, elem in enumerate(categories_beta):
            bijection_categories[elem] = idx
    for elem in t:
        if elem != "index":
            for idx, val_tmp in enumerate(t[elem]):
                try:
                    val = float(val_tmp)
                except:
                    val = val_tmp
                if not (use_number_as_category):
                    result += [[elem, t["index"][idx], val]]
                else:
                    result += [
                        [
                            bijection_categories[elem],
                            bijection_categories[t["index"][idx]],
                            val,
                        ]
                    ]
    if use_number_as_category:
        return result, categories_alpha, categories_beta
    else:
        return result


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


def get_session(add_username: bool = True):
    current_session = executeSQL(
        query="SELECT /*+LABEL(get_session)*/ CURRENT_SESSION();",
        method="fetchfirstelem",
        print_time_sql=False,
    )
    current_session = current_session.split(":")[1]
    current_session = int(current_session, base=16)
    if add_username:
        username = executeSQL(
            query="SELECT /*+LABEL(get_session)*/ USERNAME();",
            method="fetchfirstelem",
            print_time_sql=False,
        )
        return f"{username}_{current_session}"
    return current_session


def get_vertica_type(dtype):
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


def get_verticapy_function(key: str, method: str = ""):
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


def print_query(query: str, title: str = ""):
    from verticapy.io.sql.utils._format import indentSQL

    screen_columns = shutil.get_terminal_size().columns
    query_print = indentSQL(query)
    if isnotebook():
        display(HTML(f"<h4>{title}</h4>"))
        query_print = query_print.replace("\n", " <br>").replace("  ", " &emsp; ")
        display(HTML(query_print))
    else:
        print(f"$ {title} $\n")
        print(query_print)
        print("-" * int(screen_columns) + "\n")


def print_time(elapsed_time: float):
    screen_columns = shutil.get_terminal_size().columns
    if isnotebook():
        display(HTML(f"<div><b>Execution: </b> {round(elapsed_time, 3)}s</div>"))
    else:
        print(f"Execution: {round(elapsed_time, 3)}s")
        print("-" * int(screen_columns) + "\n")


def reverse_score(metric: str):
    if metric in [
        "logloss",
        "max",
        "mae",
        "median",
        "mse",
        "msle",
        "rmse",
        "aic",
        "bic",
        "auto",
    ]:
        return False
    return True


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


#
#
# Tools to merge similar names/categories together.
#


def erase_prefix_in_name(name: str, prefix: list = []):
    """----
Excludes the input lists of prefixes from the input name and returns it.
When there is a match, the other elements of the list are ignored.

Parameters
---------- 
name: str
    Input name.
prefix: list, optional
    List of prefixes.

Returns
-------
name
    The name without the prefixes.
    """
    name_tmp = name
    for p in prefix:
        n = len(p)
        if p in name_tmp and name_tmp[:n] == p:
            name_tmp = name_tmp[n:]
            break
    return name_tmp


def erase_suffix_in_name(name: str, suffix: list = []):
    """----
Excludes the input lists of suffixes from the input name and returns it.
When there is a match, the other elements of the list are ignored.

Parameters
---------- 
name: str
    Input name.
suffix: list, optional
    List of suffixes.

Returns
-------
name
    The name without the suffixes.
    """
    name_tmp = name
    for s in suffix:
        n = len(s)
        if s in name_tmp and name_tmp[-n:] == s:
            name_tmp = name_tmp[:-n]
            break
    return name_tmp


def erase_word_in_name(name: str, word: list = []):
    """----
Excludes the input lists of words from the input name and returns it.
When there is a match, the other elements of the list are ignored.

Parameters
---------- 
name: str
    Input name.
word: list, optional
    List of words.

Returns
-------
name
    The name without the input words.
    """
    for w in word:
        if w in name:
            return name.replace(w, "")
            break
    return name


def erase_in_name(
    name: str,
    suffix: list = [],
    prefix: list = [],
    word: list = [],
    order: list = ["p", "s", "w"],
):
    """----
Excludes the input lists of suffixes and prefixes from the input name and 
returns it. When there is a match, the other elements of the list are ignored.

Parameters
---------- 
name: str
    Input name.
suffix: list, optional
    List of suffixes.
prefix: list, optional
    List of prefixes.
word: list, optional
    List of words.
order: list, optional
    The order of the process.
        s: suffix
        p: prefix
        w: word
    For example the list ["p", "s", "w"] will start by excluding the 
    prefixes, then suffixes and finally the input words.

Returns
-------
name
    The name without the prefixes, suffixes and input words.
    """
    name_tmp = name
    f = {
        "p": (erase_prefix_in_name, prefix),
        "s": (erase_suffix_in_name, suffix),
        "w": (erase_word_in_name, word),
    }
    for x in order:
        name_tmp = f[x][0](name_tmp, f[x][1])
    return name_tmp


def is_similar_name(
    name1: str,
    name2: str,
    skip_suffix: list = [],
    skip_prefix: list = [],
    skip_word: list = [],
    order: list = ["p", "s", "w"],
):
    """----
Excludes the input lists of suffixes, prefixes and words from the input name 
and returns it.

Parameters
---------- 
name1: str
    First name to compare.
name2: str
    Second name to compare.
skip_suffix: list, optional
    List of suffixes to exclude.
skip_prefix: list, optional
    List of prefixes to exclude.
skip_word: list, optional
    List of words to exclude.
order: list, optional
    The order of the process.
        s: suffix
        p: prefix
        w: word
    For example the list ["p", "s", "w"] will start by excluding the 
    prefixes, then suffixes and finally the input words.
    

Returns
-------
bool
    True if the two names are similar, false otherwise.
    """
    n1 = erase_in_name(
        name=name1, suffix=skip_suffix, prefix=skip_prefix, word=skip_word, order=order,
    )
    n2 = erase_in_name(
        name=name2, suffix=skip_suffix, prefix=skip_prefix, word=skip_word, order=order,
    )
    return n1 == n2


def belong_to_group(
    name: str,
    group: list,
    skip_suffix: list = [],
    skip_prefix: list = [],
    skip_word: list = [],
    order: list = ["p", "s", "w"],
):
    """----
Excludes the input lists of suffixes, prefixes and words from the input name 
and looks if it belongs to a specific group.

Parameters
---------- 
name: str
    Input Name.
group: list
    List of names.
skip_suffix: list, optional
    List of suffixes to exclude.
skip_prefix: list, optional
    List of prefixes to exclude.
skip_word: list, optional
    List of words to exclude.
order: list, optional
    The order of the process.
        s: suffix
        p: prefix
        w: word
    For example the list ["p", "s", "w"] will start by excluding the 
    prefixes, then suffixes and finally the input words.

Returns
-------
bool
    True if the name belong to the input group, false otherwise.
    """
    for name2 in group:
        if is_similar_name(
            name1=name,
            name2=name2,
            skip_suffix=skip_suffix,
            skip_prefix=skip_prefix,
            skip_word=skip_word,
            order=order,
        ):
            return True
    return False


def group_similar_names(
    colnames: list,
    skip_suffix: list = [],
    skip_prefix: list = [],
    skip_word: list = [],
    order: list = ["p", "s", "w"],
):
    """----
Creates similar group using the input column names.

Parameters
---------- 
colnames: list
    List of input names.
skip_suffix: list, optional
    List of suffixes to exclude.
skip_prefix: list, optional
    List of prefixes to exclude.
skip_word: list, optional
    List of words to exclude.
order: list, optional
    The order of the process.
        s: suffix
        p: prefix
        w: word
    For example the list ["p", "s", "w"] will start by excluding the 
    prefixes, then suffixes and finally the input words.

Returns
-------
dict
    dictionary including the different groups.
    """
    result = {}
    for col in colnames:
        groupname = erase_in_name(
            name=col,
            suffix=skip_suffix,
            prefix=skip_prefix,
            word=skip_word,
            order=order,
        )
        if groupname not in result:
            result[groupname] = [col]
        else:
            result[groupname] += [col]
    return result


def gen_coalesce(group_dict: dict):
    """----
Generates the SQL statement to merge the groups together.

Parameters
---------- 
group_dict: dict
    Dictionary including the different groups.

Returns
-------
str
    SQL statement.
    """
    result = []
    for g in group_dict:
        L = [quote_ident(elem) for elem in group_dict[g]]
        g_ident = quote_ident(g)
        if len(L) == 1:
            sql_tmp = quote_ident(group_dict[g][0])
            result += [f"{sql_tmp} AS {g_ident}"]
        else:
            sql_tmp = ", ".join(L)
            result += [f"COALESCE({sql_tmp}) AS {g_ident}"]
    return ",\n".join(result)
