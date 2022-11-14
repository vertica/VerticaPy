# (c) Copyright [2018-2022] Micro Focus or one of its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
# VerticaPy is a Python library with scikit-like functionality for conducting
# data science projects on data stored in Vertica, taking advantage Vertica’s
# speed and built-in analytics and machine learning features. It supports the
# entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize
# data transformation operations, and offers beautiful graphical options.
#
# VerticaPy aims to do all of the above. The idea is simple: instead of moving
# data around for processing, VerticaPy brings the logic to the data.
#
#
# Modules
#
# Standard Python Modules
import shutil, re, sys, warnings, random, itertools, datetime, time, html, os
from collections.abc import Iterable
from typing import Union

# VerticaPy Modules
import verticapy
from verticapy.errors import *

# Other Modules
import numpy as np

#
#
# Functions to use to simplify the coding.
#
# ---#
def all_comb(X: list):
    all_configuration = []
    for r in range(len(X) + 1):
        combinations_object = itertools.combinations(X, r)
        combinations_list = list(combinations_object)
        if combinations_list[0]:
            all_configuration += combinations_list
    return all_configuration


# ---#
def arange(start: float, stop: float, step: float):
    check_types(
        [
            ("start", start, [int, float]),
            ("stop", stop, [int, float]),
            ("step", step, [int, float]),
        ]
    )
    if step < 0:
        raise ParameterError("Parameter 'step' must be greater than 0")
    L_final = []
    tmp = start
    while tmp < stop:
        L_final += [tmp]
        tmp += step
    return L_final


# ---#
def bin_spatial_to_str(
    category: str, column: str = "{}",
):
    if category == "vmap":
        return f"MAPTOSTRING({column})"
    elif category == "binary":
        return f"TO_HEX({column})"
    elif category == "spatial":
        return f"ST_AsText({column})"
    else:
        return column


# ---#
def check_types(types_list: list = []):
    for elem in types_list:
        list_check = False
        for sub_elem in elem[2]:
            if not (isinstance(sub_elem, type)):
                list_check = True
        if list_check:
            if not (isinstance(elem[1], str)) and (elem[1] != None):
                warning_message = (
                    "Parameter '{0}' must be of type {1}, found type {2}"
                ).format(elem[0], str, type(elem[1]))
                warnings.warn(warning_message, Warning)
            if (elem[1] != None) and (
                elem[1].lower() not in elem[2] and elem[1] not in elem[2]
            ):
                warning_message = "Parameter '{0}' must be in [{1}], found '{2}'".format(
                    elem[0], "|".join(elem[2]), elem[1]
                )
                warnings.warn(warning_message, Warning)
        else:
            all_types = elem[2] + [type(None)]
            if str in all_types:
                all_types += [str_sql]
            if not (isinstance(elem[1], tuple(all_types))):
                if (
                    (list in elem[2])
                    and isinstance(elem[1], Iterable)
                    and not (isinstance(elem[1], (dict, str)))
                ):
                    pass
                elif len(elem[2]) == 1:
                    warning_message = "Parameter '{0}' must be of type {1}, found type {2}".format(
                        elem[0], elem[2][0], type(elem[1])
                    )
                    warnings.warn(warning_message, Warning)
                else:
                    warning_message = (
                        "Parameter '{0}' type must be one of the following"
                        " {1}, found type {2}"
                    ).format(elem[0], elem[2], type(elem[1]))
                    warnings.warn(warning_message, Warning)


# ---#
def clean_query(query: str):
    res = re.sub(r"--.+(\n|\Z)", "", query)
    res = res.replace("\t", " ").replace("\n", " ")
    res = re.sub(" +", " ", res)

    while len(res) > 0 and (res[-1] in (";", " ")):
        res = res[0:-1]

    while len(res) > 0 and (res[0] in (";", " ")):
        res = res[1:]

    return res


# ---#
def color_dict(d: dict, idx: int = 0):
    if "color" in d:
        if isinstance(d["color"], str):
            return d["color"]
        else:
            return d["color"][idx % len(d["color"])]
    else:
        from verticapy.plot import gen_colors

        return gen_colors()[idx % len(gen_colors())]


# ---#
def find_val_in_dict(x: str, d: dict, return_key: bool = False):
    for elem in d:
        if quote_ident(x).lower() == quote_ident(elem).lower():
            if return_key:
                return elem
            return d[elem]
    raise NameError(f'Key "{x}" was not found in {d}.')


# ---#
def flat_dict(d: dict) -> str:
    # converts dictionary to string with a specific format
    res = []
    for elem in d:
        q = '"' if isinstance(d[elem], str) else ""
        res += ["{}={}{}{}".format(elem, q, d[elem], q)]
    res = ", ".join(res)
    if res:
        res = ", {}".format(res)
    return res


# ---#
def erase_space_start_end_in_list_values(L: list):
    L_tmp = [elem for elem in L]
    for idx in range(len(L_tmp)):
        L_tmp[idx] = L_tmp[idx].strip()
    return L_tmp


# ---#
def executeSQL(
    query: str,
    title: str = "",
    data: list = [],
    method: str = "cursor",
    path: str = "",
    print_time_sql: bool = True,
):
    check_types(
        [
            ("query", query, [str]),
            ("title", title, [str]),
            (
                "method",
                method,
                ["cursor", "fetchrow", "fetchall", "fetchfirstelem", "copy"],
            ),
        ]
    )
    from verticapy.connect import current_cursor

    query = clean_query(query)
    cursor = current_cursor()
    if verticapy.options["sql_on"] and print_time_sql:
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
    if verticapy.options["time_on"] and print_time_sql:
        print_time(elapsed_time)
    if method == "fetchrow":
        return cursor.fetchone()
    elif method == "fetchfirstelem":
        return cursor.fetchone()[0]
    elif method == "fetchall":
        return cursor.fetchall()
    return cursor


# ---#
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


# ---#
def extract_compression(path: str):
    file_extension = path.split(".")[-1].lower()
    lookup_table = {"gz": "GZIP", "bz": "BZIP", "lz": "LZO", "zs": "ZSTD"}
    if file_extension[0:2] in lookup_table:
        return lookup_table[file_extension[0:2]]
    else:
        return "UNCOMPRESSED"


# ---#
def format_magic(x, return_cat: bool = False, cast_float_int_to_str: bool = False):

    from verticapy.vcolumn import vColumn

    if isinstance(x, vColumn):
        val = x.alias
    elif (isinstance(x, (int, float)) and not (cast_float_int_to_str)) or isinstance(
        x, str_sql
    ):
        val = x
    elif isinstance(x, type(None)):
        val = "NULL"
    elif isinstance(x, (int, float)) or not (cast_float_int_to_str):
        val = "'{}'".format(str(x).replace("'", "''"))
    else:
        val = x
    if return_cat:
        return (val, get_category_from_python_type(x))
    else:
        return val


# ---#
def gen_name(L: list):
    return "_".join(
        [
            "".join(ch for ch in str(elem).lower() if ch.isalnum() or ch == "_")
            for elem in L
        ]
    )


# ---#
def gen_tmp_name(schema: str = "", name: str = ""):
    session_user = get_session()
    L = session_user.split("_")
    L[0] = "".join(filter(str.isalnum, L[0]))
    L[1] = "".join(filter(str.isalnum, L[1]))
    random_int = random.randint(0, 10e9)
    name = '"_verticapy_tmp_{}_{}_{}_{}_"'.format(name.lower(), L[0], L[1], random_int)
    if schema:
        name = "{}.{}".format(quote_ident(schema), name)
    return name


# ---#
def get_category_from_python_type(expr):
    try:
        category = expr.category()
    except:
        if isinstance(expr, (float)):
            category = "float"
        elif isinstance(expr, (int)):
            category = "int"
        elif isinstance(expr, (str)):
            category = "text"
        elif isinstance(expr, (datetime.date, datetime.datetime)):
            category = "date"
        elif isinstance(expr, (dict, list, np.ndarray)):
            category = "complex"
        else:
            category = ""
    return category


# ---#
def get_category_from_vertica_type(ctype: str = ""):
    check_types([("ctype", ctype, [str])])
    ctype = ctype.lower()
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
        else:
            return "text"
    else:
        return "undefined"


# ---#
def get_first_file(path: str, ext: str):
    dirname = os.path.dirname(path)
    files = os.listdir(dirname)
    for f in files:
        file_ext = f.split(".")[-1]
        if file_ext == ext:
            return dirname + "/" + f
    return None


# ---#
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


# ---#
def get_header_name_csv(path: str, sep: str):
    f = open(path, "r")
    file_header = f.readline().replace("\n", "").replace('"', "").split(sep)
    f.close()
    for idx, col in enumerate(file_header):
        if col == "":
            if idx == 0:
                position = "beginning"
            elif idx == len(file_header) - 1:
                position = "end"
            else:
                position = "middle"
            file_header[idx] = "col{}".format(idx)
            warning_message = (
                "An inconsistent name was found in the {0} of the "
                "file header (isolated separator). It will be replaced "
                "by col{1}."
            ).format(position, idx)
            if idx == 0:
                warning_message += (
                    "\nThis can happen when exporting a pandas DataFrame "
                    "to CSV while retaining its indexes.\nTip: Use "
                    "index=False when exporting with pandas.DataFrame.to_csv."
                )
            warnings.warn(warning_message, Warning)
    return erase_space_start_end_in_list_values(file_header)


# ---#
def get_match_index(x: str, col_list: list, str_check: bool = True):
    for idx, col in enumerate(col_list):
        if (str_check and quote_ident(x.lower()) == quote_ident(col.lower())) or (
            x == col
        ):
            return idx
    return None


# ---#
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


# ---#
def get_magic_options(line: str):

    # parsing the line
    i, n, splits = 0, len(line), []
    while i < n:
        while i < n and line[i] == " ":
            i += 1
        if i < n:
            k = i
            op = line[i]
            if op in ('"', "'"):
                i += 1
                while i < n - 1:
                    if line[i] == op and line[i + 1] != op:
                        break
                    i += 1
                i += 1
                quote_in = True
            else:
                while i < n and line[i] != " ":
                    i += 1
                quote_in = False
            if quote_in:
                splits += [line[k + 1 : i - 1]]
            else:
                splits += [line[k:i]]

    # Creating the dictionary
    n, i, all_options_dict = len(splits), 0, {}
    while i < n:
        if splits[i][0] != "-":
            raise ParsingError(
                "Can not parse option '{0}'. Options must start with '-'.".format(
                    splits[i][0]
                )
            )
        all_options_dict[splits[i]] = splits[i + 1]
        i += 2

    return all_options_dict


# ---#
def get_random_function(rand_int=None):
    random_state = verticapy.options["random_state"]
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


# ---#
def get_session(add_username: bool = True):
    query = "SELECT /*+LABEL(get_session)*/ CURRENT_SESSION();"
    result = executeSQL(query, method="fetchfirstelem", print_time_sql=False)
    result = result.split(":")[1]
    result = int(result, base=16)
    if add_username:
        query = "SELECT /*+LABEL(get_session)*/ USERNAME();"
        result = "{}_{}".format(
            executeSQL(query, method="fetchfirstelem", print_time_sql=False), result
        )
    return result


# ---#
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
        dtype = dtype.lower()
    return dtype


# ---#
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
            key = "{}%".format(int(float(key[start:-1])))
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


# ---#
def heuristic_length(i):
    GAMMA = 0.5772156649
    if i == 2:
        return 1
    elif i > 2:
        return 2 * (np.log(i - 1) + GAMMA) - 2 * (i - 1) / i
    else:
        return 0


# ---#
def indentSQL(query: str):
    query = (
        query.replace("SELECT", "\n   SELECT\n    ")
        .replace("FROM", "\n   FROM\n")
        .replace(",", ",\n    ")
    )
    query = query.replace("VERTICAPY_SUBTABLE", "\nVERTICAPY_SUBTABLE")
    n = len(query)
    return_l = []
    j = 1
    while j < n - 9:
        if (
            query[j] == "("
            and (query[j - 1].isalnum() or query[j - 5 : j] == "OVER ")
            and query[j + 1 : j + 7] != "SELECT"
        ):
            k = 1
            while k > 0 and j < n - 9:
                j += 1
                if query[j] == "\n":
                    return_l += [j]
                elif query[j] == ")":
                    k -= 1
                elif query[j] == "(":
                    k += 1
        else:
            j += 1
    query_print = ""
    i = 0 if query[0] != "\n" else 1
    while return_l:
        j = return_l[0]
        query_print += query[i:j]
        if query[j] != "\n":
            query_print += query[j]
        else:
            i = j + 1
            while query[i] == " " and i < n - 9:
                i += 1
            query_print += " "
        del return_l[0]
    query_print += query[i:n]
    return query_print


# ---#
def insert_verticapy_schema(
    model_name: str,
    model_type: str,
    model_save: dict,
    category: str = "VERTICAPY_MODELS",
):
    sql = "SELECT /*+LABEL(insert_verticapy_schema)*/ * FROM columns WHERE table_schema='verticapy';"
    result = executeSQL(sql, method="fetchrow", print_time_sql=False)
    if not (result):
        warning_message = (
            "The VerticaPy schema doesn't exist or is "
            "incomplete. The model can not be stored.\n"
            "Please use create_verticapy_schema function "
            "to set up the schema and the drop function to "
            "drop it if it is corrupted."
        )
        warnings.warn(warning_message, Warning)
    else:
        size = sys.getsizeof(model_save)
        create_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        try:
            model_name = quote_ident(model_name)
            sql = "SELECT /*+LABEL(insert_verticapy_schema)*/ * FROM verticapy.models WHERE LOWER(model_name) = '{}'".format(
                model_name.lower()
            )
            result = executeSQL(sql, method="fetchrow", print_time_sql=False)
            if result:
                raise NameError("The model named {} already exists.".format(model_name))
            else:
                sql = (
                    "INSERT /*+LABEL(insert_verticapy_schema)*/ INTO verticapy.models(model_name, category, "
                    "model_type, create_time, size) VALUES ('{}', '{}', '{}', "
                    "'{}', {});"
                ).format(model_name, category, model_type, create_time, size)
                executeSQL(sql, print_time_sql=False)
                executeSQL("COMMIT;", print_time_sql=False)
                for elem in model_save:
                    sql = (
                        "INSERT /*+LABEL(insert_verticapy_schema)*/ INTO verticapy.attr(model_name, attr_name, value) "
                        "VALUES ('{0}', '{1}', '{2}');"
                    ).format(model_name, elem, str(model_save[elem]).replace("'", "''"))
                    executeSQL(sql, print_time_sql=False)
                    executeSQL("COMMIT;", print_time_sql=False)
        except Exception as e:
            warning_message = "The VerticaPy model could not be stored:\n{}".format(e)
            warnings.warn(warning_message, Warning)
            raise


# ---#
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


# ---#
def levenshtein(s: str, t: str):
    rows = len(s) + 1
    cols = len(t) + 1
    dist = [[0 for x in range(cols)] for x in range(rows)]
    for i in range(1, rows):
        dist[i][0] = i
    for i in range(1, cols):
        dist[0][i] = i
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(
                dist[row - 1][col] + 1,
                dist[row][col - 1] + 1,
                dist[row - 1][col - 1] + cost,
            )
    return dist[row][col]


# ---#
def print_query(query: str, title: str = ""):
    screen_columns = shutil.get_terminal_size().columns
    query_print = indentSQL(query)
    if isnotebook():
        from IPython.core.display import HTML, display

        display(HTML("<h4>{}</h4>".format(title)))
        query_print = query_print.replace("\n", " <br>").replace("  ", " &emsp; ")
        display(HTML(query_print))
    else:
        print("$ {} $\n".format(title))
        print(query_print)
        print("-" * int(screen_columns) + "\n")


# ---#
def print_table(
    data_columns,
    is_finished: bool = True,
    offset: int = 0,
    repeat_first_column: bool = False,
    first_element: str = "",
    return_html: bool = False,
    dtype: dict = {},
    percent: dict = {},
):
    if not (return_html):
        data_columns_rep = [] + data_columns
        if repeat_first_column:
            del data_columns_rep[0]
            columns_ljust_val = min(
                len(max([str(item) for item in data_columns[0]], key=len)) + 4, 40
            )
        else:
            columns_ljust_val = len(str(len(data_columns[0]))) + 2
        screen_columns = shutil.get_terminal_size().columns
        formatted_text = ""
        rjust_val = []
        for idx in range(0, len(data_columns_rep)):
            rjust_val += [
                min(
                    len(max([str(item) for item in data_columns_rep[idx]], key=len))
                    + 2,
                    40,
                )
            ]
        total_column_len = len(data_columns_rep[0])
        while rjust_val != []:
            columns_to_print = [data_columns_rep[0]]
            columns_rjust_val = [rjust_val[0]]
            max_screen_size = int(screen_columns) - 14 - int(rjust_val[0])
            del data_columns_rep[0]
            del rjust_val[0]
            while (max_screen_size > 0) and (rjust_val != []):
                columns_to_print += [data_columns_rep[0]]
                columns_rjust_val += [rjust_val[0]]
                max_screen_size = max_screen_size - 7 - int(rjust_val[0])
                del data_columns_rep[0]
                del rjust_val[0]
            if repeat_first_column:
                columns_to_print = [data_columns[0]] + columns_to_print
            else:
                columns_to_print = [
                    [i + offset for i in range(0, total_column_len)]
                ] + columns_to_print
            columns_to_print[0][0] = first_element
            columns_rjust_val = [columns_ljust_val] + columns_rjust_val
            column_count = len(columns_to_print)
            for i in range(0, total_column_len):
                for k in range(0, column_count):
                    val = columns_to_print[k][i]
                    if len(str(val)) > 40:
                        val = str(val)[0:37] + "..."
                    if k == 0:
                        formatted_text += str(val).ljust(columns_rjust_val[k])
                    else:
                        formatted_text += str(val).rjust(columns_rjust_val[k]) + "  "
                if rjust_val != []:
                    formatted_text += " \\\\"
                formatted_text += "\n"
            if not (is_finished) and (i == total_column_len - 1):
                for k in range(0, column_count):
                    if k == 0:
                        formatted_text += "...".ljust(columns_rjust_val[k])
                    else:
                        formatted_text += "...".rjust(columns_rjust_val[k]) + "  "
                if rjust_val != []:
                    formatted_text += " \\\\"
                formatted_text += "\n"
        return formatted_text
    else:
        if not (repeat_first_column):
            data_columns = [
                [""] + list(range(1 + offset, len(data_columns[0]) + offset))
            ] + data_columns
        m, n = len(data_columns), len(data_columns[0])
        cell_width = []
        for elem in data_columns:
            cell_width += [min(5 * max([len(str(item)) for item in elem]) + 80, 280)]
        html_table = "<table>"
        for i in range(n):
            if i == 0:
                html_table += '<thead style = "display: table; ">'
            if i == 1 and n > 0:
                html_table += (
                    '<tbody style = "display: block; max-height: '
                    '300px; overflow-y: scroll;">'
                )
            html_table += "<tr>"
            for j in range(m):
                val = data_columns[j][i]
                if isinstance(val, str):
                    val = html.escape(val)
                if val == None:
                    val = "[null]"
                    color = "#999999"
                else:
                    if isinstance(val, bool) and (
                        verticapy.options["mode"] in ("full", None)
                    ):
                        val = (
                            "<center>&#9989;</center>"
                            if (val)
                            else "<center>&#10060;</center>"
                        )
                    color = "black"
                html_table += '<td style="background-color: '
                if (
                    (j == 0)
                    or (i == 0)
                    or (verticapy.options["mode"] not in ("full", None))
                ):
                    html_table += " #FFFFFF; "
                elif val == "[null]":
                    html_table += " #EEEEEE; "
                else:
                    html_table += " #FAFAFA; "
                html_table += "color: {}; white-space:nowrap; ".format(color)
                if verticapy.options["mode"] in ("full", None):
                    if (j == 0) or (i == 0):
                        html_table += "border: 1px solid #AAAAAA; "
                    else:
                        html_table += "border-top: 1px solid #DDDDDD; "
                        if ((j == m - 1) and (i == n - 1)) or (j == m - 1):
                            html_table += "border-right: 1px solid #AAAAAA; "
                        else:
                            html_table += "border-right: 1px solid #DDDDDD; "
                        if ((j == m - 1) and (i == n - 1)) or (i == n - 1):
                            html_table += "border-bottom: 1px solid #AAAAAA; "
                        else:
                            html_table += "border-bottom: 1px solid #DDDDDD; "
                if i == 0:
                    html_table += "height: 30px; "
                if (j == 0) or (cell_width[j] < 120):
                    html_table += "text-align: center; "
                else:
                    html_table += "text-align: center; "
                html_table += 'min-width: {}px; max-width: {}px;"'.format(
                    cell_width[j], cell_width[j]
                )
                if (j == 0) or (i == 0):
                    if j != 0:
                        type_val, category, missing_values = "", "", ""
                        if data_columns[j][0] in dtype and (
                            verticapy.options["mode"] in ("full", None)
                        ):
                            if dtype[data_columns[j][0]] != "undefined":
                                type_val = dtype[data_columns[j][0]].capitalize()
                                category = get_category_from_vertica_type(type_val)
                                if (category == "spatial") or (
                                    (
                                        "lat" in val.lower().split(" ")
                                        or "latitude" in val.lower().split(" ")
                                        or "lon" in val.lower().split(" ")
                                        or "longitude" in val.lower().split(" ")
                                    )
                                    and category == "float"
                                ):
                                    category = '<div style="margin-bottom: 6px;">&#x1f30e;</div>'
                                elif type_val.lower() == "boolean":
                                    category = '<div style="margin-bottom: 6px; color: #0073E7;">010</div>'
                                elif category in ("int", "float", "binary", "uuid"):
                                    category = '<div style="margin-bottom: 6px; color: #19A26B;">123</div>'
                                elif category == "text":
                                    category = (
                                        '<div style="margin-bottom: 6px;">Abc</div>'
                                    )
                                elif category in ("complex", "vmap"):
                                    category = '<div style="margin-bottom: 6px;">&#128736;</div>'
                                elif category == "date":
                                    category = '<div style="margin-bottom: 6px;">&#128197;</div>'
                            else:
                                category = '<div style="margin-bottom: 6px;"></div>'
                        if type_val != "":
                            ctype = (
                                '<div style="overflow-y: scroll; color: #FE5016; '
                                f'margin-top: 6px; font-size: 0.95em;">{type_val}</div>'
                            )
                        else:
                            ctype = '<div style="color: #FE5016; margin-top: 6px; font-size: 0.95em;"></div>'
                        if data_columns[j][0] in percent:
                            per = int(float(percent[data_columns[j][0]]))
                            try:
                                if per == 100:
                                    diff = 36
                                elif per > 10:
                                    diff = 30
                                else:
                                    diff = 24
                            except:
                                pass
                            missing_values = (
                                '<div style="float: right; margin-top: 6px;">{0}%</div><div '
                                'style="width: calc(100% - {1}px); height: 8px; margin-top: '
                                '10px; border: 1px solid black;"><div style="width: {0}%; '
                                'height: 6px; background-color: orange;"></div></div>'
                            ).format(per, diff)
                    else:
                        ctype, missing_values, category = "", "", ""
                    if (i == 0) and (j == 0):
                        if dtype and (verticapy.options["mode"] in ("full", None)):
                            val = verticapy.gen_verticapy_logo_html(size="45px")
                        else:
                            val = ""
                    elif cell_width[j] > 240:
                        val = (
                            '<input style="border: none; text-align: center; width: {0}px;" '
                            'type="text" value="{1}" readonly>'
                        ).format(cell_width[j] - 10, val)
                    html_table += ">{}<b>{}</b>{}{}</td>".format(
                        category, val, ctype, missing_values
                    )
                elif cell_width[j] > 240:
                    background = "#EEEEEE" if val == "[null]" else "#FAFAFA"
                    if verticapy.options["mode"] not in ("full", None):
                        background = "#FFFFFF"
                    html_table += (
                        '><input style="background-color: {0}; border: none; '
                        'text-align: center; width: {1}px;" type="text" '
                        'value="{2}" readonly></td>'
                    ).format(background, cell_width[j] - 10, val)
                else:
                    html_table += ">{}</td>".format(val)
            html_table += "</tr>"
            if i == 0:
                html_table += "</thead>"
            if i == n - 1 and n > 0:
                html_table += "</tbody>"
        html_table += "</table>"
        return html_table


# ---#
def print_time(elapsed_time: float):
    screen_columns = shutil.get_terminal_size().columns
    if isnotebook():
        from IPython.core.display import HTML, display

        display(
            HTML("<div><b>Execution: </b> {0}s</div>".format(round(elapsed_time, 3)))
        )
    else:
        print("Execution: {0}s".format(round(elapsed_time, 3)))
        print("-" * int(screen_columns) + "\n")


# ---#
def quote_ident(column: str):
    """
    ---------------------------------------------------------------------------
    Returns the specified string argument in the format that is required in
    order to use that string as an identifier in an SQL statement.

    Parameters
    ----------
    column: str
        Column's name.

    Returns
    -------
    str
        Formatted column' name.
    """
    tmp_column = str(column)
    if len(tmp_column) >= 2 and (tmp_column[0] == tmp_column[-1] == '"'):
        tmp_column = tmp_column[1:-1]
    return '"{}"'.format(str(tmp_column).replace('"', '""'))


# ---#
def replace_vars_in_query(query: str, locals_dict: dict):
    from verticapy import vDataFrame, tablesample, pandas_to_vertica
    import pandas as pd

    variables, query_tmp = re.findall("(?<!:):[A-Za-z0-9_\[\]]+", query), query
    for v in variables:
        fail = True
        if len(v) > 1 and not (v[1].isdigit()):
            try:
                var = v[1:]
                n, splits = var.count("["), []
                if var.count("]") == n and n > 0:
                    i, size = 0, len(var)
                    while i < size:
                        if var[i] == "[":
                            k = i + 1
                            while i < size and var[i] != "]":
                                i += 1
                            splits += [(k, i)]
                        i += 1
                    var = var[: splits[0][0] - 1]
                val = locals_dict[var]
                if splits:
                    for s in splits:
                        val = val[int(v[s[0] + 1 : s[1] + 1])]
                fail = False
            except Exception as e:
                warning_message = "Failed to replace variables in the query.\nError: {0}".format(
                    e
                )
                warnings.warn(warning_message, Warning)
                fail = True
        if not (fail):
            if isinstance(val, vDataFrame):
                val = val.__genSQL__()
            elif isinstance(val, tablesample):
                val = "({0}) VERTICAPY_SUBTABLE".format(val.to_sql())
            elif isinstance(val, pd.DataFrame):
                val = pandas_to_vertica(val).__genSQL__()
            elif isinstance(val, list):
                val = ", ".join(["NULL" if elem is None else str(elem) for elem in val])
            query_tmp = query_tmp.replace(v, str(val))
    return query_tmp


# ---#
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


# ---#
def schema_relation(relation):
    from verticapy import vDataFrame

    if isinstance(relation, vDataFrame):
        schema, relation = verticapy.options["temp_schema"], ""
    else:
        quote_nb = relation.count('"')
        if quote_nb not in (0, 2, 4):
            raise ParsingError("The format of the input relation is incorrect.")
        if quote_nb == 4:
            schema_input_relation = relation.split('"')[1], relation.split('"')[3]
        elif quote_nb == 4:
            schema_input_relation = (
                relation.split('"')[1],
                relation.split('"')[2][1:]
                if (relation.split('"')[0] == "")
                else relation.split('"')[0][0:-1],
                relation.split('"')[1],
            )
        else:
            schema_input_relation = relation.split(".")
        if len(schema_input_relation) == 1:
            schema, relation = "public", relation
        else:
            schema, relation = schema_input_relation[0], schema_input_relation[1]
    return (quote_ident(schema), quote_ident(relation))


# ---#
def format_schema_table(schema: str, table_name: str):
    if not (schema):
        schema = "public"
    return quote_ident(schema) + "." + quote_ident(table_name)


# ---#
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


# ---#
class str_sql:
    # ---#
    def __init__(self, alias, category="", init_transf=""):
        self.alias = alias
        self.category_ = category
        if not (init_transf):
            self.init_transf = alias
        else:
            self.init_transf = init_transf

    # ---#
    def __repr__(self):
        return str(self.init_transf)

    # ---#
    def __str__(self):
        return str(self.init_transf)

    # ---#
    def __abs__(self):
        return str_sql("ABS({})".format(self.init_transf), self.category())

    # ---#
    def __add__(self, x):
        from verticapy.vcolumn import vColumn

        if (isinstance(self, vColumn) and self.isarray()) and (
            isinstance(x, vColumn) and x.isarray()
        ):
            return str_sql(
                "ARRAY_CAT({}, {})".format(self.init_transf, x.init_transf), "complex"
            )
        val = format_magic(x)
        op = (
            "||" if self.category() == "text" and isinstance(x, (str, str_sql)) else "+"
        )
        return str_sql(
            "({}) {} ({})".format(self.init_transf, op, val), self.category()
        )

    # ---#
    def __radd__(self, x):
        from verticapy.vcolumn import vColumn

        if (isinstance(self, vColumn) and self.isarray()) and (
            isinstance(x, vColumn) and x.isarray()
        ):
            return str_sql(
                "ARRAY_CAT({}, {})".format(x.init_transf, self.init_transf), "complex"
            )
        val = format_magic(x)
        op = (
            "||" if self.category() == "text" and isinstance(x, (str, str_sql)) else "+"
        )
        return str_sql(
            "({}) {} ({})".format(val, op, self.init_transf), self.category()
        )

    # ---#
    def __and__(self, x):
        val = format_magic(x)
        return str_sql("({}) AND ({})".format(self.init_transf, val), self.category())

    # ---#
    def __rand__(self, x):
        val = format_magic(x)
        return str_sql("({}) AND ({})".format(val, self.init_transf), self.category())

    # ---#
    def _between(self, x, y):
        val1 = str(format_magic(x))
        val2 = str(format_magic(y))
        return str_sql(
            "({}) BETWEEN ({}) AND ({})".format(self.init_transf, val1, val2),
            self.category(),
        )

    # ---#
    def _in(self, *argv):
        if (len(argv) == 1) and (isinstance(argv[0], list)):
            x = argv[0]
        elif len(argv) == 0:
            ParameterError("Method 'in_' doesn't work with no parameters.")
        else:
            x = [elem for elem in argv]
        assert isinstance(x, Iterable) and not (
            isinstance(x, str)
        ), "Method '_in' only works on iterable elements other than str. Found {}.".format(
            x
        )
        val = [str(format_magic(elem)) for elem in x]
        val = ", ".join(val)
        return str_sql("({}) IN ({})".format(self.init_transf, val), self.category())

    # ---#
    def _not_in(self, *argv):
        if (len(argv) == 1) and (isinstance(argv[0], list)):
            x = argv[0]
        elif len(argv) == 0:
            ParameterError("Method '_not_in' doesn't work with no parameters.")
        else:
            x = [elem for elem in argv]
        assert isinstance(x, Iterable) and not (
            isinstance(x, str)
        ), "Method '_not_in' only works on iterable elements other than str. Found {}.".format(
            x
        )
        val = [str(format_magic(elem)) for elem in x]
        val = ", ".join(val)
        return str_sql(
            "({}) NOT IN ({})".format(self.init_transf, val), self.category()
        )

    # ---#
    def _as(self, x):
        return str_sql("({}) AS {}".format(self.init_transf, x), self.category())

    # ---#
    def _distinct(self):
        return str_sql("DISTINCT ({})".format(self.init_transf), self.category())

    # ---#
    def _over(self, by: (str, list) = [], order_by: (str, list) = []):
        if isinstance(by, str):
            by = [by]
        if isinstance(order_by, str):
            order_by = [order_by]
        by = ", ".join([str(elem) for elem in by])
        if by:
            by = "PARTITION BY {}".format(by)
        order_by = ", ".join([str(elem) for elem in order_by])
        if order_by:
            order_by = "ORDER BY {}".format(order_by)
        return str_sql(
            "{} OVER ({} {})".format(self.init_transf, by, order_by), self.category()
        )

    # ---#
    def __eq__(self, x):
        op = "IS" if (x == None) and not (isinstance(x, str_sql)) else "="
        val = format_magic(x)
        if val != "NULL":
            val = "({})".format(val)
        return str_sql("({}) {} {}".format(self.init_transf, op, val), self.category())

    # ---#
    def __ne__(self, x):
        op = "IS NOT" if (x == None) and not (isinstance(x, str_sql)) else "!="
        val = format_magic(x)
        if val != "NULL":
            val = "({})".format(val)
        return str_sql("({}) {} {}".format(self.init_transf, op, val), self.category())

    # ---#
    def __ge__(self, x):
        val = format_magic(x)
        return str_sql("({}) >= ({})".format(self.init_transf, val), self.category())

    # ---#
    def __gt__(self, x):
        val = format_magic(x)
        return str_sql("({}) > ({})".format(self.init_transf, val), self.category())

    # ---#
    def __le__(self, x):
        val = format_magic(x)
        return str_sql("({}) <= ({})".format(self.init_transf, val), self.category())

    # ---#
    def __lt__(self, x):
        val = format_magic(x)
        return str_sql("({}) < ({})".format(self.init_transf, val), self.category())

    # ---#
    def __mul__(self, x):
        if self.category() == "text" and isinstance(x, (int)):
            return str_sql(
                "REPEAT({}, {})".format(self.init_transf, x), self.category()
            )
        val = format_magic(x)
        return str_sql("({}) * ({})".format(self.init_transf, val), self.category())

    # ---#
    def __rmul__(self, x):
        if self.category() == "text" and isinstance(x, (int)):
            return str_sql(
                "REPEAT({}, {})".format(self.init_transf, x), self.category()
            )
        val = format_magic(x)
        return str_sql("({}) * ({})".format(val, self.init_transf), self.category())

    # ---#
    def __or__(self, x):
        val = format_magic(x)
        return str_sql("({}) OR ({})".format(self.init_transf, val), self.category())

    # ---#
    def __ror__(self, x):
        val = format_magic(x)
        return str_sql("({}) OR ({})".format(val, self.init_transf), self.category())

    # ---#
    def __pos__(self):
        return str_sql("+({})".format(self.init_transf), self.category())

    # ---#
    def __neg__(self):
        return str_sql("-({})".format(self.init_transf), self.category())

    # ---#
    def __pow__(self, x):
        val = format_magic(x)
        return str_sql("POWER({}, {})".format(self.init_transf, val), self.category())

    # ---#
    def __rpow__(self, x):
        val = format_magic(x)
        return str_sql("POWER({}, {})".format(val, self.init_transf), self.category())

    # ---#
    def __mod__(self, x):
        val = format_magic(x)
        return str_sql("MOD({}, {})".format(self.init_transf, val), self.category())

    # ---#
    def __rmod__(self, x):
        val = format_magic(x)
        return str_sql("MOD({}, {})".format(val, self.init_transf), self.category())

    # ---#
    def __sub__(self, x):
        val = format_magic(x)
        return str_sql("({}) - ({})".format(self.init_transf, val), self.category())

    # ---#
    def __rsub__(self, x):
        val = format_magic(x)
        return str_sql("({}) - ({})".format(val, self.init_transf), self.category())

    # ---#
    def __truediv__(self, x):
        val = format_magic(x)
        return str_sql("({}) / ({})".format(self.init_transf, val), self.category())

    # ---#
    def __rtruediv__(self, x):
        val = format_magic(x)
        return str_sql("({}) / ({})".format(val, self.init_transf), self.category())

    # ---#
    def __floordiv__(self, x):
        val = format_magic(x)
        return str_sql("({}) // ({})".format(self.init_transf, val), self.category())

    # ---#
    def __rfloordiv__(self, x):
        val = format_magic(x)
        return str_sql("({}) // ({})".format(val, self.init_transf), self.category())

    # ---#
    def __ceil__(self):
        return str_sql("CEIL({})".format(self.init_transf), self.category())

    # ---#
    def __floor__(self):
        return str_sql("FLOOR({})".format(self.init_transf), self.category())

    # ---#
    def __trunc__(self):
        return str_sql("TRUNC({})".format(self.init_transf), self.category())

    # ---#
    def __invert__(self):
        return str_sql("-({}) - 1".format(self.init_transf), self.category())

    # ---#
    def __round__(self, x):
        return str_sql("ROUND({}, {})".format(self.init_transf, x), self.category())

    def category(self):
        return self.category_
