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
import re
import warnings
from typing import Any, Iterable, Literal, Optional

import numpy as np

import pandas as pd

import verticapy._config.config as conf
from verticapy._utils._object import read_pd
from verticapy._utils._sql._cast import to_dtype_category
from verticapy._typing import NoneType, SQLColumns, SQLExpression
from verticapy.errors import ParsingError


def clean_query(query: SQLExpression) -> SQLExpression:
    """
    Cleans the input query by erasing comments, spaces,
    and other unnecessary characters.
    Comments using '/*' and '*/' are left in the query.
    """
    if isinstance(query, list):
        return [clean_query(q) for q in query]
    else:
        query = re.sub(r"--.+(\n|\Z)", "", query)
        query = query.replace("\t", " ").replace("\n", " ")
        query = re.sub(" +", " ", query)

        while len(query) > 0 and query.endswith((";", " ")):
            query = query[0:-1]

        while len(query) > 0 and query.startswith((";", " ")):
            query = query[1:]

        return query.strip().replace("\xa0", " ")


def erase_comment(query: str) -> str:
    """
    Removes comments from the input query.
    """
    query = re.sub(r"--.+(\n|\Z)", "", query)
    query = re.sub(r"/\*(.+?)\*/", "", query)
    return query.strip()


def erase_label(query: str) -> str:
    """
    Removes labels from the input query.
    """
    labels = re.findall(r"\/\*\+LABEL(.*?)\*\/", query)
    for label in labels:
        query = query.replace(f"/*+LABEL{label}*/", "")
    return query.strip()


def extract_subquery(query: str) -> str:
    """
    Extracts the SQL subquery from the input query.
    """
    query_tmp = clean_query(query)
    query_tmp = erase_comment(query_tmp)
    if query_tmp[0] == "(" and query_tmp[-1] != ")":
        query = ")".join("(".join(query_tmp.split("(")[1:]).split(")")[:-1])
    return query.strip()


def extract_and_rename_subquery(query: str, alias: str) -> str:
    """
    Extracts the SQL subquery from the input query
    and renames it.
    """
    query_tmp = extract_subquery(query)
    query_clean = clean_query(query)
    query_clean = erase_comment(query)
    if query != query_tmp or query_clean[0:6].lower() == "select":
        query = f"({query_tmp})"
    return f"{query} AS {quote_ident(alias)}"


def extract_precision_scale(ctype: str) -> tuple:
    """
    Extracts the precision and scale from the
    input SQL type.
    """
    if "(" not in ctype:
        return (0, 0)
    else:
        precision_scale = ctype.split("(")[1].split(")")[0].split(",")
        if len(precision_scale) == 1:
            precision, scale = int(precision_scale[0]), 0
        else:
            precision, scale = precision_scale
        return int(precision), int(scale)


def format_magic(
    x: Any, return_cat: bool = False, cast_float_int_to_str: bool = False
) -> Any:
    """
    Formats  the input element using SQL rules.
    Ex: None values are represented by NULL and
        string are enclosed by single quotes "'"
    """
    object_type = None
    if hasattr(x, "object_type"):
        object_type = x.object_type
    if object_type == "vDataColumn":
        val = x._alias
    elif (isinstance(x, (int, float, np.int_)) and not cast_float_int_to_str) or (
        object_type == "StringSQL"
    ):
        val = x
    elif isinstance(x, NoneType):
        val = "NULL"
    elif isinstance(x, (int, float, np.int_)) or not cast_float_int_to_str:
        x_str = str(x).replace("'", "''")
        val = f"'{x_str}'"
    else:
        val = x
    if return_cat:
        return (val, to_dtype_category(x))
    else:
        return val


def format_type(*args, dtype: Literal[NoneType, dict, list], na_out: Any = None) -> Any:
    """
    Format the input objects by using the input type. This
    simplifies the code because many functions check
    types and instantiate the corresponding object.
    """
    res = ()
    for arg in args:
        if isinstance(arg, NoneType):
            if not isinstance(na_out, NoneType):
                r = na_out
            elif dtype == list:
                r = []
            elif dtype == dict:
                r = {}
            else:
                r = None
        elif isinstance(arg, (float, int, str)):
            if dtype == list:
                r = [arg]
            else:
                r = arg
        else:
            r = arg
        res += (r,)
    if len(res) == 1:
        return res[0]
    else:
        return res


def indentSQL(query: str) -> str:
    """
    Indents the input SQL query.
    """
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


def list_strip(L: list) -> list:
    """
    Erases all the start / end spaces from the
    input list.
    """
    return [val.strip() for val in L]


def quote_ident(column: Optional[SQLColumns], lower: bool = False) -> SQLColumns:
    """
    Returns the specified string argument in the format
    that is required in order to use that string as an
    identifier in an SQL statement.

    Parameters
    ----------
    column: str
        Column's name.

    Returns
    -------
    str
        Formatted column name.
    """
    if isinstance(column, str):
        tmp_column = str(column)
        if len(tmp_column) >= 2 and (tmp_column[0] == tmp_column[-1] == '"'):
            tmp_column = tmp_column[1:-1]
        temp_column_str = str(tmp_column).replace('"', '""')
        temp_column_str = f'"{temp_column_str}"'
        if temp_column_str == '""':
            return ""
        elif lower:
            temp_column_str = temp_column_str.lower()
        return temp_column_str
    elif isinstance(column, NoneType):
        return ""
    elif isinstance(column, Iterable):
        return [quote_ident(x) for x in column]
    else:
        return str(column)


def replace_vars_in_query(query: str, locals_dict: dict) -> str:
    """
    Replaces the input variables with their respective SQL
    representations. If a input variable does not have a
    SQL representation, it is materialised by a temporary
    local table.
    """
    variables, query_tmp = re.findall(r"(?<!:):[A-Za-z0-9_\[\]]+", query), query
    for v in variables:
        fail = True
        if len(v) > 1 and not v[1].isdigit():
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
                warning_message = (
                    f"Failed to replace variables in the query.\nError: {e}"
                )
                warnings.warn(warning_message, Warning)
                fail = True
        if not fail:
            object_type = None
            if hasattr(val, "object_type"):
                object_type = val.object_type
            if object_type == "vDataFrame":
                val = val.current_relation()
            elif object_type == "TableSample":
                val = f"({val.to_sql()}) VERTICAPY_SUBTABLE"
            elif isinstance(val, pd.DataFrame):
                val = read_pd(val).current_relation()
            elif isinstance(val, list):
                val = ", ".join(["NULL" if x is None else str(x) for x in val])
            query_tmp = query_tmp.replace(v, str(val))
    return query_tmp


def schema_relation(relation: Any) -> tuple[str, str]:
    """
    Extracts the schema and the table from the input
    relation. If the input relation does not have a schema,
    the temporary schema is used.
    """
    if isinstance(relation, str):
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
    else:
        schema, relation = conf.get_option("temp_schema"), ""
    return (quote_ident(schema), quote_ident(relation))


def format_schema_table(schema: str, table_name: str) -> str:
    """
    Returns the formatted relation. If the schema is not
    defined, the 'public' schema is used.
    """
    if not schema:
        schema = "public"
    return f"{quote_ident(schema)}.{quote_ident(table_name)}"
