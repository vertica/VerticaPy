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
import time
from typing import Union, Literal
from verticapy.utils._decorators import save_verticapy_logs
from verticapy.core.str_sql import str_sql


def _executeSQL(
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
    import verticapy as vp
    from verticapy.sdk.vertica.dblink import replace_external_queries_in_query
    from verticapy.sql._utils._format import clean_query, erase_label
    from verticapy.sql._utils._display import print_query

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


@save_verticapy_logs
def readSQL(query: str, time_on: bool = False, limit: int = 100):
    """
    Returns the result of a SQL query as a tablesample object.

    Parameters
    ----------
    query: str
        SQL Query.
    time_on: bool, optional
        If set to True, displays the query elapsed time.
    limit: int, optional
        Maximum number of elements to display.

    Returns
    -------
    tablesample
        Result of the query.
    """
    from verticapy.sql.read import _executeSQL
    import verticapy as vp

    while len(query) > 0 and query[-1] in (";", " "):
        query = query[:-1]
    if vp.OPTIONS["count_on"]:
        count = _executeSQL(
            f"""SELECT 
                    /*+LABEL('utilities.readSQL')*/ COUNT(*) 
                FROM ({query}) VERTICAPY_SUBTABLE""",
            method="fetchfirstelem",
            print_time_sql=False,
        )
    else:
        count = -1
    sql_on_init = vp.OPTIONS["sql_on"]
    time_on_init = vp.OPTIONS["time_on"]
    try:
        vp.OPTIONS["time_on"] = time_on
        vp.OPTIONS["sql_on"] = False
        try:
            result = to_tablesample(f"{query} LIMIT {limit}")
        except:
            result = to_tablesample(query)
    finally:
        vp.OPTIONS["time_on"] = time_on_init
        vp.OPTIONS["sql_on"] = sql_on_init
    result.count = count
    if vp.OPTIONS["percent_bar"]:
        vdf = vDataFrameSQL(f"({query}) VERTICAPY_SUBTABLE")
        percent = vdf.agg(["percent"]).transpose().values
        for column in result.values:
            result.dtype[column] = vdf[column].ctype()
            result.percent[column] = percent[vdf.format_colnames(column)][0]
    return result


def to_tablesample(
    query: Union[str, str_sql],
    title: str = "",
    max_columns: int = -1,
    sql_push_ext: bool = False,
    symbol: str = "$",
):
    """
    Returns the result of a SQL query as a tablesample object.

    Parameters
    ----------
    query: str, optional
        SQL Query.
    title: str, optional
        Query title when the query is displayed.
    max_columns: int, optional
        Maximum number of columns to display.
    sql_push_ext: bool, optional
        If set to True, the entire query is pushed to the external table. 
        This can increase performance but might increase the error rate. 
        For instance, some DBs might not support the same SQL as Vertica.
    symbol: str, optional
        One of the following:
        "$", "€", "£", "%", "@", "&", "§", "%", "?", "!"
        Symbol used to identify the external connection.
        See the connect.set_external_connection function for more information.

    Returns
    -------
    tablesample
        Result of the query.

    See Also
    --------
    tablesample : Object in memory created for rendering purposes.
    """
    import verticapy as vp
    from verticapy.core.tablesample import tablesample
    from verticapy.sql.read import _executeSQL
    from verticapy.sql.dtypes import vertica_python_dtype
    from verticapy.sql._utils._display import print_query, print_time

    if vp.OPTIONS["sql_on"]:
        print_query(query, title)
    start_time = time.time()
    cursor = _executeSQL(
        query, print_time_sql=False, sql_push_ext=sql_push_ext, symbol=symbol
    )
    description, dtype = cursor.description, {}
    for elem in description:
        dtype[elem[0]] = vertica_python_dtype(
            type_name=elem.type_name,
            display_size=elem[2],
            precision=elem[4],
            scale=elem[5],
        )
    elapsed_time = time.time() - start_time
    if vp.OPTIONS["time_on"]:
        print_time(elapsed_time)
    result = cursor.fetchall()
    columns = [column[0] for column in cursor.description]
    data_columns = [[item] for item in columns]
    data = [item for item in result]
    for row in data:
        for idx, val in enumerate(row):
            data_columns[idx] += [val]
    values = {}
    for column in data_columns:
        values[column[0]] = column[1 : len(column)]
    return tablesample(
        values=values, dtype=dtype, max_columns=max_columns,
    ).decimal_to_float()


def vDataFrameSQL(
    relation: str,
    name: str = "VDF",
    schema: str = "public",
    history: list = [],
    saving: list = [],
    vdf=None,
):
    """
Creates a vDataFrame based on a customized relation.

Parameters
----------
relation: str
    Relation. You can also specify a customized relation, 
    but you must enclose it with an alias. For example "(SELECT 1) x" is 
    correct whereas "(SELECT 1)" and "SELECT 1" are incorrect.
name: str, optional
    Name of the vDataFrame. It is used only when displaying the vDataFrame.
schema: str, optional
    Relation schema. It can be to use to be less ambiguous and allow to 
    create schema and relation name with dots '.' inside.
history: list, optional
    vDataFrame history (user modifications). To use to keep the previous 
    vDataFrame history.
saving: list, optional
    List to use to reconstruct the vDataFrame from previous transformations.

Returns
-------
vDataFrame
    The vDataFrame associated to the input relation.
    """
    # Initialization
    from verticapy.core.vdataframe import vDataFrame
    from verticapy.sql.flex import isvmap
    from verticapy.sql.dtypes import get_data_types
    from verticapy.utils._cast import to_category
    from verticapy.sql._utils._format import quote_ident

    if isinstance(vdf, vDataFrame):
        external = vdf._VERTICAPY_VARIABLES_["external"]
        symbol = vdf._VERTICAPY_VARIABLES_["symbol"]
        sql_push_ext = vdf._VERTICAPY_VARIABLES_["sql_push_ext"]
        vdf.__init__("", empty=True)
        vdf._VERTICAPY_VARIABLES_["external"] = external
        vdf._VERTICAPY_VARIABLES_["symbol"] = symbol
        vdf._VERTICAPY_VARIABLES_["sql_push_ext"] = sql_push_ext
    else:
        vdf = vDataFrame("", empty=True)
    vdf._VERTICAPY_VARIABLES_["input_relation"] = name
    vdf._VERTICAPY_VARIABLES_["main_relation"] = relation
    vdf._VERTICAPY_VARIABLES_["schema"] = schema
    vdf._VERTICAPY_VARIABLES_["where"] = []
    vdf._VERTICAPY_VARIABLES_["order_by"] = {}
    vdf._VERTICAPY_VARIABLES_["exclude_columns"] = []
    vdf._VERTICAPY_VARIABLES_["history"] = history
    vdf._VERTICAPY_VARIABLES_["saving"] = saving
    dtypes = get_data_types(f"SELECT * FROM {relation} LIMIT 0")
    vdf._VERTICAPY_VARIABLES_["columns"] = ['"' + item[0] + '"' for item in dtypes]

    # Creating the vColumns
    for column, ctype in dtypes:
        if '"' in column:
            column_str = column.replace('"', "_")
            warning_message = (
                f'A double quote " was found in the column {column}, its '
                f"alias was changed using underscores '_' to {column_str}"
            )
            warnings.warn(warning_message, Warning)
        from verticapy.core.vcolumn import vColumn

        column_name = '"' + column.replace('"', "_") + '"'
        category = to_category(ctype)
        if (ctype.lower()[0:12] in ("long varbina", "long varchar")) and (
            isvmap(expr=relation, column=column,)
        ):
            category = "vmap"
            ctype = "VMAP(" + "(".join(ctype.split("(")[1:]) if "(" in ctype else "VMAP"
        new_vColumn = vColumn(
            column_name,
            parent=vdf,
            transformations=[(quote_ident(column), ctype, category,)],
        )
        setattr(vdf, column_name, new_vColumn)
        setattr(vdf, column_name[1:-1], new_vColumn)
        new_vColumn.init = False

    return vdf


vdf_from_relation = vDataFrameSQL
