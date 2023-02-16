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

from verticapy.errors import ParameterError
from verticapy._utils._collect import save_verticapy_logs
from verticapy._utils._sql import _executeSQL
from verticapy.sql._utils._format import quote_ident
from verticapy.connect import current_cursor

from verticapy.core.str_sql import str_sql


@save_verticapy_logs
def compute_flextable_keys(flex_name: str, usecols: list = []):
    """
Computes the flex table keys and returns the predicted data types.

Parameters
----------
flex_name: str
    Flex table name.
usecols: list, optional
    List of columns to consider.

Returns
-------
List of tuples
    List of virtual column names and their respective data types.
    """
    _executeSQL(
        query=f"""
            SELECT 
                /*+LABEL('utilities.compute_flex_table_keys')*/
                compute_flextable_keys('{flex_name}');""",
        title="Guessing flex tables keys.",
    )
    usecols_str = [
        "'" + str(column).lower().replace("'", "''") + "'" for column in usecols
    ]
    usecols_str = ", ".join(usecols_str)
    where = f" WHERE LOWER(key_name) IN ({usecols_str})" if (usecols) else ""
    result = _executeSQL(
        query=f"""
            SELECT 
                /*+LABEL('utilities.compute_flex_table_keys')*/
                key_name,
                data_type_guess 
            FROM {flex_name}_keys{where}""",
        title="Guessing the data types.",
        method="fetchall",
    )
    return result


@save_verticapy_logs
def compute_vmap_keys(
    expr: Union[str, str_sql], vmap_col: str, limit: int = 100,
):
    """
Computes the most frequent keys in the input VMap.

Parameters
----------
expr: str / vDataFrame
    Input expression. You can also specify a vDataFrame or a customized 
    relation, but you must enclose it with an alias. For example, "(SELECT 1) x" 
    is allowed, whereas "(SELECT 1)" and "SELECT 1" are not.
vmap_col: str
    VMap column.
limit: int, optional
    Maximum number of keys to consider.

Returns
-------
List of tuples
    List of virtual column names and their respective frequencies.
    """
    from verticapy.core.vdataframe.base import vDataFrame

    vmap = quote_ident(vmap_col)
    if isinstance(expr, vDataFrame):
        assert expr[vmap_col].isvmap(), ParameterError(
            f"Virtual column {vmap_col} is not a VMAP."
        )
        expr = expr.__genSQL__()
    result = _executeSQL(
        (
            "SELECT /*+LABEL('utilities.compute_vmap_keys')*/ keys, COUNT(*) FROM "
            f"(SELECT MAPKEYS({vmap}) OVER (PARTITION BEST) FROM {expr})"
            f" VERTICAPY_SUBTABLE GROUP BY 1 ORDER BY 2 DESC LIMIT {limit};"
        ),
        title="Getting vmap most occurent keys.",
        method="fetchall",
    )
    return result


def isflextable(table_name: str, schema: str):
    """
Checks if the input relation is a flextable.

Parameters
----------
table_name: str
    Name of the table to check.
schema: str
    Table schema.

Returns
-------
bool
    True if the relation is a flex table.
    """
    table_name = quote_ident(table_name)[1:-1]
    schema = quote_ident(schema)[1:-1]
    sql = (
        f"SELECT is_flextable FROM v_catalog.tables WHERE table_name = '{table_name}' AND "
        f"table_schema = '{schema}' AND is_flextable LIMIT 1;"
    )
    result = _executeSQL(
        sql, title="Checking if the table is a flextable.", method="fetchall",
    )
    return bool(result)


def isvmap(
    expr: Union[str, str_sql], column: str,
):
    """
Checks if the input column is a VMap.

Parameters
----------
expr: str / vDataFrame
    Any relation or expression. If you enter an expression,
    you must enclose it in parentheses and provide an alias.
column: str
    Name of the column to check.

Returns
-------
bool
    True if the column is a VMap.
    """
    # -#
    from verticapy.vdataframe import vDataFrame

    column = quote_ident(column)
    if isinstance(expr, vDataFrame):
        expr = expr.__genSQL__()
    sql = f"SELECT MAPVERSION({column}) AS isvmap, {column} FROM {expr} WHERE {column} IS NOT NULL LIMIT 1;"
    try:
        result = _executeSQL(
            sql, title="Checking if the column is a vmap.", method="fetchall",
        )
        dtype = current_cursor().description[1][1]
        if dtype not in (
            115,
            116,
        ):  # 116 is for long varbinary and 115 is for long varchar
            return False
    except Exception as e:
        if "'utf-8' codec can't decode byte" in str(e):
            try:
                sql = f"SELECT MAPVERSION({column}) AS isvmap FROM {expr} WHERE {column} IS NOT NULL LIMIT 1;"
                result = _executeSQL(
                    sql, title="Checking if the column is a vmap.", method="fetchall",
                )
            except:
                return False
        else:
            return False
    if len(result) == 0 or (result[0][0] == -1):
        return False
    else:
        return True
