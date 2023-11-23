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
import vertica_python
from typing import Optional, Union

from vertica_python.errors import QueryError

import verticapy._config.config as conf
from verticapy._utils._gen import gen_tmp_name
from verticapy._utils._sql._format import format_type, format_schema_table, quote_ident
from verticapy._utils._sql._sys import _executeSQL
from verticapy.connection import current_cursor


from verticapy.sql.drop import drop


def vertica_python_dtype(
    type_name: str, display_size: int = 0, precision: int = 0, scale: int = 0
) -> str:
    """
    Takes as input the Vertica Python type code and
    returns its corresponding data type.
    """
    res = type_name
    has_precision_scale = not (
        type_name.lower().startswith(
            ("array", "bool", "date", "int", "map", "row", "set", "uuid")
        )
    )
    if display_size and has_precision_scale:
        res += f"({display_size})"
    elif scale and precision and has_precision_scale:
        res += f"({precision},{scale})"
    return res


def get_data_types(
    expr: Optional[str] = None,
    column: Optional[str] = None,
    table_name: Optional[str] = None,
    schema: Optional[str] = None,
    usecols: Optional[list] = None,
) -> Union[tuple, list[tuple]]:
    """
    Returns customized relation columns and the
    respective data types. This process creates
    a temporary table.

    If table_name is defined, the expression is
    ignored and the function returns the table/
    view column names and data types.

    Parameters
    ----------
    expr: str, optional
        An expression in pure SQL. If empty, the
        parameter 'table_name' must be defined.
    column: str, optional
        If  not empty, the function returns only
        the data type of the input column if it
        is in the relation.
    table_name: str, optional
        Input table Name.
    schema: str, optional
        Table schema.
    usecols: list, optional
        List   of  columns   to  consider.   This
        parameter can not be  used if 'column' is
        defined.

    Returns
    -------
    list of tuples
        The  list  of the  different columns  and
        their respective type.

    Examples
    --------
    .. ipython:: python

        from verticapy.sql import get_data_types

        # returns the data type of multiple columns
        get_data_types("SELECT pclass, embarked, AVG(survived) FROM public.titanic GROUP BY 1, 2")

        # returns the data type of one column
        get_data_types("SELECT pclass, embarked, AVG(survived) FROM public.titanic GROUP BY 1, 2",
                        column="pclass")
    """
    if not (schema):
        schema = conf.get_option("temp_schema")
    usecols = format_type(usecols, dtype=list)
    if not expr and not table_name:
        raise ValueError(
            "Missing parameter: 'expr' and 'table_name' can not both be empty."
        )
    if (column) and (usecols):
        raise ValueError("Parameters 'column' and 'usecols' can not both be defined.")
    if (expr) and (table_name):
        warning_message = (
            "As parameter 'table_name' is defined, "
            "parameter 'expression' is ignored."
        )
        warnings.warn(warning_message, Warning)
    if isinstance(current_cursor(), vertica_python.vertica.cursor.Cursor) and not (
        table_name
    ):
        if column:
            column_name_ident = quote_ident(column)
            query = f"SELECT {column_name_ident} FROM ({expr}) x LIMIT 0;"
        elif usecols:
            query = f"""
                SELECT 
                    {", ".join(quote_ident(usecols))} 
                FROM ({expr}) x 
                LIMIT 0;"""
        else:
            query = expr
        try:
            _executeSQL(query, print_time_sql=False)
            description, ctype = current_cursor().description, []
            for d in description:
                ctype += [
                    [
                        d[0],
                        vertica_python_dtype(
                            type_name=d.type_name,
                            display_size=d[2],
                            precision=d[4],
                            scale=d[5],
                        ),
                    ]
                ]
            if column:
                return ctype[0][1]
            return ctype
        except QueryError:
            pass
    if not table_name:
        table_name, schema = gen_tmp_name(name="table"), "v_temp_schema"
        drop(format_schema_table(schema, table_name), method="table")
        try:
            if schema == "v_temp_schema":
                table = table_name
                local = "LOCAL"
            else:
                table = format_schema_table(schema, table_name)
                local = ""
            _executeSQL(
                query=f"""
                    CREATE {local} TEMPORARY TABLE {table} 
                    ON COMMIT PRESERVE ROWS 
                    AS {expr}""",
                print_time_sql=False,
            )
        finally:
            drop(format_schema_table(schema, table_name), method="table")
        drop_final_table = True
    else:
        drop_final_table = False
    usecols_str, column_name = "", ""
    if usecols:
        usecols_str = [
            "'" + column.lower().replace("'", "''") + "'" for column in usecols
        ]
        usecols_str = f" AND LOWER(column_name) IN ({', '.join(usecols_str)})"
    if column:
        column_name = f"column_name = '{column}' AND "
    query = f"""
        SELECT 
            column_name,
            data_type,
            ordinal_position 
        FROM {{}}
        WHERE {column_name}table_name = '{table_name}' 
            AND table_schema = '{schema}'{usecols_str}"""
    cursor = _executeSQL(
        query=f"""
            SELECT 
                /*+LABEL('get_data_types')*/ 
                column_name,
                data_type 
            FROM 
                (({query.format("columns")}) 
                 UNION 
                 ({query.format("view_columns")})) x 
                ORDER BY ordinal_position""",
        title="Getting the data types.",
    )
    ctype = cursor.fetchall()
    if column and ctype:
        ctype = ctype[0][1]
    if drop_final_table:
        drop(format_schema_table(schema, table_name), method="table")
    return ctype
