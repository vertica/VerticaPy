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
from typing import Optional

from vertica_python.errors import QueryError

from verticapy._utils._sql._format import format_schema_table, quote_ident
from verticapy._utils._sql._sys import _executeSQL


def create_schema(
    schema: str,
    raise_error: bool = False,
) -> bool:
    """
    Creates a new schema.

    Parameters
    ----------
    schema: str
        Schema name.
    raise_error: bool, optional
        If the schema couldn't be created, the
        function raises an error.

    Returns
    -------
    bool
        True  if  the schema was  successfully
        created, False otherwise.

    Examples
    --------
    .. ipython:: python

        from verticapy.sql import create_schema

        create_schema(schema = "employees")

    .. ipython:: python
        :suppress:

        from verticapy import drop

        drop("employees.")
    """
    try:
        _executeSQL(f"CREATE SCHEMA {schema};", title="Creating the new schema.")
        return True
    except QueryError:
        if raise_error:
            raise
        return False


def create_table(
    table_name: str,
    dtype: dict,
    schema: Optional[str] = None,
    temporary_table: bool = False,
    temporary_local_table: bool = True,
    genSQL: bool = False,
    raise_error: bool = False,
) -> bool:
    """
    Creates a new table using the input columns'
    names and data types.

    Parameters
    ----------
    table_name: str
        The final table name.
    dtype: dict
        Dictionary  of the user types. Each  key
        represents  a column name and each value
        represents its data type.
        Example: {"age": "int", "name": "varchar"}
    schema: str, optional
        Schema name.
    temporary_table: bool, optional
        If set to True, a temporary table is
        created.
    temporary_local_table: bool, optional
        If  set to True,  a temporary local table
        is be created.  The  parameter 'schema'
        must be empty,  otherwise  this parameter
        is ignored.
    genSQL: bool, optional
        If set to True, the SQL code for creating
        the final table is generated but not
        executed.
    raise_error: bool, optional
        If  the  relation  couldn't  be  created,
        raises the entire error.

    Returns
    -------
    bool
        True   if  the  table  was   successfully
        created, False otherwise.

    Examples
    --------
    .. ipython:: python

        from verticapy.sql import create_table

        # Generates the SQL needed to create the Table
        create_table(
            table_name = "employees",
            schema = "public",
            dtype = {"name": "VARCHAR(60)", "salary": "FLOAT"},
            genSQL = True
        )

        # Creates the table
        create_table(
            table_name = "employees",
            schema = "public",
            dtype = {"name": "VARCHAR(60)", "salary": "FLOAT"}
        )

    .. code-block:: python

        %load_ext verticapy.sql

        %%sql
        SELECT * FROM public.employees;

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame, drop

        html_file = open("SPHINX_DIRECTORY/figures/sql_create_create_table.html", "w")
        html_file.write(vDataFrame(input_relation = '"public"."employees"')._repr_html_())
        html_file.close()

        drop("public.employees")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_create_create_table.html

    """
    if schema.lower() == "v_temp_schema":
        schema = ""
        temporary_local_table = True
    if schema:
        input_relation = format_schema_table(schema, table_name)
    else:
        input_relation = quote_ident(table_name)
    temp = "TEMPORARY " if temporary_table else ""
    if not schema:
        temp = "LOCAL TEMPORARY " if temporary_local_table else ""
    dtype_str = [f"{quote_ident(column)} {dtype[column]}" for column in dtype]
    dtype_str = ", ".join(dtype_str)
    on_commit = " ON COMMIT PRESERVE ROWS" if temp else ""
    query = f"CREATE {temp}TABLE {input_relation}({dtype_str}){on_commit};"
    if genSQL:
        return query
    try:
        _executeSQL(query, title="Creating the new table.")
        return True
    except QueryError:
        if raise_error:
            raise
        return False
