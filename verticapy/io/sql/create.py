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


def create_schema(
    schema: str, raise_error: bool = False,
):
    """
Creates a new schema.

Parameters
----------
schema: str
    Schema name.
raise_error: bool, optional
    If the schema couldn't be created, the function raises an error.

Returns
-------
bool
    True if the schema was successfully created, False otherwise.
    """
    from verticapy.utils._toolbox import executeSQL
    from verticapy.io.sql._utils._format import quote_ident

    try:
        executeSQL(f"CREATE SCHEMA {schema};", title="Creating the new schema.")
        return True
    except:
        if raise_error:
            raise
        return False


def create_table(
    table_name: str,
    dtype: dict,
    schema: str = "",
    temporary_table: bool = False,
    temporary_local_table: bool = True,
    genSQL: bool = False,
    raise_error: bool = False,
):
    """
Creates a new table using the input columns' names and data types.

Parameters
----------
table_name: str, optional
    The final table name.
dtype: dict
    Dictionary of the user types. Each key represents a column name and each
    value represents its data type. 
    Example: {"age": "int", "name": "varchar"}
schema: str, optional
    Schema name.
temporary_table: bool, optional
    If set to True, a temporary table will be created.
temporary_local_table: bool, optional
    If set to True, a temporary local table will be created. The parameter 
    'schema' must be empty, otherwise this parameter is ignored.
genSQL: bool, optional
    If set to True, the SQL code for creating the final table will be 
    generated but not executed.
raise_error: bool, optional
    If the relation couldn't be created, raises the entire error.

Returns
-------
bool
    True if the table was successfully created, False otherwise.
    """
    from verticapy.utils._toolbox import executeSQL, quote_ident

    # -#
    if schema.lower() == "v_temp_schema":
        schema = ""
        temporary_local_table = True
    if schema:
        input_relation = quote_ident(schema) + "." + quote_ident(table_name)
    else:
        input_relation = quote_ident(table_name)
    temp = "TEMPORARY " if temporary_table else ""
    if not (schema):
        temp = "LOCAL TEMPORARY " if temporary_local_table else ""
    dtype_str = [f"{quote_ident(column)} {dtype[column]}" for column in dtype]
    dtype_str = ", ".join(dtype_str)
    on_commit = " ON COMMIT PRESERVE ROWS" if temp else ""
    query = f"CREATE {temp}TABLE {input_relation}({dtype_str}){on_commit};"
    if genSQL:
        return query
    try:
        executeSQL(query, title="Creating the new table.")
        return True
    except:
        if raise_error:
            raise
        return False


def create_verticapy_schema():
    """
Creates a schema named 'verticapy' used to store VerticaPy extended models.
    """
    from verticapy.utils._toolbox import executeSQL

    sql = "CREATE SCHEMA IF NOT EXISTS verticapy;"
    executeSQL(sql, title="Creating VerticaPy schema.")
    sql = """CREATE TABLE IF NOT EXISTS verticapy.models (model_name VARCHAR(128), 
                                                          category VARCHAR(128), 
                                                          model_type VARCHAR(128), 
                                                          create_time TIMESTAMP, 
                                                          size INT);"""
    executeSQL(sql, title="Creating the models table.")
    sql = """CREATE TABLE IF NOT EXISTS verticapy.attr (model_name VARCHAR(128), 
                                                        attr_name VARCHAR(128), 
                                                        value VARCHAR(65000));"""
    executeSQL(sql, title="Creating the attr table.")
