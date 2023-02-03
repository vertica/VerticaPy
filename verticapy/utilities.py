# (c) Copyright [2018-2023] Micro Focus or one of its affiliates.
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
import os, math, shutil, re, time, decimal, warnings, datetime, inspect, csv
from typing import Union

# VerticaPy Modules
import vertica_python
import verticapy
from verticapy.decorators import (
    save_verticapy_logs,
    check_dtypes,
    check_minimum_version,
)
from verticapy.toolbox import *
from verticapy.javascript import datatables_repr
from verticapy.errors import *

# Other Modules
import pandas as pd

# IPython - Optional
try:
    from IPython.display import HTML, display, Markdown
except:
    pass

#
# Utilities Functions
#
# ---#
@check_dtypes
@save_verticapy_logs
def compute_flextable_keys(flex_name: str, usecols: list = []):
    """
----------------------------------------------------------------------------------------
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
    executeSQL(
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
    result = executeSQL(
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


# ---#
@check_dtypes
@save_verticapy_logs
def compute_vmap_keys(
    expr: Union[str, str_sql], vmap_col: str, limit: int = 100,
):
    """
----------------------------------------------------------------------------------------
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
    from verticapy import vDataFrame

    vmap = quote_ident(vmap_col)
    if isinstance(expr, vDataFrame):
        assert expr[vmap_col].isvmap(), ParameterError(
            f"Virtual column {vmap_col} is not a VMAP."
        )
        expr = expr.__genSQL__()
    result = executeSQL(
        (
            "SELECT /*+LABEL('utilities.compute_vmap_keys')*/ keys, COUNT(*) FROM "
            f"(SELECT MAPKEYS({vmap}) OVER (PARTITION BEST) FROM {expr})"
            f" VERTICAPY_SUBTABLE GROUP BY 1 ORDER BY 2 DESC LIMIT {limit};"
        ),
        title="Getting vmap most occurent keys.",
        method="fetchall",
    )
    return result


# ---#
@check_dtypes
def create_schema(
    schema: str, raise_error: bool = False,
):
    """
----------------------------------------------------------------------------------------
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
    try:
        executeSQL(f"CREATE SCHEMA {schema};", title="Creating the new schema.")
        return True
    except:
        if raise_error:
            raise
        return False


# ---#
@check_dtypes
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
----------------------------------------------------------------------------------------
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


# ---#
def create_verticapy_schema():
    """
----------------------------------------------------------------------------------------
Creates a schema named 'verticapy' used to store VerticaPy extended models.
    """
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


# ---#
@check_dtypes
def drop(name: str = "", method: str = "auto", raise_error: bool = False, **kwds):
    """
----------------------------------------------------------------------------------------
Drops the input relation. This can be a model, view, table, text index,
schema, or geo index.

Parameters
----------
name: str, optional
    Relation name. If empty, it will drop all VerticaPy temporary 
    elements.
method / relation_type: str, optional
    Method used to drop.
        auto   : identifies the table/view/index/model to drop. 
                 It will never drop an entire schema unless the 
                 method is set to 'schema'.
        model  : drops the input model.
        table  : drops the input table.
        view   : drops the input view.        
        geo    : drops the input geo index.
        text   : drops the input text index.
        schema : drops the input schema.
raise_error: bool, optional
    If the object couldn't be dropped, this function raises an error.

Returns
-------
bool
    True if the relation was dropped, False otherwise.
    """
    # -#
    if "relation_type" in kwds and method == "auto":
        method = kwds["relation_type"]
    raise_error_if_not_in(
        "method", method, ["table", "view", "model", "geo", "text", "auto", "schema"],
    )
    schema, relation = schema_relation(name)
    schema, relation = schema[1:-1], relation[1:-1]
    if not (name):
        method = "temp"
    if method == "auto":
        fail, end_conditions = False, False
        result = executeSQL(
            query=f"""
            SELECT 
                /*+LABEL('utilities.drop')*/ * 
            FROM columns 
            WHERE table_schema = '{schema}' 
                AND table_name = '{relation}'""",
            print_time_sql=False,
            method="fetchrow",
        )
        if not (result):
            result = executeSQL(
                query=f"""
                SELECT 
                    /*+LABEL('utilities.drop')*/ * 
                FROM view_columns 
                WHERE table_schema = '{schema}' 
                    AND table_name = '{relation}'""",
                print_time_sql=False,
                method="fetchrow",
            )
        elif not (end_conditions):
            method = "table"
            end_conditions = True
        if not (result):
            try:
                result = executeSQL(
                    query=f"""
                    SELECT 
                        /*+LABEL('utilities.drop')*/ model_type 
                    FROM verticapy.models 
                    WHERE LOWER(model_name) = '{quote_ident(name).lower()}'""",
                    print_time_sql=False,
                    method="fetchrow",
                )
            except:
                result = []
        elif not (end_conditions):
            method = "view"
            end_conditions = True
        if not (result):
            result = executeSQL(
                query=f"""
                SELECT 
                    /*+LABEL('utilities.drop')*/ * 
                FROM models 
                WHERE schema_name = '{schema}' 
                    AND model_name = '{relation}'""",
                print_time_sql=False,
                method="fetchrow",
            )
        elif not (end_conditions):
            method = "model"
            end_conditions = True
        if not (result):
            result = executeSQL(
                query=f"""
                SELECT 
                    /*+LABEL('utilities.drop')*/ * 
                FROM 
                    (SELECT STV_Describe_Index () OVER ()) x  
                WHERE name IN ('{schema}.{relation}',
                               '{relation}',
                               '\"{schema}\".\"{relation}\"',
                               '\"{relation}\"',
                               '{schema}.\"{relation}\"',
                               '\"{schema}\".{relation}')""",
                print_time_sql=False,
                method="fetchrow",
            )
        elif not (end_conditions):
            method = "model"
            end_conditions = True
        if not (result):
            try:
                executeSQL(
                    query=f"""
                        SELECT 
                            /*+LABEL(\'utilities.drop\')*/ * 
                        FROM "{schema}"."{relation}" LIMIT 0;""",
                    print_time_sql=False,
                )
                method = "text"
            except:
                fail = True
        elif not (end_conditions):
            method = "geo"
            end_conditions = True
        if fail:
            if raise_error:
                raise MissingRelation(
                    f"No relation / index / view / model named '{name}' was detected."
                )
            return False
    query = ""
    if method == "model":
        model_type = kwds["model_type"] if "model_type" in kwds else None
        try:
            result = executeSQL(
                query=f"""
                    SELECT 
                        /*+LABEL('utilities.drop')*/ model_type 
                    FROM verticapy.models 
                    WHERE LOWER(model_name) = '{quote_ident(name).lower()}'""",
                print_time_sql=False,
                method="fetchfirstelem",
            )
            is_in_verticapy_schema = True
            if not (model_type):
                model_type = result
        except:
            is_in_verticapy_schema = False
        if (
            model_type
            in (
                "DBSCAN",
                "LocalOutlierFactor",
                "CountVectorizer",
                "KernelDensity",
                "AutoDataPrep",
                "KNeighborsRegressor",
                "KNeighborsClassifier",
                "NearestCentroid",
            )
            or is_in_verticapy_schema
        ):
            if model_type in ("DBSCAN", "LocalOutlierFactor"):
                drop(name, method="table")
            elif model_type == "CountVectorizer":
                drop(name, method="text")
                if is_in_verticapy_schema:
                    res = executeSQL(
                        query=f"""
                            SELECT 
                                /*+LABEL('utilities.drop')*/ value 
                            FROM verticapy.attr 
                            WHERE LOWER(model_name) = '{quote_ident(name).lower()}' 
                                AND attr_name = 'countvectorizer_table'""",
                        print_time_sql=False,
                        method="fetchrow",
                    )
                    if res and res[0]:
                        drop(res[0], method="table")
            elif model_type == "KernelDensity":
                table_name = name.replace('"', "") + "_KernelDensity_Map"
                drop(table_name, method="table")
                model_name = name.replace('"', "") + "_KernelDensity_Tree"
                drop(model_name, method="model")
            elif model_type == "AutoDataPrep":
                drop(name, method="table")
            if is_in_verticapy_schema:
                executeSQL(
                    query=f"""
                        DELETE /*+LABEL('utilities.drop')*/ 
                        FROM verticapy.models 
                        WHERE LOWER(model_name) = '{quote_ident(name).lower()}';""",
                    title="Deleting vModel.",
                )
                executeSQL("COMMIT;", title="Commit.")
                executeSQL(
                    query=f"""
                        DELETE /*+LABEL('utilities.drop')*/ 
                        FROM verticapy.attr 
                        WHERE LOWER(model_name) = '{quote_ident(name).lower()}';""",
                    title="Deleting vModel attributes.",
                )
                executeSQL("COMMIT;", title="Commit.")
        else:
            query = f"DROP MODEL {name};"
    elif method == "table":
        query = f"DROP TABLE {name};"
    elif method == "view":
        query = f"DROP VIEW {name};"
    elif method == "geo":
        query = f"SELECT STV_Drop_Index(USING PARAMETERS index ='{name}') OVER ();"
    elif method == "text":
        query = f"DROP TEXT INDEX {name};"
    elif method == "schema":
        query = f"DROP SCHEMA {name} CASCADE;"
    if query:
        try:
            executeSQL(query, title="Deleting the relation.")
            result = True
        except:
            if raise_error:
                raise
            result = False
    elif method == "temp":
        sql = """SELECT /*+LABEL('utilities.drop')*/
                    table_schema, table_name 
                 FROM columns 
                 WHERE LOWER(table_name) LIKE '%_verticapy_tmp_%' 
                 GROUP BY 1, 2;"""
        all_tables = result = executeSQL(sql, print_time_sql=False, method="fetchall")
        for elem in all_tables:
            table = format_schema_table(
                elem[0].replace('"', '""'), elem[1].replace('"', '""')
            )
            drop(table, method="table")
        sql = """SELECT /*+LABEL('utilities.drop')*/
                    table_schema, table_name 
                 FROM view_columns 
                 WHERE LOWER(table_name) LIKE '%_verticapy_tmp_%' 
                 GROUP BY 1, 2;"""
        all_views = executeSQL(sql, print_time_sql=False, method="fetchall")
        for elem in all_views:
            view = format_schema_table(
                elem[0].replace('"', '""'), elem[1].replace('"', '""')
            )
            drop(view, method="view")
        result = True
    else:
        result = True
    return result


# ---#
@check_dtypes
def get_data_types(
    expr: str = "",
    column: str = "",
    table_name: str = "",
    schema: str = "public",
    usecols: list = [],
):
    """
----------------------------------------------------------------------------------------
Returns customized relation columns and the respective data types.
This process creates a temporary table.

If table_name is defined, the expression is ignored and the function
returns the table/view column names and data types.

Parameters
----------
expr: str, optional
    An expression in pure SQL. If empty, the parameter 'table_name' must be
    defined.
column: str, optional
    If not empty, it will return only the data type of the input column if it
    is in the relation.
table_name: str, optional
    Input table Name.
schema: str, optional
    Table schema.
usecols: list, optional
    List of columns to consider. This parameter can not be used if 'column'
    is defined.

Returns
-------
list of tuples
    The list of the different columns and their respective type.
    """
    assert expr or table_name, ParameterError(
        "Missing parameter: 'expr' and 'table_name' can not both be empty."
    )
    assert not (column) or not (usecols), ParameterError(
        "Parameters 'column' and 'usecols' can not both be defined."
    )
    if expr and table_name:
        warning_message = (
            "As parameter 'table_name' is defined, parameter 'expression' is ignored."
        )
        warnings.warn(warning_message, Warning)

    from verticapy.connect import current_cursor

    if isinstance(current_cursor(), vertica_python.vertica.cursor.Cursor) and not (
        table_name
    ):
        try:
            if column:
                column_name_ident = quote_ident(column)
                query = f"SELECT {column_name_ident} FROM ({expr}) x LIMIT 0;"
            elif usecols:
                query = f"""
                    SELECT 
                        {", ".join([quote_ident(column) for column in usecols])} 
                    FROM ({expr}) x 
                    LIMIT 0;"""
            else:
                query = expr
            executeSQL(query, print_time_sql=False)
            description, ctype = current_cursor().description, []
            for d in description:
                ctype += [
                    [
                        d[0],
                        get_final_vertica_type(
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
        except:
            pass
    if not (table_name):
        table_name, schema = gen_tmp_name(name="table"), "v_temp_schema"
        drop(format_schema_table(schema, table_name), method="table")
        try:
            if schema == "v_temp_schema":
                table = table_name
                local = "LOCAL"
            else:
                table = format_schema_table(schema, table_name)
                local = ""
            executeSQL(
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
    cursor = executeSQL(
        query=f"""
            SELECT 
                /*+LABEL('utilities.get_data_types')*/ 
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


# ---#
@save_verticapy_logs
def help_start():
    """
----------------------------------------------------------------------------------------
VERTICAPY Interactive Help (FAQ).
    """
    path = os.path.dirname(verticapy.__file__)
    img1 = verticapy.gen_verticapy_logo_html(size="10%")
    img2 = verticapy.gen_verticapy_logo_str()
    message = img1 if (isnotebook()) else img2
    message += (
        "\n\n&#128226; Welcome to the <b>VerticaPy</b> help module."
        "\n\nThis module can help you connect to Vertica, "
        "create a Virtual DataFrame, load your data, and more.\n "
        "- <b>[Enter  0]</b> Overview of the library\n "
        "- <b>[Enter  1]</b> Load an example dataset\n "
        "- <b>[Enter  2]</b> View an example of data analysis with VerticaPy\n "
        "- <b>[Enter  3]</b> Contribute on GitHub\n "
        "- <b>[Enter  4]</b> View the SQL code generated by a vDataFrame and "
        "the time elapsed for the query\n "
        "- <b>[Enter  5]</b> Load your own dataset into Vertica \n "
        "- <b>[Enter  6]</b> Write SQL queries in Jupyter\n "
        "- <b>[Enter -1]</b> Exit"
    )
    if not (isnotebook()):
        message = message.replace("<b>", "").replace("</b>", "")
    display(Markdown(message)) if (isnotebook()) else print(message)
    try:
        response = int(input())
    except:
        print("Invalid choice.\nPlease enter a number between 0 and 11.")
        try:
            response = int(input())
        except:
            print("Invalid choice.\nRerun the help_start function when you need help.")
            return
    if response == 0:
        link = "https://www.vertica.com/python/quick-start.php"
    elif response == 1:
        link = "https://www.vertica.com/python/documentation_last/datasets/"
    elif response == 2:
        link = "https://www.vertica.com/python/examples/"
    elif response == 3:
        link = "https://github.com/vertica/VerticaPy/"
    elif response == 4:
        link = "https://www.vertica.com/python/documentation_last/utilities/set_option/"
    elif response == 5:
        link = "https://www.vertica.com/python/documentation_last/datasets/"
    elif response == 6:
        link = "https://www.vertica.com/python/documentation_last/extensions/sql/"
    elif response == -1:
        message = "Thank you for using the VerticaPy help module."
    elif response == 666:
        message = (
            "Thank you so much for using this library. My only purpose is to solve "
            "real Big Data problems in the context of Data Science. I worked years "
            "to be able to create this API and give you a real way to analyze your "
            "data.\n\nYour devoted Data Scientist: <i>Badr Ouali</i>"
        )
    else:
        message = "Invalid choice.\nPlease enter a number between -1 and 6."
    if 0 <= response <= 6:
        if not (isnotebook()):
            message = f"Please go to {link}"
        else:
            message = f"Please go to <a href='{link}'>{link}</a>"
    display(Markdown(message)) if (isnotebook()) else print(message)


# ---#
def init_interactive_mode(all_interactive=False):
    """Activate the datatables representation for all the vDataFrames."""
    set_option("interactive", all_interactive)


# ---#
@check_dtypes
@save_verticapy_logs
def insert_into(
    table_name: str,
    data: list,
    schema: str = "",
    column_names: list = [],
    copy: bool = True,
    genSQL: bool = False,
):
    """
----------------------------------------------------------------------------------------
Inserts the dataset into an existing Vertica table.

Parameters
----------
table_name: str
    Name of the table to insert into.
data: list
    The data to ingest.
schema: str, optional
    Schema name.
column_names: list, optional
    Name of the column(s) to insert into.
copy: bool, optional
    If set to True, the batch insert is converted to a COPY statement 
    with prepared statements. Otherwise, the INSERTs are performed
    sequentially.
genSQL: bool, optional
    If set to True, the SQL code that would be used to insert the data 
    is generated, but not executed.

Returns
-------
int
    The number of rows ingested.

See Also
--------
pandas_to_vertica : Ingests a pandas DataFrame into the Vertica database.
    """
    if not (schema):
        schema = verticapy.OPTIONS["temp_schema"]
    input_relation = format_schema_table(schema, table_name)
    if not (column_names):
        result = executeSQL(
            query=f"""
                SELECT /*+LABEL('utilities.insert_into')*/
                    column_name
                FROM columns 
                WHERE table_name = '{table_name}' 
                    AND table_schema = '{schema}' 
                ORDER BY ordinal_position""",
            title=f"Getting the table {input_relation} column names.",
            method="fetchall",
        )
        column_names = [elem[0] for elem in result]
        assert column_names, MissingRelation(
            f"The table {input_relation} does not exist."
        )
    cols = [quote_ident(col) for col in column_names]
    if copy and not (genSQL):
        executeSQL(
            query=f"""
                INSERT INTO {input_relation} 
                ({", ".join(cols)})
                VALUES ({", ".join(["%s" for i in range(len(cols))])})""",
            title=(
                f"Insert new lines in the {table_name} table. "
                "The batch insert is converted into a COPY "
                "statement by using prepared statements."
            ),
            data=list(map(tuple, data)),
        )
        executeSQL("COMMIT;", title="Commit.")
        return len(data)
    else:
        if genSQL:
            sql = []
        i, n, total_rows = 0, len(data), 0
        header = f"""
            INSERT INTO {input_relation}
            ({", ".join(cols)}) VALUES """
        for i in range(n):
            sql_tmp = "("
            for d in data[i]:
                if isinstance(d, str):
                    d_str = d.replace("'", "''")
                    sql_tmp += f"'{d_str}'"
                elif d is None or d != d:
                    sql_tmp += "NULL"
                else:
                    sql_tmp += f"'{d}'"
                sql_tmp += ","
            sql_tmp = sql_tmp[:-1] + ");"
            query = header + sql_tmp
            if genSQL:
                sql += [clean_query(query)]
            else:
                try:
                    executeSQL(
                        query=query,
                        title=f"Insert a new line in the relation: {input_relation}.",
                    )
                    executeSQL("COMMIT;", title="Commit.")
                    total_rows += 1
                except Exception as e:
                    warning_message = f"Line {i} was skipped.\n{e}"
                    warnings.warn(warning_message, Warning)
        if genSQL:
            return sql
        else:
            return total_rows


# ---#
@check_dtypes
def isflextable(table_name: str, schema: str):
    """
----------------------------------------------------------------------------------------
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
    result = executeSQL(
        sql, title="Checking if the table is a flextable.", method="fetchall",
    )
    return bool(result)


# ---#
@check_dtypes
def isvmap(
    expr: Union[str, str_sql], column: str,
):
    """
----------------------------------------------------------------------------------------
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
    from verticapy.connect import current_cursor

    column = quote_ident(column)
    if isinstance(expr, vDataFrame):
        expr = expr.__genSQL__()
    sql = f"SELECT MAPVERSION({column}) AS isvmap, {column} FROM {expr} WHERE {column} IS NOT NULL LIMIT 1;"
    try:
        result = executeSQL(
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
                result = executeSQL(
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


# ---#
@check_dtypes
@save_verticapy_logs
def pandas_to_vertica(
    df: pd.DataFrame,
    name: str = "",
    schema: str = "",
    dtype: dict = {},
    parse_nrows: int = 10000,
    temp_path: str = "",
    insert: bool = False,
):
    """
----------------------------------------------------------------------------------------
Ingests a pandas DataFrame into the Vertica database by creating a 
CSV file and then using flex tables to load the data.

Parameters
----------
df: pandas.DataFrame
    The pandas.DataFrame to ingest.
name: str, optional
    Name of the new relation or the relation in which to insert the 
    data. If unspecified, a temporary local table is created. This 
    temporary table is dropped at the end of the local session.
schema: str, optional
    Schema of the new relation. If empty, a temporary schema is used. 
    To modify the temporary schema, use the 'set_option' function.
dtype: dict, optional
    Dictionary of input types. Providing a dictionary can increase 
    ingestion speed and precision. If specified, rather than parsing 
    the intermediate CSV and guessing the input types, VerticaPy uses 
    the specified input types instead.
parse_nrows: int, optional
    If this parameter is greater than 0, VerticaPy creates and 
    ingests a temporary file containing 'parse_nrows' number 
    of rows to determine the input data types before ingesting 
    the intermediate CSV file containing the rest of the data. 
    This method of data type identification is less accurate, 
    but is much faster for large datasets.
temp_path: str, optional
    The path to which to write the intermediate CSV file. This 
    is useful in cases where the user does not have write 
    permissions on the current directory.
insert: bool, optional
    If set to True, the data are ingested into the input relation. 
    The column names of your table and the pandas.DataFrame must 
    match.
    
Returns
-------
vDataFrame
    vDataFrame of the new relation.

See Also
--------
read_csv  : Ingests a  CSV file into the Vertica database.
read_json : Ingests a JSON file into the Vertica database.
    """
    if not (schema):
        schema = verticapy.OPTIONS["temp_schema"]
    assert name or not (insert), ParameterError(
        "Parameter 'name' can not be empty when parameter 'insert' is set to True."
    )
    if not (name):
        tmp_name = gen_tmp_name(name="df")[1:-1]
    else:
        tmp_name = ""
    sep = "/" if (len(temp_path) > 1 and temp_path[-1] != "/") else ""
    path = f"{temp_path}{sep}{name}.csv"
    try:
        # Adding the quotes to STR pandas columns in order to simplify the ingestion.
        # Not putting them can lead to wrong data ingestion.
        str_cols = []
        for c in df.columns:
            if df[c].dtype == object and isinstance(
                df[c].loc[df[c].first_valid_index()], str
            ):
                str_cols += [c]
        if str_cols:
            tmp_df = df.copy()
            for c in str_cols:
                tmp_df[c] = '"' + tmp_df[c].str.replace('"', '""') + '"'
            clear = True
        else:
            tmp_df = df
            clear = False
        tmp_df.to_csv(
            path, index=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar="\027",
        )
        if str_cols:
            # to_csv is adding an undesired special character
            # we remove it
            with open(path, "r") as f:
                filedata = f.read()
            filedata = filedata.replace(",", ",")
            with open(path, "w") as f:
                f.write(filedata)

        if insert:
            input_relation = format_schema_table(schema, name)
            tmp_df_columns_str = ", ".join(
                ['"' + col.replace('"', '""') + '"' for col in tmp_df.columns]
            )
            executeSQL(
                query=f"""
                    COPY {input_relation}
                    ({tmp_df_columns_str}) 
                    FROM LOCAL '{path}' 
                    DELIMITER ',' 
                    NULL ''
                    ENCLOSED BY '\"' 
                    ESCAPE AS '\\' 
                    SKIP 1;""",
                title="Inserting the pandas.DataFrame.",
            )
            from verticapy import vDataFrame

            vdf = vDataFrame(name, schema=schema)
        elif tmp_name:
            vdf = read_csv(
                path,
                table_name=tmp_name,
                dtype=dtype,
                temporary_local_table=True,
                parse_nrows=parse_nrows,
                escape="\027",
            )
        else:
            vdf = read_csv(
                path,
                table_name=name,
                dtype=dtype,
                schema=schema,
                temporary_local_table=False,
                parse_nrows=parse_nrows,
                escape="\027",
            )
    finally:
        os.remove(path)
        if clear:
            del tmp_df
    return vdf


# ---#
def pcsv(
    path: str,
    sep: str = ",",
    header: bool = True,
    header_names: list = [],
    na_rep: str = "",
    quotechar: str = '"',
    escape: str = "\027",
    record_terminator: str = "\n",
    trim: bool = True,
    omit_empty_keys: bool = False,
    reject_on_duplicate: bool = False,
    reject_on_empty_key: bool = False,
    reject_on_materialized_type_error: bool = False,
    ingest_local: bool = True,
    flex_name: str = "",
    genSQL: bool = False,
):
    """
----------------------------------------------------------------------------------------
Parses a CSV file using flex tables. It will identify the columns and their
respective types.

Parameters
----------
path: str
    Absolute path where the CSV file is located.
sep: str, optional
    Column separator.
header: bool, optional
    If set to False, the parameter 'header_names' will be to use to name the 
    different columns.
header_names: list, optional
    List of the columns names.
na_rep: str, optional
    Missing values representation.
quotechar: str, optional
    Char which is enclosing the str values.
escape: str, optional
    Separator between each record.
record_terminator: str, optional
    A single-character value used to specify the end of a record.
trim: bool, optional
    Boolean, specifies whether to trim white space from header names and 
    key values.
omit_empty_keys: bool, optional
    Boolean, specifies how the parser handles header keys without values. 
    If true, keys with an empty value in the header row are not loaded.
reject_on_duplicate: bool, optional
    Boolean, specifies whether to ignore duplicate records (False), or to 
    reject duplicates (True). In either case, the load continues.
reject_on_empty_key: bool, optional
    Boolean, specifies whether to reject any row containing a key without a 
    value.
reject_on_materialized_type_error: bool, optional
    Boolean, specifies whether to reject any materialized column value that the 
    parser cannot coerce into a compatible data type.
ingest_local: bool, optional
    If set to True, the file will be ingested from the local machine.
flex_name: str, optional
    Flex table name.
genSQL: bool, optional
    If set to True, the SQL code for creating the final table is 
    generated but not executed. This is a good way to change the
    final relation types or to customize the data ingestion.

Returns
-------
dict
    dictionary containing each column and its type.

See Also
--------
read_csv  : Ingests a CSV file into the Vertica database.
read_json : Ingests a JSON file into the Vertica database.
    """
    if record_terminator == "\n":
        record_terminator = "\\n"
    if not (flex_name):
        flex_name = gen_tmp_name(name="flex")[1:-1]
    if header_names:
        header_names = f"header_names = '{sep.join(header_names)}',"
    else:
        header_names = ""
    ingest_local = " LOCAL" if ingest_local else ""
    trim = str(trim).lower()
    omit_empty_keys = str(omit_empty_keys).lower()
    reject_on_duplicate = str(reject_on_duplicate).lower()
    reject_on_empty_key = str(reject_on_empty_key).lower()
    reject_on_materialized_type_error = str(reject_on_materialized_type_error).lower()
    compression = extract_compression(path)
    query = f"CREATE FLEX LOCAL TEMP TABLE {flex_name}(x int) ON COMMIT PRESERVE ROWS;"
    query2 = f"""
       COPY {flex_name} 
       FROM{ingest_local} '{path}' {compression} 
       PARSER FCSVPARSER(
            type = 'traditional', 
            delimiter = '{sep}', 
            header = {header}, {header_names} 
            enclosed_by = '{quotechar}', 
            escape = '{escape}',
            record_terminator = '{record_terminator}',
            trim = {trim},
            omit_empty_keys = {omit_empty_keys},
            reject_on_duplicate = {reject_on_duplicate},
            reject_on_empty_key = {reject_on_empty_key},
            reject_on_materialized_type_error = {reject_on_materialized_type_error}) 
       NULL '{na_rep}';"""
    if genSQL:
        return [clean_query(query), clean_query(query2)]
    executeSQL(
        query=query, title="Creating flex table to identify the data types.",
    )
    executeSQL(
        query=query2, title="Parsing the data.",
    )
    result = compute_flextable_keys(flex_name)
    dtype = {}
    for column_dtype in result:
        try:
            executeSQL(
                query=f"""
                    SELECT /*+LABEL('utilities.pcsv')*/
                        (CASE 
                            WHEN "{column_dtype[0]}"=\'{na_rep}\' THEN NULL 
                            ELSE "{column_dtype[0]}" 
                         END)::{column_dtype[1]} AS "{column_dtype[0]}" 
                    FROM {flex_name} 
                    WHERE "{column_dtype[0]}" IS NOT NULL 
                    LIMIT 1000""",
                print_time_sql=False,
            )
            dtype[column_dtype[0]] = column_dtype[1]
        except:
            dtype[column_dtype[0]] = "Varchar(100)"
    drop(flex_name, method="table")
    return dtype


vHelp = help_start
# ---#
def pjson(path: str, ingest_local: bool = True):
    """
----------------------------------------------------------------------------------------
Parses a JSON file using flex tables. It will identify the columns and their
respective types.

Parameters
----------
path: str
    Absolute path where the JSON file is located.
ingest_local: bool, optional
    If set to True, the file will be ingested from the local machine.

Returns
-------
dict
    dictionary containing for each column its type.

See Also
--------
read_csv  : Ingests a CSV file into the Vertica database.
read_json : Ingests a JSON file into the Vertica database.
    """
    flex_name = gen_tmp_name(name="flex")[1:-1]
    executeSQL(
        query=f"""
            CREATE FLEX LOCAL TEMP TABLE {flex_name}
            (x int) ON COMMIT PRESERVE ROWS;""",
        title="Creating a flex table.",
    )
    path_str = path.replace("'", "''")
    local = " LOCAL" if ingest_local else ""
    executeSQL(
        query=f"""
            COPY {flex_name} FROM{local} '{path_str}' 
            PARSER FJSONPARSER();""",
        title="Ingesting the data.",
    )
    result = compute_flextable_keys(flex_name)
    dtype = {}
    for column_dtype in result:
        dtype[column_dtype[0]] = column_dtype[1]
    drop(name=flex_name, method="table")
    return dtype


# ---#
@save_verticapy_logs
def read_avro(
    path: str,
    schema: str = "",
    table_name: str = "",
    usecols: list = [],
    new_name: dict = {},
    insert: bool = False,
    reject_on_materialized_type_error: bool = False,
    flatten_maps: bool = True,
    flatten_arrays: bool = False,
    temporary_table: bool = False,
    temporary_local_table: bool = True,
    gen_tmp_table_name: bool = True,
    ingest_local: bool = True,
    genSQL: bool = False,
    materialize: bool = True,
    use_complex_dt: bool = False,
):
    """
----------------------------------------------------------------------------------------
Ingests an AVRO file using flex tables.

Parameters
----------
path: str
    Absolute path where the AVRO file is located.
schema: str, optional
    Schema where the AVRO file will be ingested.
table_name: str, optional
    Final relation name.
usecols: list, optional
    List of the AVRO parameters to ingest. The other ones will be 
    ignored. If empty all the AVRO parameters will be ingested.
new_name: dict, optional
    Dictionary of the new columns name. If the AVRO file is nested, 
    it is advised to change the final names as special characters 
    will be included.
    For example, {"param": {"age": 3, "name": Badr}, "date": 1993-03-11} 
    will create 3 columns: "param.age", "param.name" and "date". 
    You can rename these columns using the 'new_name' parameter with 
    the following dictionary:
    {"param.age": "age", "param.name": "name"}
insert: bool, optional
    If set to True, the data will be ingested to the input relation.
    The AVRO parameters must be the same as the input relation otherwise
    they will not be ingested. Also, table_name cannot be empty if this is true.
reject_on_materialized_type_error: bool, optional
    Boolean, whether to reject a data row that contains a materialized 
    column value that cannot be coerced into a compatible data type. 
    If the value is false and the type cannot be coerced, the parser 
    sets the value in that column to null.
    If the column is a strongly-typed complex type, as opposed to a 
    flexible complex type, then a type mismatch anywhere in the complex 
    type causes the entire column to be treated as a mismatch. The parser 
    does not partially load complex types.
flatten_maps: bool, optional
    Boolean, whether to flatten all AVRO maps. Key names are 
    concatenated with nested levels. This value is recursive and 
    affects all data in the load.
flatten_arrays: bool, optional
    Boolean, whether to convert lists to sub-maps with integer keys. 
    When lists are flattened, key names are concatenated as for maps. 
    Lists are not flattened by default. This value affects all data in 
    the load, including nested lists.
temporary_table: bool, optional
    If set to True, a temporary table will be created.
temporary_local_table: bool, optional
    If set to True, a temporary local table will be created. The parameter 
    'schema' must be empty, otherwise this parameter is ignored.
gen_tmp_table_name: bool, optional
    Sets the name of the temporary table. This parameter is only used when 
    the parameter 'temporary_local_table' is set to True and if the parameters 
    "table_name" and "schema" are unspecified.
ingest_local: bool, optional
    If set to True, the file will be ingested from the local machine.
genSQL: bool, optional
    If set to True, the SQL code for creating the final table is 
    generated but not executed. This is a good way to change the final
    relation types or to customize the data ingestion.
materialize: bool, optional
    If set to True, the flex table is materialized into a table.
    Otherwise, it will remain a flex table. Flex tables simplify the
    data ingestion but have worse performace compared to regular tables.
use_complex_dt: bool, optional
    Boolean, whether the input data file has complex structure.
    When this is true, most of the other parameters will be ignored.

Returns
-------
vDataFrame
    The vDataFrame of the relation.

See Also
--------
read_json : Ingests a JSON file into the Vertica database.
    """
    return read_json(
        path=path,
        schema=schema,
        table_name=table_name,
        usecols=usecols,
        new_name=new_name,
        insert=insert,
        reject_on_materialized_type_error=reject_on_materialized_type_error,
        flatten_maps=flatten_maps,
        flatten_arrays=flatten_arrays,
        temporary_table=temporary_table,
        temporary_local_table=temporary_local_table,
        gen_tmp_table_name=gen_tmp_table_name,
        ingest_local=ingest_local,
        genSQL=genSQL,
        materialize=materialize,
        use_complex_dt=use_complex_dt,
        is_avro=True,
    )


# ---#
@check_dtypes
@save_verticapy_logs
def read_csv(
    path: str,
    schema: str = "",
    table_name: str = "",
    sep: str = "",
    header: bool = True,
    header_names: list = [],
    dtype: dict = {},
    na_rep: str = "",
    quotechar: str = '"',
    escape: str = "\027",
    record_terminator: str = "\n",
    trim: bool = True,
    omit_empty_keys: bool = False,
    reject_on_duplicate: bool = False,
    reject_on_empty_key: bool = False,
    reject_on_materialized_type_error: bool = False,
    parse_nrows: int = -1,
    insert: bool = False,
    temporary_table: bool = False,
    temporary_local_table: bool = True,
    gen_tmp_table_name: bool = True,
    ingest_local: bool = True,
    genSQL: bool = False,
    materialize: bool = True,
):
    """
----------------------------------------------------------------------------------------
Ingests a CSV file using flex tables.

Parameters
----------
path: str
	Absolute path where the CSV file is located.
schema: str, optional
	Schema where the CSV file will be ingested.
table_name: str, optional
	The final relation/table name. If unspecified, the the name is set to the 
    name of the file or parent directory.
sep: str, optional
	Column separator. 
    If empty, the separator is guessed. This is only possible if the files
    are not compressed.
header: bool, optional
	If set to False, the parameter 'header_names' will be to use to name the 
	different columns.
header_names: list, optional
	List of the columns names.
dtype: dict, optional
    Dictionary of the user types. Providing a dictionary can increase 
    ingestion speed and precision; instead of parsing the file to guess 
    the different types, VerticaPy will use the input types.
na_rep: str, optional
	Missing values representation.
quotechar: str, optional
	Char which is enclosing the str values.
escape: str, optional
	Separator between each record.
record_terminator: str, optional
    A single-character value used to specify the end of a record.
trim: bool, optional
    Boolean, specifies whether to trim white space from header names and 
    key values.
omit_empty_keys: bool, optional
    Boolean, specifies how the parser handles header keys without values. 
    If true, keys with an empty value in the header row are not loaded.
reject_on_duplicate: bool, optional
    Boolean, specifies whether to ignore duplicate records (False), or to 
    reject duplicates (True). In either case, the load continues.
reject_on_empty_key: bool, optional
    Boolean, specifies whether to reject any row containing a key without a 
    value.
reject_on_materialized_type_error: bool, optional
    Boolean, specifies whether to reject any materialized column value that the 
    parser cannot coerce into a compatible data type.
parse_nrows: int, optional
	If this parameter is greater than 0. A new file of 'parse_nrows' rows
	will be created and ingested first to identify the data types. It will be
	then dropped and the entire file will be ingested. The data types identification
	will be less precise but this parameter can make the process faster if the
	file is heavy.
insert: bool, optional
	If set to True, the data will be ingested to the input relation. Be sure
	that your file has a header corresponding to the name of the relation
	columns, otherwise ingestion will fail.
temporary_table: bool, optional
    If set to True, a temporary table will be created.
temporary_local_table: bool, optional
    If set to True, a temporary local table will be created. The parameter 'schema'
    must be empty, otherwise this parameter is ignored.
gen_tmp_table_name: bool, optional
    Sets the name of the temporary table. This parameter is only used when the 
    parameter 'temporary_local_table' is set to True and if the parameters 
    "table_name" and "schema" are unspecified.
ingest_local: bool, optional
    If set to True, the file will be ingested from the local machine.
genSQL: bool, optional
    If set to True, the SQL code for creating the final table is 
    generated but not executed. This is a good way to change the final
    relation types or to customize the data ingestion.
materialize: bool, optional
    If set to True, the flex table is materialized into a table.
    Otherwise, it will remain a flex table. Flex tables simplify the
    data ingestion but have worse performace compared to regular tables.

Returns
-------
vDataFrame
	The vDataFrame of the relation.

See Also
--------
read_json : Ingests a JSON file into the Vertica database.
	"""
    from verticapy import vDataFrame

    if schema:
        temporary_local_table = False
    elif temporary_local_table:
        schema = "v_temp_schema"
    else:
        schema = "public"
    if header_names and dtype:
        warning_message = (
            "Parameters 'header_names' and 'dtype' are both defined. "
            "Only 'dtype' will be used."
        )
        warnings.warn(warning_message, Warning)
    basename = ".".join(path.split("/")[-1].split(".")[0:-1])
    if gen_tmp_table_name and temporary_local_table and not (table_name):
        table_name = gen_tmp_name(name=basename)
    assert not (temporary_table) or not (temporary_local_table), ParameterError(
        "Parameters 'temporary_table' and 'temporary_local_table' can not be both "
        "set to True."
    )
    path, sep, header_names, na_rep, quotechar, escape = (
        path.replace("'", "''"),
        sep.replace("'", "''"),
        [str(elem).replace("'", "''") for elem in header_names],
        na_rep.replace("'", "''"),
        quotechar.replace("'", "''"),
        escape.replace("'", "''"),
    )
    file_extension = path.split(".")[-1].lower()
    compression = extract_compression(path)
    if file_extension != "csv" and (compression == "UNCOMPRESSED"):
        raise ExtensionError("The file extension is incorrect !")
    multiple_files = False
    if "*" in basename:
        multiple_files = True
    if not (genSQL):
        table_name_str = table_name.replace("'", "''")
        schema_str = schema.replace("'", "''")
        result = executeSQL(
            query=f"""
                SELECT /*+LABEL('utilities.read_csv')*/
                    column_name 
               FROM columns 
               WHERE table_name = '{table_name_str}' 
                 AND table_schema = '{schema_str}' 
               ORDER BY ordinal_position""",
            title="Looking if the relation exists.",
            method="fetchall",
        )
    input_relation = format_schema_table(schema, table_name)
    if not (genSQL) and (result != []) and not (insert) and not (genSQL):
        raise NameError(f"The table {input_relation} already exists !")
    elif not (genSQL) and (result == []) and (insert):
        raise MissingRelation(f"The table {input_relation} doesn't exist !")
    else:
        if temporary_local_table:
            input_relation = f"v_temp_schema.{quote_ident(table_name)}"
        file_header = []
        path_first_file_in_folder = path
        if multiple_files and ingest_local:
            path_first_file_in_folder = get_first_file(path, "csv")
        if (
            not (header_names)
            and not (dtype)
            and (compression == "UNCOMPRESSED")
            and ingest_local
        ):
            if not (path_first_file_in_folder):
                raise ParameterError("No CSV file detected in the folder.")
            file_header = get_header_name_csv(path_first_file_in_folder, sep)
        elif not (header_names) and not (dtype) and (compression != "UNCOMPRESSED"):
            raise ParameterError(
                "The input file is compressed and parameters 'dtypes' and 'header_names'"
                " are not defined. It is impossible to read the file's header."
            )
        elif not (header_names) and not (dtype) and not (ingest_local):
            raise ParameterError(
                "The input file is in the Vertica server and parameters 'dtypes' and "
                "'header_names' are not defined. It is impossible to read the file's header."
            )
        if (header_names == []) and (header):
            if not (dtype):
                header_names = file_header
            else:
                header_names = [elem for elem in dtype]
            header_names = erase_space_start_end_in_list_values(header_names)
        elif len(file_header) > len(header_names):
            header_names += [
                f"ucol{i + len(header_names)}"
                for i in range(len(file_header) - len(header_names))
            ]
        if not (sep):
            try:
                f = open(path_first_file_in_folder, "r")
                file_str = f.readline()
                f.close()
                sep = guess_sep(file_str)
            except:
                sep = ","
        if not (materialize):
            suffix, prefix, final_relation = (
                "",
                " ON COMMIT PRESERVE ROWS;",
                input_relation,
            )
            if temporary_local_table:
                suffix = "LOCAL TEMP "
                final_relation = table_name
            elif temporary_table:
                suffix = "TEMP "
            else:
                prefix = ";"
            query = f"CREATE FLEX {suffix}TABLE {final_relation}(x int){prefix}"
            query2 = pcsv(
                path=path,
                sep=sep,
                header=header,
                header_names=header_names,
                na_rep=na_rep,
                quotechar=quotechar,
                escape=escape,
                record_terminator=record_terminator,
                trim=trim,
                omit_empty_keys=omit_empty_keys,
                reject_on_duplicate=reject_on_duplicate,
                reject_on_empty_key=reject_on_empty_key,
                reject_on_materialized_type_error=reject_on_materialized_type_error,
                ingest_local=ingest_local,
                flex_name=input_relation,
                genSQL=True,
            )[1]
            if genSQL and not (insert):
                return [clean_query(query), clean_query(query2)]
            elif genSQL:
                return [clean_query(query2)]
            if not (insert):
                executeSQL(
                    query, title="Creating the flex table.",
                )
            executeSQL(
                query2, title="Copying the data.",
            )
            return vDataFrame(table_name, schema=schema)
        if (
            (parse_nrows > 0)
            and not (insert)
            and (compression == "UNCOMPRESSED")
            and ingest_local
        ):
            f = open(path_first_file_in_folder, "r")
            path_test = path_first_file_in_folder.split(".")[-2] + "_verticapy_copy.csv"
            f2 = open(path_test, "w")
            for i in range(parse_nrows + int(header)):
                line = f.readline()
                f2.write(line)
            f.close()
            f2.close()
        else:
            path_test = path_first_file_in_folder
        query1 = ""
        if not (insert):
            if not (dtype):
                dtype = pcsv(
                    path_test,
                    sep,
                    header,
                    header_names,
                    na_rep,
                    quotechar,
                    escape,
                    ingest_local=ingest_local,
                )
            if parse_nrows > 0:
                os.remove(path_test)
            dtype_sorted = {}
            for elem in header_names:
                key = find_val_in_dict(elem, dtype, return_key=True)
                dtype_sorted[key] = dtype[key]
            query1 = create_table(
                table_name,
                dtype_sorted,
                schema,
                temporary_table,
                temporary_local_table,
                genSQL=True,
            )
        skip = " SKIP 1" if (header) else ""
        local = "LOCAL " if ingest_local else ""
        header_names_str = ", ".join([f'"{column}"' for column in header_names])
        query2 = f"""
            COPY {input_relation}({header_names_str}) 
            FROM {local}'{path}' {compression} 
            DELIMITER '{sep}' 
            NULL '{na_rep}' 
            ENCLOSED BY '{quotechar}' 
            ESCAPE AS '{escape}'{skip};"""
        if genSQL:
            if insert:
                return [clean_query(query2)]
            else:
                return [clean_query(query1), clean_query(query2)]
        else:
            if not (insert):
                executeSQL(query1, title="Creating the table.")
            executeSQL(
                query2, title="Ingesting the data.",
            )
            if (
                not (insert)
                and not (temporary_local_table)
                and verticapy.OPTIONS["print_info"]
            ):
                print(f"The table {input_relation} has been successfully created.")
            return vDataFrame(table_name, schema=schema)


# ---#
@check_minimum_version
@check_dtypes
@save_verticapy_logs
def read_file(
    path: str,
    schema: str = "",
    table_name: str = "",
    dtype: dict = {},
    unknown: str = "varchar",
    varchar_varbinary_length: int = 80,
    insert: bool = False,
    temporary_table: bool = False,
    temporary_local_table: bool = True,
    gen_tmp_table_name: bool = True,
    ingest_local: bool = False,
    genSQL: bool = False,
    max_files: int = 100,
):
    """
----------------------------------------------------------------------------------------
Inspects and ingests a file in CSV, Parquet, ORC, JSON, or Avro format.
This function uses the Vertica complex data type.
For new table creation, the file must be located in the server.

Parameters
----------
path: str
    Path to a file or glob. Valid paths include any path that is 
    valid for COPY and that uses a file format supported by this 
    function. 
    When inferring the data type, only one file will be read, even 
    if a glob specifies multiple files. However, in the case of JSON, 
    more than one file may be read to infer the data type.
schema: str, optional
    Schema in which to create the table.
table_name: str, optional
    Name of the table to create. If empty, the file name is used.
dtype: dict, optional
    Dictionary of customised data type. The predicted data types will 
    be replaced by the input data types. The dictionary must include 
    the name of the column as key and the new data type as value.
unknown: str, optional
    Type used to replace unknown data types.
varchar_varbinary_length: int, optional
    Default length of varchar and varbinary columns.
insert: bool, optional
    If set to True, the data is ingested into the input relation.
    When you set this parameter to True, most of the parameters are 
    ignored.
temporary_table: bool, optional
    If set to True, a temporary table is created.
temporary_local_table: bool, optional
    If set to True, a temporary local table is created. The parameter 
    'schema' must be empty, otherwise this parameter is ignored.
gen_tmp_table_name: bool, optional
    Sets the name of the temporary table. This parameter is only used 
    when the parameter 'temporary_local_table' is set to True and the 
    parameters "table_name" and "schema" are unspecified.
ingest_local: bool, optional
    If set to True, the file is ingested from the local machine. 
    This currently only works for data insertion.
genSQL: bool, optional
    If set to True, the SQL code for creating the final table is 
    generated but not executed. This is a good way to change the final
    relation types or to customize the data ingestion.
max_files: int, optional
    (JSON only.) If path is a glob, specifies maximum number of files 
    in path to inspect. Use this parameter to increase the amount of 
    data the function considers. This can be beneficial if you suspect 
    variation among files. Files are chosen arbitrarily from the glob.
    The default value is 100.

Returns
-------
vDataFrame
    The vDataFrame of the relation.
    """
    from verticapy import vDataFrame

    assert not (ingest_local) or insert, ParameterError(
        "Ingest local to create new relations is not yet supported for 'read_file'"
    )
    file_format = path.split(".")[-1].lower()
    compression = extract_compression(path)
    if compression != "UNCOMPRESSED":
        raise ExtensionError(
            f"Compressed files are not supported for 'read_file' function."
        )
    if file_format not in ("json", "parquet", "avro", "orc", "csv"):
        raise ExtensionError("The file extension is incorrect !")
    if file_format == "csv":
        return read_csv(
            path=path,
            schema=schema,
            table_name=table_name,
            dtype=dtype,
            genSQL=genSQL,
            insert=insert,
            temporary_table=temporary_table,
            temporary_local_table=temporary_local_table,
            gen_tmp_table_name=gen_tmp_table_name,
            ingest_local=ingest_local,
        )
    if insert:
        if not (table_name):
            raise ParameterError(
                "Parameter 'table_name' must be defined when parameter 'insert' is set to True."
            )
        if not (schema) and temporary_local_table:
            schema = "v_temp_schema"
        elif not (schema):
            schema = "public"
        input_relation = quote_ident(schema) + "." + quote_ident(table_name)
        file_format = file_format.upper()
        if file_format.lower() in ("json", "avro"):
            parser = f" PARSER F{file_format}PARSER()"
        else:
            parser = f" {file_format}"
        path = path.replace("'", "''")
        local = "LOCAL " if ingest_local else ""
        query = f"COPY {input_relation} FROM {local}'{path}'{parser};"
        if genSQL:
            return [clean_query(query)]
        executeSQL(query, title="Inserting the data.")
        return vDataFrame(table_name, schema=schema)
    if schema:
        temporary_local_table = False
    elif temporary_local_table:
        schema = "v_temp_schema"
    else:
        schema = "public"
    basename = ".".join(path.split("/")[-1].split(".")[0:-1])
    if gen_tmp_table_name and temporary_local_table and not (table_name):
        table_name = gen_tmp_name(name=basename)
    if not (table_name):
        table_name = basename
    sql = (
        f"SELECT INFER_TABLE_DDL ('{path}' USING PARAMETERS "
        f"format='{file_format}', table_name='y_verticapy', "
        "table_schema='x_verticapy', table_type='native', "
        "with_copy_statement=true, one_line_result=true, "
        f"max_files={max_files}, max_candidates=1);"
    )
    result = executeSQL(
        sql, title="Generating the CREATE and COPY statement.", method="fetchfirstelem",
    )
    result = result.replace("UNKNOWN", unknown)
    result = "create" + "create".join(result.split("create")[1:])
    relation = format_schema_table(schema, table_name)
    if temporary_local_table:
        create_statement = f"CREATE LOCAL TEMPORARY TABLE {quote_ident(table_name)}"
    else:
        if not (schema):
            schema = "public"
        if temporary_table:
            create_statement = f"CREATE TEMPORARY TABLE {relation}"
        else:
            create_statement = f"CREATE TABLE {relation}"
    result = result.replace(
        'create table "x_verticapy"."y_verticapy"', create_statement
    )
    if ";\n copy" in result:
        result = result.split(";\n copy")
        if temporary_local_table:
            result[0] += " ON COMMIT PRESERVE ROWS;"
        else:
            result[0] += ";"
        result[1] = "copy" + result[1].replace('"x_verticapy"."y_verticapy"', relation,)
    else:
        if temporary_local_table:
            end = result.split(")")[-1]
            result = result.split(")")[0:-1] + ") ON COMMIT PRESERVE ROWS" + end
        result = [result]
    if varchar_varbinary_length != 80:
        result[0] = (
            result[0]
            .replace(" varchar", f" varchar({varchar_varbinary_length})")
            .replace(" varbinary", f" varbinary({varchar_varbinary_length})")
        )
    for col in dtype:
        extract_col_dt = extract_col_dt_from_query(result[0], col)
        if extract_col_dt is None:
            warning_message = f"The column '{col}' was not found.\nIt will be skipped."
            warnings.warn(warning_message, Warning)
        else:
            column, ctype = extract_col_dt
            result[0] = result[0].replace(
                column + " " + ctype, column + " " + dtype[col]
            )
    if genSQL:
        for idx in range(len(result)):
            result[idx] = clean_query(result[idx])
        return result
    if len(result) == 1:
        executeSQL(
            result, title="Creating the table and ingesting the data.",
        )
    else:
        executeSQL(
            result[0], title="Creating the table.",
        )
        try:
            executeSQL(
                result[1], title="Ingesting the data.",
            )
        finally:
            drop(f'"{schema}"."{table_name}"', method="table")
    return vDataFrame(input_relation=table_name, schema=schema)


# ---#
@check_dtypes
@save_verticapy_logs
def read_json(
    path: str,
    schema: str = "",
    table_name: str = "",
    usecols: list = [],
    new_name: dict = {},
    insert: bool = False,
    start_point: str = None,
    record_terminator: str = None,
    suppress_nonalphanumeric_key_chars: bool = False,
    reject_on_materialized_type_error: bool = False,
    reject_on_duplicate: bool = False,
    reject_on_empty_key: bool = False,
    flatten_maps: bool = True,
    flatten_arrays: bool = False,
    temporary_table: bool = False,
    temporary_local_table: bool = True,
    gen_tmp_table_name: bool = True,
    ingest_local: bool = True,
    genSQL: bool = False,
    materialize: bool = True,
    use_complex_dt: bool = False,
    is_avro: bool = False,
):
    """
----------------------------------------------------------------------------------------
Ingests a JSON file using flex tables.

Parameters
----------
path: str
	Absolute path where the JSON file is located.
schema: str, optional
	Schema where the JSON file will be ingested.
table_name: str, optional
	Final relation name.
usecols: list, optional
	List of the JSON parameters to ingest. The other ones will be 
    ignored. If empty all the JSON parameters will be ingested.
new_name: dict, optional
	Dictionary of the new columns name. If the JSON file is nested, 
    it is advised to change the final names as special characters 
    will be included.
	For example, {"param": {"age": 3, "name": Badr}, "date": 1993-03-11} 
    will create 3 columns: "param.age", "param.name" and "date". 
    You can rename these columns using the 'new_name' parameter with 
    the following dictionary:
	{"param.age": "age", "param.name": "name"}
insert: bool, optional
	If set to True, the data will be ingested to the input relation.
    The JSON parameters must be the same as the input relation otherwise
    they will not be ingested. Also, table_name cannot be empty if this is true.
start_point: str, optional
    String, name of a key in the JSON load data at which to begin parsing. 
    The parser ignores all data before the start_point value. 
    The value is loaded for each object in the file. The parser processes 
    data after the first instance, and up to the second, ignoring any 
    remaining data.
record_terminator: str, optional
    When set, any invalid JSON records are skipped and parsing continues 
    with the next record. 
    Records must be terminated uniformly. For example, if your input file 
    has JSON records terminated by newline characters, set this parameter 
    to '\n'). 
    If any invalid JSON records exist, parsing continues after the next 
    record_terminator.
    Even if the data does not contain invalid records, specifying an 
    explicit record terminator can improve load performance by allowing 
    cooperative parse and apportioned load to operate more efficiently.
    When you omit this parameter, parsing ends at the first invalid JSON 
    record.
suppress_nonalphanumeric_key_chars: bool, optional
    Boolean, whether to suppress non-alphanumeric characters in JSON 
    key values. The parser replaces these characters with an underscore 
    (_) when this parameter is true.
reject_on_materialized_type_error: bool, optional
    Boolean, whether to reject a data row that contains a materialized 
    column value that cannot be coerced into a compatible data type. 
    If the value is false and the type cannot be coerced, the parser 
    sets the value in that column to null.
    If the column is a strongly-typed complex type, as opposed to a 
    flexible complex type, then a type mismatch anywhere in the complex 
    type causes the entire column to be treated as a mismatch. The parser 
    does not partially load complex types.
reject_on_duplicate: bool, optional
    Boolean, whether to ignore duplicate records (false), or to 
    reject duplicates (true). In either case, the load continues.
reject_on_empty_key: bool, optional
    Boolean, whether to reject any row containing a field key 
    without a value.
flatten_maps: bool, optional
    Boolean, whether to flatten sub-maps within the JSON data, separating map levels 
    with a period (.). This value affects all data in the load, including nested maps.
flatten_arrays: bool, optional
    Boolean, whether to convert lists to sub-maps with integer keys. 
    When lists are flattened, key names are concatenated as for maps. 
    Lists are not flattened by default. This value affects all data in 
    the load, including nested lists.
temporary_table: bool, optional
    If set to True, a temporary table will be created.
temporary_local_table: bool, optional
    If set to True, a temporary local table will be created. The parameter 
    'schema' must be empty, otherwise this parameter is ignored.
gen_tmp_table_name: bool, optional
    Sets the name of the temporary table. This parameter is only used when 
    the parameter 'temporary_local_table' is set to True and if the parameters 
    "table_name" and "schema" are unspecified.
ingest_local: bool, optional
    If set to True, the file will be ingested from the local machine.
genSQL: bool, optional
    If set to True, the SQL code for creating the final table is 
    generated but not executed. This is a good way to change the final
    relation types or to customize the data ingestion.
materialize: bool, optional
    If set to True, the flex table is materialized into a table.
    Otherwise, it will remain a flex table. Flex tables simplify the
    data ingestion but have worse performace compared to regular tables.
use_complex_dt: bool, optional
    Boolean, whether the input data file has complex structure.
    When this is true, most of the other parameters will be ignored.

Returns
-------
vDataFrame
	The vDataFrame of the relation.

See Also
--------
read_csv : Ingests a CSV file into the Vertica database.
	"""
    from verticapy import vDataFrame

    if use_complex_dt:
        assert not (new_name), ParameterError(
            "You cannot use the parameter " "new_name" " with " "use_complex_dt" "."
        )
        if is_avro:
            max_files = 1
        elif ("*" in path) and ingest_local:
            dirname = os.path.dirname(path)
            all_files = os.listdir(dirname)
            max_files = sum(1 for x in all_files if x.endswith(".json"))
        else:
            max_files = 1000
        return read_file(
            path=path,
            schema=schema,
            table_name=table_name,
            insert=insert,
            temporary_table=temporary_table,
            temporary_local_table=temporary_local_table,
            gen_tmp_table_name=gen_tmp_table_name,
            ingest_local=ingest_local,
            genSQL=genSQL,
            max_files=max_files,
        )
    if schema:
        temporary_local_table = False
    elif temporary_local_table:
        schema = "v_temp_schema"
    else:
        schema = "public"
    assert not (temporary_table) or not (temporary_local_table), ParameterError(
        "Parameters 'temporary_table' and 'temporary_local_table' can not be both set to True."
    )
    file_extension = path.split(".")[-1].lower()
    compression = extract_compression(path)
    if (
        (file_extension not in ("json",) and not (is_avro))
        or (file_extension not in ("avro",) and (is_avro))
    ) and (compression == "UNCOMPRESSED"):
        raise ExtensionError("The file extension is incorrect !")
    basename = ".".join(path.split("/")[-1].split(".")[0:-1])
    if gen_tmp_table_name and temporary_local_table and not (table_name):
        table_name = gen_tmp_name(name=basename)
    if not (table_name):
        table_name = basename
    if is_avro:
        label = "read_avro"
        parser = "FAVROPARSER"
    else:
        label = "read_json"
        parser = "FJSONPARSER"
    if not (genSQL):
        table_name_str = table_name.replace("'", "''")
        schema_str = schema.replace("'", "''")
        column_name = executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('utilities.{label}')*/ 
                    column_name,
                    data_type 
                FROM columns 
                WHERE table_name = '{table_name_str}' 
                  AND table_schema = '{schema_str}' 
                ORDER BY ordinal_position""",
            title="Looking if the relation exists.",
            method="fetchall",
        )
    input_relation = format_schema_table(schema, table_name)
    if not (genSQL) and (column_name != []) and not (insert):
        raise NameError(f"The table {input_relation} already exists !")
    elif not (genSQL) and (column_name == []) and (insert):
        raise MissingRelation(f"The table {input_relation} doesn't exist !")
    else:
        if temporary_local_table:
            input_relation = quote_ident(table_name)
        all_queries = []
        if not (materialize):
            suffix, prefix = "", "ON COMMIT PRESERVE ROWS;"
            if temporary_local_table:
                suffix = "LOCAL TEMP "
            elif temporary_table:
                suffix = "TEMP "
            else:
                prefix = ";"
            query = f"""
                CREATE FLEX {suffix}TABLE 
                {input_relation}(x int){prefix}"""
        else:
            flex_name = gen_tmp_name(name="flex")[1:-1]
            query = f"""
                CREATE FLEX LOCAL TEMP TABLE {flex_name}(x int) 
                ON COMMIT PRESERVE ROWS;"""
        if not (insert):
            all_queries += [clean_query(query)]
        options = []
        if start_point and not (is_avro):
            options += [f"start_point='{start_point}'"]
        if record_terminator and not (is_avro):
            prefix = ""
            if "\\" in record_terminator.__repr__():
                prefix = "E"
            options += [f"record_terminator={prefix}'{record_terminator}'"]
        if suppress_nonalphanumeric_key_chars and not (is_avro):
            options += ["suppress_nonalphanumeric_key_chars=true"]
        else:
            options += ["suppress_nonalphanumeric_key_chars=false"]
        if reject_on_materialized_type_error:
            assert materialize, ParameterError(
                "When using complex data types the table has to "
                "be materialized. Set materialize to True"
            )
            options += ["reject_on_materialized_type_error=true"]
        else:
            options += ["reject_on_materialized_type_error=false"]
        if reject_on_duplicate and not (is_avro):
            options += ["reject_on_duplicate=true"]
        else:
            options += ["reject_on_duplicate=false"]
        if reject_on_empty_key and not (is_avro):
            options += ["reject_on_empty_key=true"]
        else:
            options += ["reject_on_empty_key=false"]
        if flatten_arrays:
            options += ["flatten_arrays=true"]
        else:
            options += ["flatten_arrays=false"]
        if flatten_maps:
            options += ["flatten_maps=true"]
        else:
            options += ["flatten_maps=false"]
        materialize_str = flex_name if (materialize) else input_relation
        local = " LOCAL" if ingest_local else ""
        path_str = path.replace("'", "''")
        query2 = f"""
            COPY {materialize_str} 
            FROM{local} '{path_str}' {compression} 
            PARSER {parser}({", ".join(options)});"""
        all_queries = all_queries + [clean_query(query2)]
        if genSQL and insert and not (materialize):
            return [clean_query(query2)]
        elif genSQL and not (materialize):
            return all_queries
        if not (insert):
            executeSQL(
                query, title="Creating flex table.",
            )
        executeSQL(
            query2, title="Ingesting the data in the flex table.",
        )
        if not (materialize):
            return vDataFrame(table_name, schema=schema)
        result = compute_flextable_keys(flex_name)
        dtype = {}
        for column_dtype in result:
            try:
                executeSQL(
                    query=f"""
                        SELECT 
                            /*+LABEL('utilities.{label}')*/ 
                            \"{column_dtype[0]}\"::{column_dtype[1]} 
                        FROM {flex_name} 
                        LIMIT 1000""",
                    print_time_sql=False,
                )
                dtype[column_dtype[0]] = column_dtype[1]
            except:
                dtype[column_dtype[0]] = "Varchar(100)"
        if not (insert):
            cols = (
                [column for column in dtype]
                if not (usecols)
                else [column for column in usecols]
            )
            for i, column in enumerate(cols):
                column_str = column.replace('"', "")
                if column in new_name:
                    cols[i] = f'"{column_str}"::{dtype[column]} AS "{new_name[column]}"'
                else:
                    cols[i] = f'"{column_str}"::{dtype[column]}'
            if temporary_local_table:
                suffix = "LOCAL TEMPORARY "
            elif temporary_table:
                suffix = "TEMPORARY "
            else:
                suffix = ""
            on_commit = " ON COMMIT PRESERVE ROWS" if suffix else ""
            query3 = f"""
                CREATE {suffix}TABLE {input_relation}{on_commit} AS 
                    SELECT 
                        /*+LABEL('utilities.{label}')*/ 
                        {", ".join(cols)} 
                    FROM {flex_name}"""
            all_queries = all_queries + [clean_query(query3)]
            if genSQL:
                return all_queries
            executeSQL(
                query3, title="Creating table.",
            )
            if not (temporary_local_table) and verticapy.OPTIONS["print_info"]:
                print(f"The table {input_relation} has been successfully created.")
        else:
            column_name_dtype = {}
            for elem in column_name:
                column_name_dtype[elem[0]] = elem[1]
            final_cols = {}
            for column in column_name_dtype:
                final_cols[column] = None
            for column in column_name_dtype:
                if column in dtype:
                    final_cols[column] = column
                else:
                    for col in new_name:
                        if new_name[col] == column:
                            final_cols[column] = col
            final_transformation = []
            for column in final_cols:
                if final_cols[column] == None:
                    final_transformation += [f'NULL AS "{column}"']
                else:
                    final_transformation += [
                        f'"{final_cols}"::{column_name_dtype[column]} AS "{column}"'
                    ]
            query = f"""
                INSERT 
                    /*+LABEL('utilities.{label}')*/ 
                INTO {input_relation} 
                SELECT 
                    {", ".join(final_transformation)} 
                FROM {flex_name}"""
            if genSQL:
                return [clean_query(query)]
            executeSQL(
                query, title="Inserting data into table.",
            )
        drop(name=flex_name, method="table")
        return vDataFrame(table_name, schema=schema)


# ---#
@check_dtypes
@save_verticapy_logs
def read_shp(
    path: str, schema: str = "public", table_name: str = "",
):
    """
----------------------------------------------------------------------------------------
Ingests a SHP file. For the moment, only files located in the Vertica server 
can be ingested.

Parameters
----------
path: str
    Absolute path where the SHP file is located.
schema: str, optional
    Schema where the SHP file will be ingested.
table_name: str, optional
    Final relation name.

Returns
-------
vDataFrame
    The vDataFrame of the relation.
    """
    file = path.split("/")[-1]
    file_extension = file[-3 : len(file)]
    if file_extension != "shp":
        raise ExtensionError("The file extension is incorrect !")
    query = (
        f"SELECT /*+LABEL('utilities.read_shp')*/ STV_ShpCreateTable(USING PARAMETERS file='{path}')"
        " OVER() AS create_shp_table;"
    )
    result = executeSQL(query, title="Getting SHP definition.", method="fetchall")
    if not (table_name):
        table_name = file[:-4]
    result[0] = [f'CREATE TABLE "{schema}"."{table_name}"(']
    result = [elem[0] for elem in result]
    result = "".join(result)
    executeSQL(result, title="Creating the relation.")
    query = (
        f'COPY "{schema}"."{table_name}" WITH SOURCE STV_ShpSource(file=\'{path}\')'
        " PARSER STV_ShpParser();"
    )
    executeSQL(query, title="Ingesting the data.")
    print(f'The table "{schema}"."{table_name}" has been successfully created.')
    from verticapy import vDataFrame

    return vDataFrame(table_name, schema=schema)


# ---#
@check_dtypes
@save_verticapy_logs
def readSQL(query: str, time_on: bool = False, limit: int = 100):
    """
    ----------------------------------------------------------------------------------------
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
    while len(query) > 0 and query[-1] in (";", " "):
        query = query[:-1]
    if verticapy.OPTIONS["count_on"]:
        count = executeSQL(
            f"""SELECT 
                    /*+LABEL('utilities.readSQL')*/ COUNT(*) 
                FROM ({query}) VERTICAPY_SUBTABLE""",
            method="fetchfirstelem",
            print_time_sql=False,
        )
    else:
        count = -1
    sql_on_init = verticapy.OPTIONS["sql_on"]
    time_on_init = verticapy.OPTIONS["time_on"]
    try:
        verticapy.OPTIONS["time_on"] = time_on
        verticapy.OPTIONS["sql_on"] = False
        try:
            result = to_tablesample(f"{query} LIMIT {limit}")
        except:
            result = to_tablesample(query)
    finally:
        verticapy.OPTIONS["time_on"] = time_on_init
        verticapy.OPTIONS["sql_on"] = sql_on_init
    result.count = count
    if verticapy.OPTIONS["percent_bar"]:
        vdf = vDataFrameSQL(f"({query}) VERTICAPY_SUBTABLE")
        percent = vdf.agg(["percent"]).transpose().values
        for column in result.values:
            result.dtype[column] = vdf[column].ctype()
            result.percent[column] = percent[vdf.format_colnames(column)][0]
    return result


# ---#
def save_to_query_profile(
    name: str,
    path: str = "",
    json_dict: dict = {},
    query_label: str = "verticapy_json",
    return_query: bool = False,
    add_identifier: bool = True,
):
    """
----------------------------------------------------------------------------------------
Saves information about the specified VerticaPy method to the QUERY_PROFILES 
table in the Vertica database. It is used to collect usage statistics on 
methods and their parameters. This function generates a JSON string.

Parameters
----------
name: str
    Name of the method.
path: str, optional
    Path to the function or method.
json_dict: dict, optional
    Dictionary of the different parameters to store.
query_label: str, optional
    Name to give to the identifier in the query profile table. If 
    unspecified, the name of the method is used.
return_query: bool, optional
    If set to True, the query is returned.
add_identifier: bool, optional
    If set to True, the VerticaPy identifier is added to the generated json.

Returns
-------
bool
    True if the operation succeeded, False otherwise.
    """
    if not (verticapy.OPTIONS["save_query_profile"]) or (
        isinstance(verticapy.OPTIONS["save_query_profile"], list)
        and name not in verticapy.OPTIONS["save_query_profile"]
    ):
        return False
    try:

        def dict_to_json_string(
            name: str = "",
            path: str = "",
            json_dict: dict = {},
            add_identifier: bool = False,
        ):
            from verticapy import vDataFrame
            from verticapy.learn.vmodel import vModel

            json = "{"
            if name:
                json += f'"verticapy_fname": "{name}", '
            if path:
                json += f'"verticapy_fpath": "{path}", '
            if add_identifier:
                json += f'"verticapy_id": "{verticapy.OPTIONS["identifier"]}", '
            for key in json_dict:
                json += f'"{key}": '
                if isinstance(json_dict[key], bool):
                    json += "true" if json_dict[key] else "false"
                elif isinstance(json_dict[key], (float, int)):
                    json += str(json_dict[key])
                elif json_dict[key] is None:
                    json += "null"
                elif isinstance(json_dict[key], vDataFrame):
                    json_dict_str = json_dict[key].__genSQL__().replace('"', '\\"')
                    json += f'"{json_dict_str}"'
                elif isinstance(json_dict[key], vModel):
                    json += f'"{json_dict[key].type}"'
                elif isinstance(json_dict[key], dict):
                    json += dict_to_json_string(json_dict=json_dict[key])
                elif isinstance(json_dict[key], list):
                    json_dict_str = ";".join([str(item) for item in json_dict[key]])
                    json += f'"{json_dict_str}"'
                else:
                    json_dict_str = str(json_dict[key]).replace('"', '\\"')
                    json += f'"{json_dict_str}"'
                json += ", "
            json = json[:-2] + "}"
            return json

        query_label_str = query_label.replace("'", "''")
        dict_to_json_string_str = dict_to_json_string(
            name, path, json_dict, add_identifier
        ).replace("'", "''")
        query = f"SELECT /*+LABEL('{query_label_str}')*/ '{dict_to_json_string_str}'"
        if return_query:
            return query
        executeSQL(
            query=query,
            title="Sending query to save the information in query profile table.",
            print_time_sql=False,
        )
        return True
    except:
        return False


# ---#
@check_dtypes
def set_option(option: str, value: Union[bool, int, str, list] = None):
    """
    ----------------------------------------------------------------------------------------
    Sets VerticaPy options.

    Parameters
    ----------
    option: str
        Option to use.
        cache              : bool
            If set to True, the vDataFrame will save in memory the computed
            aggregations.
        colors             : list
            List of the colors used to draw the graphics.
        color_style        : str
            Style used to color the graphics, one of the following:
            "rgb", "sunset", "retro", "shimbg", "swamp", "med", "orchid", 
            "magenta", "orange", "vintage", "vivid", "berries", "refreshing", 
            "summer", "tropical", "india", "default".
        count_on           : bool
            If set to True, the total number of rows in vDataFrames and tablesamples is  
            computed and displayed in the footer (if footer_on is True).
        footer_on          : bool
            If set to True, vDataFrames and tablesamples show a footer that includes information 
            about the displayed rows and columns.
        interactive        : bool
            If set to True, verticaPy outputs will be displayed on interactive tables. 
        max_columns        : int
            Maximum number of columns to display. If the parameter is incorrect, 
            nothing is changed.
        max_rows           : int
            Maximum number of rows to display. If the parameter is incorrect, 
            nothing is changed.
        mode               : str
            How to display VerticaPy outputs.
                full  : VerticaPy regular display mode.
                light : Minimalist display mode.
        overwrite_model: bool
            If set to True and you try to train a model with an existing name. 
            It will be automatically overwritten.
        percent_bar        : bool
            If set to True, it displays the percent of non-missing values.
        print_info         : bool
            If set to True, information will be printed each time the vDataFrame 
            is modified.
        random_state       : int
            Integer used to seed the random number generation in VerticaPy.
        save_query_profile : str / list / bool
            If set to "all" or True, all function calls are stored in the query 
            profile table. This makes it possible to differentiate the VerticaPy 
            logs from the Vertica logs.
            You can also provide a list of specific methods to store. For example: 
            if you specify ["corr", "describe"], only the logs associated with 
            those two methods are stored. 
            If set to False, this functionality is deactivated.
        sql_on             : bool
            If set to True, displays all the SQL queries.
        temp_schema        : str
            Specifies the temporary schema that certain methods/functions use to 
            create intermediate objects, if needed. 
        time_on            : bool
            If set to True, displays all the SQL queries elapsed time.
        tqdm               : bool
            If set to True, a loading bar is displayed when using iterative 
            functions.
    value: object, optional
        New value of option.
    """
    raise_error_if_not_in(
        "option",
        option,
        [
            "cache",
            "colors",
            "color_style",
            "interactive",
            "count_on",
            "footer_on",
            "max_columns",
            "max_rows",
            "mode",
            "overwrite_model",
            "percent_bar",
            "print_info",
            "random_state",
            "save_query_profile",
            "sql_on",
            "temp_schema",
            "time_on",
            "tqdm",
        ],
    )
    wrong_value = False
    if option == "colors":
        if isinstance(value, list):
            verticapy.OPTIONS["colors"] = [str(elem) for elem in value]
        else:
            wrong_value = True
    elif option == "color_style":
        raise_error_if_not_in(
            "value",
            value,
            [
                "rgb",
                "sunset",
                "retro",
                "shimbg",
                "swamp",
                "med",
                "orchid",
                "magenta",
                "orange",
                "vintage",
                "vivid",
                "berries",
                "refreshing",
                "summer",
                "tropical",
                "india",
                "default",
            ],
        )
        if isinstance(value, str):
            verticapy.OPTIONS["color_style"] = value
            verticapy.OPTIONS["colors"] = []
        else:
            wrong_value = True
    elif option == "max_columns":
        if isinstance(value, int) and value > 0:
            verticapy.OPTIONS["max_columns"] = int(value)
        else:
            wrong_value = True
    elif option == "max_rows":
        if isinstance(value, int) and value >= 0:
            verticapy.OPTIONS["max_rows"] = int(value)
        else:
            wrong_value = True
    elif option == "mode":
        raise_error_if_not_in("value", value, ["light", "full"])
        verticapy.OPTIONS["mode"] = value
    elif option == "random_state":
        if isinstance(value, int) and (value < 0):
            raise ParameterError("Random State Value must be positive.")
        if isinstance(value, int):
            verticapy.OPTIONS["random_state"] = value
        elif value == None:
            verticapy.OPTIONS["random_state"] = None
        else:
            wrong_value = True
    elif option in (
        "print_info",
        "sql_on",
        "time_on",
        "count_on",
        "cache",
        "footer_on",
        "tqdm",
        "overwrite_model",
        "percent_bar",
        "interactive",
    ):
        if value in (True, False, None):
            verticapy.OPTIONS[option] = value
        else:
            wrong_value = True
    elif option == "save_query_profile":
        if value == "all":
            value = True
        elif isinstance(value, (bool, list)):
            pass
        else:
            wrong_value = True
        if not (wrong_value):
            verticapy.OPTIONS[option] = value
    elif option == "temp_schema":
        if isinstance(value, str):
            value_str = value.replace("'", "''")
            query = f"""
                SELECT /*+LABEL('utilities.set_option')*/
                  schema_name 
               FROM v_catalog.schemata 
               WHERE schema_name = '{value_str}' LIMIT 1;"""
            res = executeSQL(
                query, title="Checking if the schema exists.", method="fetchrow"
            )
            if res:
                verticapy.OPTIONS["temp_schema"] = str(value)
            else:
                raise ParameterError(f"The schema '{value}' could not be found.")
        else:
            wrong_value = True
    else:
        raise ParameterError(f"Option '{option}' does not exist.")
    if wrong_value:
        warning_message = "The parameter value is incorrect. Nothing was changed."
        warnings.warn(warning_message, Warning)


# ---#
class tablesample:
    """
----------------------------------------------------------------------------------------
The tablesample is the transition from 'Big Data' to 'Small Data'. 
This object allows you to conveniently display your results without any  
dependencies on any other module. It stores the aggregated result in memory
which can then be transformed into a pandas.DataFrame or vDataFrame.

Parameters
----------
values: dict, optional
	Dictionary of columns (keys) and their values. The dictionary must be
	similar to the following one:
	{"column1": [val1, ..., valm], ... "columnk": [val1, ..., valm]}
dtype: dict, optional
	Columns data types.
count: int, optional
	Number of elements if we had to load the entire dataset. It is used 
	only for rendering purposes.
offset: int, optional
	Number of elements that were skipped if we had to load the entire
	dataset. It is used only for rendering purposes.
percent: dict, optional
    Dictionary of missing values (Used to display the percent bars)
max_columns: int, optional
    Maximum number of columns to display.

Attributes
----------
The tablesample attributes are the same as the parameters.
	"""

    #
    # Special Methods
    #
    # ---#
    @check_dtypes
    def __init__(
        self,
        values: dict = {},
        dtype: dict = {},
        count: int = 0,
        offset: int = 0,
        percent: dict = {},
        max_columns: int = -1,
    ):
        self.values = values
        self.dtype = dtype
        self.count = count
        self.offset = offset
        self.percent = percent
        self.max_columns = max_columns
        for column in values:
            if column not in dtype:
                self.dtype[column] = "undefined"

    # ---#
    def __iter__(self):
        return (elem for elem in self.values)

    # ---#
    def __getitem__(self, key):
        return find_val_in_dict(key, self.values)

    # ---#
    def _repr_html_(self, interactive=False):
        if len(self.values) == 0:
            return ""
        n = len(self.values)
        dtype = self.dtype
        max_columns = (
            self.max_columns
            if self.max_columns > 0
            else verticapy.OPTIONS["max_columns"]
        )
        if n < max_columns:
            data_columns = [[column] + self.values[column] for column in self.values]
        else:
            k = int(max_columns / 2)
            columns = [elem for elem in self.values]
            values0 = [[columns[i]] + self.values[columns[i]] for i in range(k)]
            values1 = [["..." for i in range(len(self.values[columns[0]]) + 1)]]
            values2 = [
                [columns[i]] + self.values[columns[i]]
                for i in range(n - max_columns + k, n)
            ]
            data_columns = values0 + values1 + values2
            dtype["..."] = "undefined"
        percent = self.percent
        for elem in self.values:
            if elem not in percent and (elem != "index"):
                percent = {}
                break
        formatted_text = ""
        # get interactive table if condition true
        if verticapy.OPTIONS["interactive"] or interactive:
            formatted_text = datatables_repr(
                data_columns,
                repeat_first_column=("index" in self.values),
                offset=self.offset,
                dtype=dtype,
            )
        else:
            formatted_text = print_table(
                data_columns,
                is_finished=(self.count <= len(data_columns[0]) + self.offset),
                offset=self.offset,
                repeat_first_column=("index" in self.values),
                return_html=True,
                dtype=dtype,
                percent=percent,
            )
        if verticapy.OPTIONS["footer_on"]:
            formatted_text += '<div style="margin-top:6px; font-size:1.02em">'
            if (self.offset == 0) and (len(data_columns[0]) - 1 == self.count):
                rows = self.count
            else:
                start, end = self.offset + 1, len(data_columns[0]) - 1 + self.offset
                if start > end:
                    rows = f"0 of {self.count}" if (self.count > 0) else "0"
                else:
                    of = f" of {self.count}" if (self.count > 0) else ""
                    rows = f"{start}-{end}{of}"
            if len(self.values) == 1:
                column = list(self.values.keys())[0]
                if self.offset > self.count:
                    formatted_text += (
                        f"<b>Column:</b> {column} | "
                        f"<b>Type:</b> {self.dtype[column]}"
                    )
                else:
                    formatted_text += (
                        f"<b>Rows:</b> {rows} | <b>Column:</b> {column} "
                        f"| <b>Type:</b> {self.dtype[column]}"
                    )
            else:
                if self.offset > self.count:
                    formatted_text += f"<b>Columns:</b> {n}"
                else:
                    formatted_text += f"<b>Rows:</b> {rows} | <b>Columns:</b> {n}"
            formatted_text += "</div>"
        return formatted_text

    # ---#
    def __repr__(self):
        if len(self.values) == 0:
            return ""
        n = len(self.values)
        dtype = self.dtype
        max_columns = (
            self.max_columns
            if self.max_columns > 0
            else verticapy.OPTIONS["max_columns"]
        )
        if n < max_columns:
            data_columns = [[column] + self.values[column] for column in self.values]
        else:
            k = int(max_columns / 2)
            columns = [elem for elem in self.values]
            values0 = [[columns[i]] + self.values[columns[i]] for i in range(k)]
            values1 = [["..." for i in range(len(self.values[columns[0]]) + 1)]]
            values2 = [
                [columns[i]] + self.values[columns[i]]
                for i in range(n - max_columns + k, n)
            ]
            data_columns = values0 + values1 + values2
            dtype["..."] = "undefined"
        formatted_text = print_table(
            data_columns,
            is_finished=(self.count <= len(data_columns[0]) + self.offset),
            offset=self.offset,
            repeat_first_column=("index" in self.values),
            return_html=False,
            dtype=dtype,
            percent=self.percent,
        )
        start, end = self.offset + 1, len(data_columns[0]) - 1 + self.offset
        if (self.offset == 0) and (len(data_columns[0]) - 1 == self.count):
            rows = self.count
        else:
            if start > end:
                rows = f"0 of {self.count}" if (self.count > 0) else "0"
            else:
                count_str = f" of {self.count}" if (self.count > 0) else ""
                rows = f"{start}-{end}{count_str}"
        if len(self.values) == 1:
            column = list(self.values.keys())[0]
            if self.offset > self.count:
                formatted_text += f"Column: {column} | Type: {self.dtype[column]}"
            else:
                formatted_text += (
                    f"Rows: {rows} | Column: {column} | Type: {self.dtype[column]}"
                )
        else:
            if self.offset > self.count:
                formatted_text += f"Columns: {n}"
            else:
                formatted_text += f"Rows: {rows} | Columns: {n}"
        return formatted_text

    #
    # Methods
    #
    # ---#
    @check_dtypes
    def append(self, tbs):
        """
        ----------------------------------------------------------------------------------------
        Appends the input tablesample to a target tablesample.

        Parameters
        ----------
        tbs: tablesample
            Tablesample to append.

        Returns
        -------
        tablesample
            self
        """
        assert isinstance(tbs, tablesample), ParameterError(
            "tablesamples can only be appended to another tablesample."
        )
        n1, n2 = self.shape()[0], tbs.shape()[0]
        assert n1 == n2, ParameterError(
            "The input and target tablesamples must have the same number of columns."
            f" Expected {n1}, Found {n2}."
        )
        cols1, cols2 = [col for col in self.values], [col for col in tbs.values]
        for idx in range(n1):
            self.values[cols1[idx]] += tbs.values[cols2[idx]]
        return self

    # ---#
    def decimal_to_float(self):
        """
    ----------------------------------------------------------------------------------------
    Converts all the tablesample's decimals to floats.

    Returns
    -------
    tablesample
        self
        """
        for elem in self.values:
            if elem != "index":
                for i in range(len(self.values[elem])):
                    if isinstance(self.values[elem][i], decimal.Decimal):
                        self.values[elem][i] = float(self.values[elem][i])
        return self

    # ---#
    @check_dtypes
    def merge(self, tbs):
        """
        ----------------------------------------------------------------------------------------
        Merges the input tablesample to a target tablesample.

        Parameters
        ----------
        tbs: tablesample
            Tablesample to merge.

        Returns
        -------
        tablesample
            self
        """
        assert isinstance(tbs, tablesample), ParameterError(
            "tablesamples can only be merged with other tablesamples."
        )
        n1, n2 = self.shape()[1], tbs.shape()[1]
        assert n1 == n2, ParameterError(
            "The input and target tablesamples must have the same number of rows."
            f" Expected {n1}, Found {n2}."
        )
        for col in tbs.values:
            if col != "index":
                if col not in self.values:
                    self.values[col] = []
                self.values[col] += tbs.values[col]
        return self

    # ---#
    def shape(self):
        """
    ----------------------------------------------------------------------------------------
    Computes the tablesample shape.

    Returns
    -------
    tuple
        (number of columns, number of rows)
        """
        cols = [col for col in self.values]
        n, m = len(cols), len(self.values[cols[0]])
        return (n, m)

    # ---#
    @check_dtypes
    def sort(self, column: str, desc: bool = False):
        """
        ----------------------------------------------------------------------------------------
        Sorts the tablesample using the input column.

        Parameters
        ----------
        column: str, optional
            Column used to sort the data.
        desc: bool, optional
            If set to True, the result is sorted in descending order.

        Returns
        -------
        tablesample
            self
        """
        column = column.replace('"', "").lower()
        columns = [col for col in self.values]
        idx = None
        for i, col in enumerate(columns):
            col_tmp = col.replace('"', "").lower()
            if column == col_tmp:
                idx = i
                column = col
        if idx is None:
            raise MissingColumn(f"The Column '{column}' doesn't exist.")
        n, sort = len(self[column]), []
        for i in range(n):
            tmp_list = []
            for col in columns:
                tmp_list += [self[col][i]]
            sort += [tmp_list]
        sort.sort(key=lambda tup: tup[idx], reverse=desc)
        for i, col in enumerate(columns):
            self.values[col] = [sort[j][i] for j in range(n)]
        return self

    # ---#
    def transpose(self):
        """
	----------------------------------------------------------------------------------------
	Transposes the tablesample.

 	Returns
 	-------
 	tablesample
 		transposed tablesample
		"""
        index = [column for column in self.values]
        first_item = list(self.values.keys())[0]
        columns = [[] for i in range(len(self.values[first_item]))]
        for column in self.values:
            for idx, item in enumerate(self.values[column]):
                try:
                    columns[idx] += [item]
                except:
                    pass
        columns = [index] + columns
        values = {}
        for item in columns:
            values[item[0]] = item[1 : len(item)]
        return tablesample(values, self.dtype, self.count, self.offset, self.percent)

    # ---#
    def to_list(self):
        """
    ----------------------------------------------------------------------------------------
    Converts the tablesample to a list.

    Returns
    -------
    list
        Python list.
        """
        result = []
        all_cols = [elem for elem in self.values]
        if all_cols == []:
            return []
        for i in range(len(self.values[all_cols[0]])):
            result_tmp = []
            for elem in self.values:
                if elem != "index":
                    result_tmp += [self.values[elem][i]]
            result += [result_tmp]
        return result

    # ---#
    def to_numpy(self):
        """
    ----------------------------------------------------------------------------------------
    Converts the tablesample to a numpy array.

    Returns
    -------
    numpy.array
        Numpy Array.
        """
        import numpy as np

        return np.array(self.to_list())

    # ---#
    def to_pandas(self):
        """
	----------------------------------------------------------------------------------------
	Converts the tablesample to a pandas DataFrame.

 	Returns
 	-------
 	pandas.DataFrame
 		pandas DataFrame of the tablesample.

	See Also
	--------
	tablesample.to_sql : Generates the SQL query associated to the tablesample.
	tablesample.to_vdf : Converts the tablesample to vDataFrame.
		"""
        if "index" in self.values:
            df = pd.DataFrame(data=self.values, index=self.values["index"])
            return df.drop(columns=["index"])
        else:
            return pd.DataFrame(data=self.values)

    # ---#
    def to_sql(self):
        """
    ----------------------------------------------------------------------------------------
    Generates the SQL query associated to the tablesample.

    Returns
    -------
    str
        SQL query associated to the tablesample.

    See Also
    --------
    tablesample.to_pandas : Converts the tablesample to a pandas DataFrame.
    tablesample.to_sql    : Generates the SQL query associated to the tablesample.
        """

        def get_correct_format_and_cast(val):
            if isinstance(val, str):
                val = "'" + val.replace("'", "''") + "'"
            elif val == None:
                val = "NULL"
            elif isinstance(val, bytes):
                val = str(val)[2:-1]
                val = f"'{val}'::binary({len(val)})"
            elif isinstance(val, datetime.datetime):
                val = f"'{val}'::datetime"
            elif isinstance(val, datetime.date):
                val = f"'{val}'::date"
            elif isinstance(val, datetime.timedelta):
                val = f"'{val}'::interval"
            elif isinstance(val, datetime.time):
                val = f"'{val}'::time"
            elif isinstance(val, datetime.timezone):
                val = f"'{val}'::timestamptz"
            elif isinstance(val, (np.ndarray, list)):
                vertica_version(condition=[10, 0, 0])
                val = f"""
                ARRAY[
                    {", ".join([str(get_correct_format_and_cast(k)) for k in val])}
                     ]"""
            elif isinstance(val, dict):
                vertica_version(condition=[11, 0, 0])
                all_elems = [
                    f"{get_correct_format_and_cast(val[k])} AS {k}" for k in val
                ]
                val = ", ".join(all_elems)
                val = f"ROW({val})"
            try:
                if math.isnan(val):
                    val = "NULL"
            except:
                pass
            return val

        sql = []
        n = len(self.values[list(self.values.keys())[0]])
        for i in range(n):
            row = []
            for column in self.values:
                val = get_correct_format_and_cast(self.values[column][i])
                column_str = '"' + column.replace('"', "") + '"'
                row += [f"{val} AS {column_str}"]
            sql += [f"(SELECT {', '.join(row)})"]
        sql = " UNION ALL ".join(sql)
        return sql

    # ---#
    def to_vdf(self):
        """
	----------------------------------------------------------------------------------------
	Converts the tablesample to a vDataFrame.

 	Returns
 	-------
 	vDataFrame
 		vDataFrame of the tablesample.

	See Also
	--------
	tablesample.to_pandas : Converts the tablesample to a pandas DataFrame.
	tablesample.to_sql    : Generates the SQL query associated to the tablesample.
		"""
        return vDataFrameSQL(f"({self.to_sql()}) sql_relation")


# ---#
@check_dtypes
def to_tablesample(
    query: Union[str, str_sql],
    title: str = "",
    max_columns: int = -1,
    sql_push_ext: bool = False,
    symbol: str = "$",
):
    """
	----------------------------------------------------------------------------------------
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
    if verticapy.OPTIONS["sql_on"]:
        print_query(query, title)
    start_time = time.time()
    cursor = executeSQL(
        query, print_time_sql=False, sql_push_ext=sql_push_ext, symbol=symbol
    )
    description, dtype = cursor.description, {}
    for elem in description:
        dtype[elem[0]] = get_final_vertica_type(
            type_name=elem.type_name,
            display_size=elem[2],
            precision=elem[4],
            scale=elem[5],
        )
    elapsed_time = time.time() - start_time
    if verticapy.OPTIONS["time_on"]:
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


# ---#
@check_dtypes
def vDataFrameSQL(
    relation: str,
    name: str = "VDF",
    schema: str = "public",
    history: list = [],
    saving: list = [],
    vdf=None,
):
    """
----------------------------------------------------------------------------------------
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
    from verticapy import vDataFrame

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
        from verticapy.vcolumn import vColumn

        column_name = '"' + column.replace('"', "_") + '"'
        category = get_category_from_vertica_type(ctype)
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
# ---#
@check_dtypes
def vertica_version(condition: list = []):
    """
----------------------------------------------------------------------------------------
Returns the Vertica Version.

Parameters
----------
condition: list, optional
    List of the minimal version information. If the current version is not
    greater or equal to this one, it will raise an error.

Returns
-------
list
    List containing the version information.
    [MAJOR, MINOR, PATCH, POST]
    """
    # -#
    if condition:
        condition = condition + [0 for elem in range(4 - len(condition))]
    if not (verticapy.OPTIONS["vertica_version"]):
        current_version = executeSQL(
            "SELECT /*+LABEL('utilities.version')*/ version();",
            title="Getting the version.",
            method="fetchfirstelem",
        ).split("Vertica Analytic Database v")[1]
        current_version = current_version.split(".")
        result = []
        try:
            result += [int(current_version[0])]
            result += [int(current_version[1])]
            result += [int(current_version[2].split("-")[0])]
            result += [int(current_version[2].split("-")[1])]
        except:
            pass
        verticapy.OPTIONS["vertica_version"] = result
    else:
        result = verticapy.OPTIONS["vertica_version"]
    if condition:
        if condition[0] < result[0]:
            test = True
        elif condition[0] == result[0]:
            if condition[1] < result[1]:
                test = True
            elif condition[1] == result[1]:
                if condition[2] <= result[2]:
                    test = True
                else:
                    test = False
            else:
                test = False
        else:
            test = False
        if not (test):
            v0, v1, v2 = result[0], result[1], str(result[2]).split("-")[0]
            v = ".".join([str(c) for c in condition[:3]])
            raise VersionError(
                (
                    "This Function is not available for Vertica version "
                    f"{v0}.{v1}.{v2}.\nPlease upgrade your Vertica "
                    f"version to at least {v} to get this functionality."
                )
            )
    return result
