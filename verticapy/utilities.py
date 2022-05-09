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
import os, math, shutil, re, time, decimal, warnings, datetime
from typing import Union

# VerticaPy Modules
import vertica_python
import verticapy
from verticapy.toolbox import *
from verticapy.errors import *

# Other Modules
try:
    from IPython.core.display import display
except:
    pass

#
# ---#
def create_schema(
    schema: str, raise_error: bool = False,
):
    """
---------------------------------------------------------------------------
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
---------------------------------------------------------------------------
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
    check_types(
        [
            ("table_name", table_name, [str]),
            ("schema", schema, [str]),
            ("dtype", dtype, [dict]),
            ("genSQL", genSQL, [bool]),
            ("temporary_table", temporary_table, [bool]),
            ("temporary_local_table", temporary_local_table, [bool]),
            ("raise_error", raise_error, [bool]),
        ]
    )
    if schema.lower() == "v_temp_schema":
        schema = ""
        temporary_local_table = True
    input_relation = (
        quote_ident(schema) + "." + quote_ident(table_name)
        if schema
        else quote_ident(table_name)
    )
    temp = "TEMPORARY " if temporary_table else ""
    if not (schema):
        temp = "LOCAL TEMPORARY " if temporary_local_table else ""
    query = "CREATE {}TABLE {}({}){};".format(
        temp,
        input_relation,
        ", ".join(
            ["{} {}".format(quote_ident(column), dtype[column]) for column in dtype]
        ),
        " ON COMMIT PRESERVE ROWS" if temp else "",
    )
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
---------------------------------------------------------------------------
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
def drop(name: str = "", method: str = "auto", raise_error: bool = False, **kwds):
    """
---------------------------------------------------------------------------
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
    if "relation_type" in kwds and method == "auto":
        method = kwds["relation_type"]
    if isinstance(method, str):
        method = method.lower()
    check_types(
        [
            ("name", name, [str]),
            (
                "method",
                method,
                ["table", "view", "model", "geo", "text", "auto", "schema"],
            ),
            ("raise_error", raise_error, [bool]),
        ]
    )
    schema, relation = schema_relation(name)
    schema, relation = schema[1:-1], relation[1:-1]
    if not (name):
        method = "temp"
    if method == "auto":
        fail, end_conditions = False, False
        query = (
            f"SELECT * FROM columns WHERE table_schema = '{schema}'"
            f" AND table_name = '{relation}'"
        )
        result = executeSQL(query, print_time_sql=False, method="fetchrow")
        if not (result):
            query = (
                f"SELECT * FROM view_columns WHERE table_schema = '{schema}'"
                f" AND table_name = '{relation}'"
            )
            result = executeSQL(query, print_time_sql=False, method="fetchrow")
        elif not (end_conditions):
            method = "table"
            end_conditions = True
        if not (result):
            try:
                query = (
                    "SELECT model_type FROM verticapy.models WHERE "
                    "LOWER(model_name) = '{0}'"
                ).format(quote_ident(name).lower())
                result = executeSQL(query, print_time_sql=False, method="fetchrow")
            except:
                result = []
        elif not (end_conditions):
            method = "view"
            end_conditions = True
        if not (result):
            query = f"SELECT * FROM models WHERE schema_name = '{schema}' AND model_name = '{relation}'"
            result = executeSQL(query, print_time_sql=False, method="fetchrow")
        elif not (end_conditions):
            method = "model"
            end_conditions = True
        if not (result):
            query = (
                "SELECT * FROM (SELECT STV_Describe_Index () OVER ()) x  WHERE name IN "
                f"('{schema}.{relation}', '{relation}', '\"{schema}\".\"{relation}\"', "
                f"'\"{relation}\"', '{schema}.\"{relation}\"', '\"{schema}\".{relation}')"
            )
            result = executeSQL(query, print_time_sql=False, method="fetchrow")
        elif not (end_conditions):
            method = "model"
            end_conditions = True
        if not (result):
            try:
                query = f'SELECT * FROM "{schema}"."{relation}" LIMIT 0;'
                executeSQL(query, print_time_sql=False)
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
            query = "SELECT model_type FROM verticapy.models WHERE LOWER(model_name) = '{}'".format(
                quote_ident(name).lower()
            )
            result = executeSQL(query, print_time_sql=False, method="fetchfirstelem")
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
                query = (
                    "SELECT value FROM verticapy.attr WHERE LOWER(model_name) = '{0}' "
                    "AND attr_name = 'countvectorizer_table'"
                ).format(quote_ident(name).lower())
                res = executeSQL(query, print_time_sql=False, method="fetchrow")
                if res and res[0]:
                    drop(res[0], method="table")
            elif model_type == "KernelDensity":
                drop(name.replace('"', "") + "_KernelDensity_Map", method="table")
                drop(
                    "{}_KernelDensity_Tree".format(name.replace('"', "")),
                    method="model",
                )
            elif model_type == "AutoDataPrep":
                drop(name, method="table")
            if is_in_verticapy_schema:
                sql = "DELETE FROM verticapy.models WHERE LOWER(model_name) = '{}';".format(
                    quote_ident(name).lower()
                )
                executeSQL(sql, title="Deleting vModel.")
                executeSQL("COMMIT;", title="Commit.")
                sql = "DELETE FROM verticapy.attr WHERE LOWER(model_name) = '{}';".format(
                    quote_ident(name).lower()
                )
                executeSQL(sql, title="Deleting vModel attributes.")
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
        sql = """SELECT 
                    table_schema, table_name 
                 FROM columns 
                 WHERE LOWER(table_name) LIKE '%_verticapy_tmp_%' 
                 GROUP BY 1, 2;"""
        all_tables = result = executeSQL(sql, print_time_sql=False, method="fetchall")
        for elem in all_tables:
            table = '"{}"."{}"'.format(
                elem[0].replace('"', '""'), elem[1].replace('"', '""')
            )
            drop(table, method="table")
        sql = """SELECT 
                    table_schema, table_name 
                 FROM view_columns 
                 WHERE LOWER(table_name) LIKE '%_verticapy_tmp_%' 
                 GROUP BY 1, 2;"""
        all_views = executeSQL(sql, print_time_sql=False, method="fetchall")
        for elem in all_views:
            view = '"{}"."{}"'.format(
                elem[0].replace('"', '""'), elem[1].replace('"', '""')
            )
            drop(view, method="view")
        result = True
    else:
        result = True
    return result


# ---#
def readSQL(query: str, time_on: bool = False, limit: int = 100):
    """
	---------------------------------------------------------------------------
	Returns the result of a SQL query as a tablesample object.

	Parameters
	----------
	query: str, optional
		SQL Query.
	time_on: bool, optional
		If set to True, displays the query elapsed time.
	limit: int, optional
		Number maximum of elements to display.

 	Returns
 	-------
 	tablesample
 		Result of the query.
	"""
    check_types(
        [
            ("query", query, [str]),
            ("time_on", time_on, [bool]),
            ("limit", limit, [int, float]),
        ]
    )
    while len(query) > 0 and query[-1] in (";", " "):
        query = query[:-1]
    count = executeSQL(
        "SELECT COUNT(*) FROM ({}) VERTICAPY_SUBTABLE".format(query),
        method="fetchfirstelem",
        print_time_sql=False,
    )
    sql_on_init = verticapy.options["sql_on"]
    time_on_init = verticapy.options["time_on"]
    try:
        verticapy.options["time_on"] = time_on
        verticapy.options["sql_on"] = False
        try:
            result = to_tablesample("{} LIMIT {}".format(query, limit))
        except:
            result = to_tablesample(query)
    except:
        verticapy.options["time_on"] = time_on_init
        verticapy.options["sql_on"] = sql_on_init
        raise
    verticapy.options["time_on"] = time_on_init
    verticapy.options["sql_on"] = sql_on_init
    result.count = count
    if verticapy.options["percent_bar"]:
        vdf = vDataFrameSQL("({}) VERTICAPY_SUBTABLE".format(query))
        percent = vdf.agg(["percent"]).transpose().values
        for column in result.values:
            result.dtype[column] = vdf[column].ctype()
            result.percent[column] = percent[vdf.format_colnames(column)][0]
    return result


# ---#
def get_data_types(expr: str, column_name: str = ""):
    """
---------------------------------------------------------------------------
Returns customized relation columns and the respective data types.
This process creates a temporary table.

Parameters
----------
expr: str
	An expression in pure SQL.
column_name: str, optional
	If not empty, it will return only the data type of the input column if it
	is in the relation.

Returns
-------
list of tuples
	The list of the different columns and their respective type.
	"""
    from verticapy.connect import current_cursor

    if isinstance(current_cursor(), vertica_python.vertica.cursor.Cursor):
        try:
            if column_name:
                executeSQL(expr, print_time_sql=False)
                description = current_cursor().description[0]
                return type_code_to_dtype(
                    type_code=description[1],
                    display_size=description[2],
                    precision=description[4],
                    scale=description[5],
                )
            else:
                executeSQL(expr, print_time_sql=False)
                description, ctype = current_cursor().description, []
                for elem in description:
                    ctype += [
                        [
                            elem[0],
                            type_code_to_dtype(
                                type_code=elem[1],
                                display_size=elem[2],
                                precision=elem[4],
                                scale=elem[5],
                            ),
                        ]
                    ]
                return ctype
        except:
            pass
    tmp_name, schema = gen_tmp_name(name="table"), "v_temp_schema"
    drop("{}.{}".format(schema, tmp_name), method="table")
    try:
        if schema == "v_temp_schema":
            executeSQL(
                "CREATE LOCAL TEMPORARY TABLE {} ON COMMIT PRESERVE ROWS AS {}".format(
                    tmp_name, expr
                ),
                print_time_sql=False,
            )
        else:
            executeSQL(
                "CREATE TEMPORARY TABLE {}.{} ON COMMIT PRESERVE ROWS AS {}".format(
                    schema, tmp_name, expr
                ),
                print_time_sql=False,
            )
    except:
        drop("{}.{}".format(schema, tmp_name), method="table")
        raise
    query = (
        "SELECT column_name, data_type FROM columns WHERE {0}table_name = '{1}'"
        " AND table_schema = '{2}' ORDER BY ordinal_position"
    ).format(
        f"column_name = '{column_name}' AND " if (column_name) else "",
        tmp_name,
        schema,
    )
    cursor = executeSQL(query, title="Getting the data types.")
    if column_name:
        ctype = cursor.fetchone()[1]
    else:
        ctype = cursor.fetchall()
    drop("{}.{}".format(schema, tmp_name), method="table")
    return ctype


# ---#
def insert_into(
    table_name: str,
    data: list,
    schema: str = "",
    column_names: list = [],
    copy: bool = True,
    genSQL: bool = False,
):
    """
---------------------------------------------------------------------------
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
    check_types(
        [
            ("table_name", table_name, [str]),
            ("column_names", column_names, [list]),
            ("data", data, [list]),
            ("schema", schema, [str]),
            ("copy", copy, [bool]),
            ("genSQL", genSQL, [bool]),
        ]
    )
    if not (schema):
        schema = verticapy.options["temp_schema"]
    input_relation = "{}.{}".format(quote_ident(schema), quote_ident(table_name))
    if not (column_names):
        query = f"""SELECT 
                        column_name
                    FROM columns 
                    WHERE table_name = '{table_name}' 
                        AND table_schema = '{schema}' 
                    ORDER BY ordinal_position"""
        result = executeSQL(
            query,
            title=f"Getting the table {input_relation} column names.",
            method="fetchall",
        )
        column_names = [elem[0] for elem in result]
        assert column_names, MissingRelation(
            f"The table {input_relation} does not exist."
        )
    cols = [quote_ident(col) for col in column_names]
    if copy and not (genSQL):
        sql = "INSERT INTO {} ({}) VALUES ({})".format(
            input_relation,
            ", ".join(cols),
            ", ".join(["%s" for i in range(len(cols))]),
        )
        executeSQL(
            sql,
            title=(
                f"Insert new lines in the {table_name} table. The batch insert is "
                "converted into a COPY statement by using prepared statements."
            ),
            data=list(map(tuple, data)),
        )
        executeSQL("COMMIT;", title="Commit.")
        return len(data)
    else:
        if genSQL:
            sql = []
        i, n, total_rows = 0, len(data), 0
        header = "INSERT INTO {} ({}) VALUES ".format(input_relation, ", ".join(cols))
        for i in range(n):
            sql_tmp = "("
            for elem in data[i]:
                if isinstance(elem, str):
                    sql_tmp += "'{}'".format(elem.replace("'", "''"))
                elif elem is None or elem != elem:
                    sql_tmp += "NULL"
                else:
                    sql_tmp += "'{}'".format(elem)
                sql_tmp += ","
            sql_tmp = sql_tmp[:-1] + ");"
            query = header + sql_tmp
            if genSQL:
                sql += [query]
            else:
                try:
                    executeSQL(
                        query,
                        title="Insert a new line in the relation: {}.".format(
                            input_relation
                        ),
                    )
                    executeSQL("COMMIT;", title="Commit.")
                    total_rows += 1
                except Exception as e:
                    warning_message = "Line {} was skipped.\n{}".format(i, e)
                    warnings.warn(warning_message, Warning)
        if genSQL:
            return sql
        else:
            return total_rows


# ---#
def pandas_to_vertica(
    df,
    name: str = "",
    schema: str = "",
    dtype: dict = {},
    parse_nrows: int = 10000,
    temp_path: str = "",
    insert: bool = False,
):
    """
---------------------------------------------------------------------------
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
    check_types(
        [
            ("name", name, [str]),
            ("schema", schema, [str]),
            ("parse_nrows", parse_nrows, [int]),
            ("dtype", dtype, [dict]),
            ("temp_path", temp_path, [str]),
            ("insert", insert, [bool]),
        ]
    )
    if not (schema):
        schema = verticapy.options["temp_schema"]
    assert name or not (insert), ParameterError(
        "Parameter 'name' can not be empty when parameter 'insert' is set to True."
    )
    if not (name):
        tmp_name = gen_tmp_name(name="df")[1:-1]
    else:
        tmp_name = ""
    path = "{0}{1}{2}.csv".format(
        temp_path, "/" if (len(temp_path) > 1 and temp_path[-1] != "/") else "", name
    )
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
        import csv

        tmp_df.to_csv(
            path, index=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar="\027"
        )
        if insert:
            input_relation = "{}.{}".format(quote_ident(schema), quote_ident(name))
            query = """COPY {0}({1}) 
                       FROM LOCAL '{2}' 
                       DELIMITER ',' 
                       NULL ''
                       ENCLOSED BY '\"' 
                       ESCAPE AS '\\' 
                       SKIP 1;""".format(
                input_relation,
                ", ".join(
                    ['"' + col.replace('"', '""') + '"' for col in tmp_df.columns]
                ),
                path,
            )
            executeSQL(query, title="Inserting the pandas.DataFrame.")
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
        os.remove(path)
    except:
        os.remove(path)
        if clear:
            del tmp_df
        raise
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
    ingest_local: bool = True,
):
    """
---------------------------------------------------------------------------
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
        """CREATE FLEX LOCAL TEMP TABLE {0}(x int) 
           ON COMMIT PRESERVE ROWS;""".format(
            flex_name
        ),
        title="Creating flex table to identify the data types.",
    )
    header_names = (
        ""
        if not (header_names)
        else "header_names = '{0}',".format(sep.join(header_names))
    )
    executeSQL(
        """COPY {0} 
           FROM{1} '{2}' 
           PARSER FCSVPARSER(
                type = 'traditional', 
                delimiter = '{3}', 
                header = {4}, {5} 
                enclosed_by = '{6}', 
                escape = '{7}') 
           NULL '{8}';""".format(
            flex_name,
            " LOCAL" if ingest_local else "",
            path,
            sep,
            header,
            header_names,
            quotechar,
            escape,
            na_rep,
        ),
        title="Parsing the data.",
    )
    executeSQL(
        f"SELECT compute_flextable_keys('{flex_name}');",
        title="Guessing flex tables keys.",
    )
    result = executeSQL(
        f"SELECT key_name, data_type_guess FROM {flex_name}_keys",
        title="Guessing the data types.",
        method="fetchall",
    )
    dtype = {}
    for column_dtype in result:
        try:
            query = """SELECT 
                        (CASE 
                            WHEN "{0}"=\'{1}\' THEN NULL 
                            ELSE "{0}" 
                         END)::{2} AS "{0}" 
                       FROM {3} 
                       WHERE "{0}" IS NOT NULL 
                       LIMIT 1000""".format(
                column_dtype[0], na_rep, column_dtype[1], flex_name,
            )
            executeSQL(query, print_time_sql=False)
            dtype[column_dtype[0]] = column_dtype[1]
        except:
            dtype[column_dtype[0]] = "Varchar(100)"
    drop(flex_name, method="table")
    return dtype


# ---#
def help_start():
    """
---------------------------------------------------------------------------
VERTICAPY Interactive Help (FAQ).
    """
    try:
        from IPython.core.display import HTML, display, Markdown
    except:
        pass
    path = os.path.dirname(verticapy.__file__)
    img1 = verticapy.gen_verticapy_logo_html(size="10%")
    img2 = verticapy.gen_verticapy_logo_str()
    message = img1 if (isnotebook()) else img2
    message += (
        "\n\n&#128226; Welcome to the <b>VerticaPy</b> help module."
        "\n\nFrom here, you can learn how to connect to Vertica, "
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

vHelp = help_start
# ---#
def pjson(path: str, ingest_local: bool = True):
    """
---------------------------------------------------------------------------
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
        "CREATE FLEX LOCAL TEMP TABLE {}(x int) ON COMMIT PRESERVE ROWS;".format(
            flex_name
        ),
        title="Creating a flex table.",
    )
    executeSQL(
        "COPY {} FROM{} '{}' PARSER FJSONPARSER();".format(
            flex_name, " LOCAL" if ingest_local else "", path.replace("'", "''")
        ),
        title="Ingesting the data.",
    )
    executeSQL(
        "SELECT compute_flextable_keys('{}');".format(flex_name),
        title="Computing flex table keys.",
    )
    result = executeSQL(
        "SELECT key_name, data_type_guess FROM {}_keys".format(flex_name),
        title="Guessing data types.",
        method="fetchall",
    )
    dtype = {}
    for column_dtype in result:
        dtype[column_dtype[0]] = column_dtype[1]
    drop(name=flex_name, method="table")
    return dtype


# ---#
def read_csv(
    path: str,
    schema: str = "",
    table_name: str = "",
    sep: str = ",",
    header: bool = True,
    header_names: list = [],
    dtype: dict = {},
    na_rep: str = "",
    quotechar: str = '"',
    escape: str = "\027",
    genSQL: bool = False,
    parse_nrows: int = -1,
    insert: bool = False,
    temporary_table: bool = False,
    temporary_local_table: bool = True,
    gen_tmp_table_name: bool = True,
    ingest_local: bool = True,
):
    """
---------------------------------------------------------------------------
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
genSQL: bool, optional
	If set to True, the SQL code for creating the final table will be 
	generated but not executed. It is a good way to change the final
	relation types or to customize the data ingestion.
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

Returns
-------
vDataFrame
	The vDataFrame of the relation.

See Also
--------
read_json : Ingests a JSON file into the Vertica database.
	"""
    check_types(
        [
            ("path", path, [str]),
            ("schema", schema, [str]),
            ("table_name", table_name, [str]),
            ("sep", sep, [str]),
            ("header", header, [bool]),
            ("header_names", header_names, [list]),
            ("na_rep", na_rep, [str]),
            ("dtype", dtype, [dict]),
            ("quotechar", quotechar, [str]),
            ("escape", escape, [str]),
            ("genSQL", genSQL, [bool]),
            ("parse_nrows", parse_nrows, [int, float]),
            ("insert", insert, [bool]),
            ("temporary_table", temporary_table, [bool]),
            ("temporary_local_table", temporary_local_table, [bool]),
            ("gen_tmp_table_name", gen_tmp_table_name, [bool]),
            ("ingest_local", ingest_local, [bool]),
        ]
    )
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
    if gen_tmp_table_name and temporary_local_table and not (table_name):
        table_name = gen_tmp_name(name=path.split("/")[-1].split(".csv")[0])
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
    file = path.split("/")[-1]
    file_extension = file[-3 : len(file)]
    if file_extension != "csv":
        raise ExtensionError("The file extension is incorrect !")
    if not (table_name):
        table_name = path.split("/")[-1].split(".csv")[0]
    if table_name == "*":
        assert dtype, ParameterError(
            "Parameter 'dtype' must include the types of all columns in "
            "the table when ingesting multiple files."
        )
        table_name = path.split("/")[-2]
    query = """SELECT 
                    column_name 
               FROM columns 
               WHERE table_name = '{0}' 
                 AND table_schema = '{1}' 
               ORDER BY ordinal_position""".format(
        table_name.replace("'", "''"), schema.replace("'", "''")
    )
    result = executeSQL(
        query, title="Looking if the relation exists.", method="fetchall"
    )
    if (result != []) and not (insert) and not (genSQL):
        raise NameError(
            'The table "{}"."{}" already exists !'.format(schema, table_name)
        )
    elif (result == []) and (insert):
        raise MissingRelation(
            'The table "{}"."{}" doesn\'t exist !'.format(schema, table_name)
        )
    else:
        if not (temporary_local_table):
            input_relation = "{}.{}".format(
                quote_ident(schema), quote_ident(table_name)
            )
        else:
            input_relation = "v_temp_schema.{}".format(quote_ident(table_name))
        f = open(path, "r")
        file_header = f.readline().replace("\n", "").replace('"', "").split(sep)
        f.close()
        if not (header_names) and not (dtype):
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
        if (header_names == []) and (header):
            if not (dtype):
                header_names = file_header
            else:
                header_names = [elem for elem in dtype]
            for idx in range(len(header_names)):
                h = header_names[idx]
                n = len(h)
                while n > 0 and h[0] == " ":
                    h = h[1:]
                    n -= 1
                while n > 0 and h[-1] == " ":
                    h = h[:-1]
                    n -= 1
                header_names[idx] = h
        elif len(file_header) > len(header_names):
            header_names += [
                "ucol{}".format(i + len(header_names))
                for i in range(len(file_header) - len(header_names))
            ]
        if (parse_nrows > 0) and not (insert):
            f = open(path, "r")
            f2 = open(path[0:-4] + "verticapy_copy.csv", "w")
            for i in range(parse_nrows + int(header)):
                line = f.readline()
                f2.write(line)
            f.close()
            f2.close()
            path_test = path[0:-4] + "verticapy_copy.csv"
        else:
            path_test = path
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
                os.remove(path[0:-4] + "verticapy_copy.csv")
            dtype_sorted = {}
            for elem in header_names:
                dtype_sorted[elem] = dtype[elem]
            query1 = create_table(
                table_name,
                dtype_sorted,
                schema,
                temporary_table,
                temporary_local_table,
                genSQL=True,
            )
        skip = " SKIP 1" if (header) else ""
        query2 = """COPY {0}({1}) 
                    FROM {2} 
                    DELIMITER '{3}' 
                    NULL '{4}' 
                    ENCLOSED BY '{5}' 
                    ESCAPE AS '{6}'{7};""".format(
            input_relation,
            ", ".join(['"' + column + '"' for column in header_names]),
            "{}",
            sep,
            na_rep,
            quotechar,
            escape,
            skip,
        )
        if genSQL:
            return [clean_query(query1), clean_query(query2)]
        else:
            if query1:
                executeSQL(query1, "Creating the table.")
            executeSQL(
                query2.format("{}'{}'".format("LOCAL " if ingest_local else "", path)),
                "Ingesting the data.",
            )
            if (
                query1
                and not (temporary_local_table)
                and verticapy.options["print_info"]
            ):
                print(
                    "The table {} has been successfully created.".format(input_relation)
                )
            from verticapy import vDataFrame

            return vDataFrame(table_name, schema=schema)


# ---#
def read_json(
    path: str,
    schema: str = "",
    table_name: str = "",
    usecols: list = [],
    new_name: dict = {},
    insert: bool = False,
    temporary_table: bool = False,
    temporary_local_table: bool = True,
    gen_tmp_table_name: bool = True,
    ingest_local: bool = True,
):
    """
---------------------------------------------------------------------------
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
	List of the JSON parameters to ingest. The other ones will be ignored. If
	empty all the JSON parameters will be ingested.
new_name: dict, optional
	Dictionary of the new columns name. If the JSON file is nested, it is advised
	to change the final names as special characters will be included.
	For example, {"param": {"age": 3, "name": Badr}, "date": 1993-03-11} will 
	create 3 columns: "param.age", "param.name" and "date". You can rename these 
	columns using the 'new_name' parameter with the following dictionary:
	{"param.age": "age", "param.name": "name"}
insert: bool, optional
	If set to True, the data will be ingested to the input relation. The JSON
	parameters must be the same than the input relation otherwise they will
	not be ingested.
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

Returns
-------
vDataFrame
	The vDataFrame of the relation.

See Also
--------
read_csv : Ingests a CSV file into the Vertica database.
	"""
    check_types(
        [
            ("schema", schema, [str]),
            ("table_name", table_name, [str]),
            ("usecols", usecols, [list]),
            ("new_name", new_name, [dict]),
            ("insert", insert, [bool]),
            ("temporary_table", temporary_table, [bool]),
            ("temporary_local_table", temporary_local_table, [bool]),
            ("gen_tmp_table_name", gen_tmp_table_name, [bool]),
            ("ingest_local", ingest_local, [bool]),
        ]
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
    file = path.split("/")[-1]
    file_extension = file[-4 : len(file)]
    if file_extension != "json":
        raise ExtensionError("The file extension is incorrect !")
    if gen_tmp_table_name and temporary_local_table and not (table_name):
        table_name = gen_tmp_name(name=path.split("/")[-1].split(".json")[0])
    if not (table_name):
        table_name = path.split("/")[-1].split(".json")[0]
    query = (
        "SELECT column_name, data_type FROM columns WHERE table_name = '{0}' "
        "AND table_schema = '{1}' ORDER BY ordinal_position"
    ).format(table_name.replace("'", "''"), schema.replace("'", "''"))
    column_name = executeSQL(
        query, title="Looking if the relation exists.", method="fetchall"
    )
    if (column_name != []) and not (insert):
        raise NameError(
            'The table "{}"."{}" already exists !'.format(schema, table_name)
        )
    elif (column_name == []) and (insert):
        raise MissingRelation(
            'The table "{}"."{}" doesn\'t exist !'.format(schema, table_name)
        )
    else:
        if not (temporary_local_table):
            input_relation = '"{}"."{}"'.format(schema, table_name)
        else:
            input_relation = '"{}"'.format(table_name)
        flex_name = gen_tmp_name(name="flex")[1:-1]
        executeSQL(
            "CREATE FLEX LOCAL TEMP TABLE {0}(x int) ON COMMIT PRESERVE ROWS;".format(
                flex_name
            ),
            title="Creating flex table.",
        )
        executeSQL(
            "COPY {} FROM{} '{}' PARSER FJSONPARSER();".format(
                flex_name, " LOCAL" if ingest_local else "", path.replace("'", "''")
            ),
            title="Ingesting the data in the flex table.",
        )
        executeSQL(
            "SELECT compute_flextable_keys('{}');".format(flex_name),
            title="Computing flex table keys.",
        )
        result = executeSQL(
            "SELECT key_name, data_type_guess FROM {}_keys".format(flex_name),
            title="Guessing data types.",
            method="fetchall",
        )
        dtype = {}
        for column_dtype in result:
            try:
                executeSQL(
                    'SELECT "{}"::{} FROM {} LIMIT 1000'.format(
                        column_dtype[0], column_dtype[1], flex_name
                    ),
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
                cols[i] = (
                    '"{}"::{} AS "{}"'.format(
                        column.replace('"', ""), dtype[column], new_name[column]
                    )
                    if (column in new_name)
                    else '"{}"::{}'.format(column.replace('"', ""), dtype[column])
                )
            temp = "TEMPORARY " if temporary_table else ""
            temp = "LOCAL TEMPORARY " if temporary_local_table else ""
            executeSQL(
                "CREATE {}TABLE {}{} AS SELECT {} FROM {}".format(
                    temp,
                    input_relation,
                    " ON COMMIT PRESERVE ROWS" if temp else "",
                    ", ".join(cols),
                    flex_name,
                ),
                title="Creating table.",
            )
            if not (temporary_local_table) and verticapy.options["print_info"]:
                print(
                    "The table {} has been successfully created.".format(input_relation)
                )
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
                final_transformation += (
                    ['NULL AS "{}"'.format(column)]
                    if (final_cols[column] == None)
                    else [
                        '"{}"::{} AS "{}"'.format(
                            final_cols[column], column_name_dtype[column], column
                        )
                    ]
                )
            executeSQL(
                "INSERT INTO {} SELECT {} FROM {}".format(
                    input_relation, ", ".join(final_transformation), flex_name
                ),
                title="Inserting data into table.",
            )
        drop(name=flex_name, method="table")
        from verticapy import vDataFrame

        return vDataFrame(table_name, schema=schema)


# ---#
def read_shp(
    path: str, schema: str = "public", table_name: str = "",
):
    """
---------------------------------------------------------------------------
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
    check_types(
        [
            ("path", path, [str]),
            ("schema", schema, [str]),
            ("table_name", table_name, [str]),
        ]
    )
    file = path.split("/")[-1]
    file_extension = file[-3 : len(file)]
    if file_extension != "shp":
        raise ExtensionError("The file extension is incorrect !")
    query = (
        f"SELECT STV_ShpCreateTable(USING PARAMETERS file='{path}')"
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
def set_option(option: str, value: Union[bool, int, str] = None):
    """
    ---------------------------------------------------------------------------
    Sets VerticaPy options.

    Parameters
    ----------
    option: str
        Option to use.
        cache          : bool
            If set to True, the vDataFrame will save in memory the computed
            aggregations.
        colors         : list
            List of the colors used to draw the graphics.
        color_style    : str
            Style used to color the graphics, one of the following:
            "rgb", "sunset", "retro", "shimbg", "swamp", "med", "orchid", 
            "magenta", "orange", "vintage", "vivid", "berries", "refreshing", 
            "summer", "tropical", "india", "default".
        max_columns    : int
            Maximum number of columns to display. If the parameter is incorrect, 
            nothing is changed.
        max_rows       : int
            Maximum number of rows to display. If the parameter is incorrect, 
            nothing is changed.
        mode           : str
            How to display VerticaPy outputs.
                full  : VerticaPy regular display mode.
                light : Minimalist display mode.
        overwrite_model: bool
            If set to True and you try to train a model with an existing name. 
            It will be automatically overwritten.
        percent_bar    : bool
            If set to True, it displays the percent of non-missing values.
        print_info     : bool
            If set to True, information will be printed each time the vDataFrame 
            is modified.
        random_state   : int
            Integer used to seed the random number generation in VerticaPy.
        sql_on         : bool
            If set to True, displays all the SQL queries.
        temp_schema    : str
            Specifies the temporary schema that certain methods/functions use to 
            create intermediate objects, if needed. 
        time_on        : bool
            If set to True, displays all the SQL queries elapsed time.
        tqdm           : bool
            If set to True, a loading bar is displayed when using iterative 
            functions.
    value: object, optional
        New value of option.
    """
    if isinstance(option, str):
        option = option.lower()
    check_types(
        [
            (
                "option",
                option,
                [
                    "cache",
                    "colors",
                    "color_style",
                    "max_columns",
                    "max_rows",
                    "mode",
                    "overwrite_model",
                    "percent_bar",
                    "print_info",
                    "random_state",
                    "sql_on",
                    "temp_schema",
                    "time_on",
                    "tqdm",
                ],
            ),
        ]
    )
    if option == "cache":
        check_types([("value", value, [bool])])
        if isinstance(value, bool):
            verticapy.options["cache"] = value
    elif option == "colors":
        check_types([("value", value, [list])])
        if isinstance(value, list):
            verticapy.options["colors"] = [str(elem) for elem in value]
    elif option == "color_style":
        check_types(
            [
                (
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
            ]
        )
        if isinstance(value, str):
            verticapy.options["color_style"] = value
    elif option == "max_columns":
        check_types([("value", value, [int, float])])
        if value > 0:
            verticapy.options["max_columns"] = int(value)
    elif option == "max_rows":
        check_types([("value", value, [int, float])])
        if value >= 0:
            verticapy.options["max_rows"] = int(value)
    elif option == "mode":
        check_types([("value", value, ["light", "full"])])
        if value.lower() in ["light", "full", None]:
            verticapy.options["mode"] = value.lower()
    elif option == "percent_bar":
        check_types([("value", value, [bool])])
        if value in (True, False, None):
            verticapy.options["percent_bar"] = value
    elif option == "print_info":
        check_types([("value", value, [bool])])
        if isinstance(value, bool):
            verticapy.options["print_info"] = value
    elif option == "overwrite_model":
        check_types([("value", value, [bool])])
        if value in (True, False, None):
            verticapy.options["overwrite_model"] = value
    elif option == "random_state":
        check_types([("value", value, [int])])
        if value < 0:
            raise ParameterError("Random State Value must be positive.")
        if isinstance(value, int):
            verticapy.options["random_state"] = int(value)
        elif value == None:
            verticapy.options["random_state"] = None
    elif option == "sql_on":
        check_types([("value", value, [bool])])
        if value in (True, False, None):
            verticapy.options["sql_on"] = value
    elif option == "temp_schema":
        check_types([("value", value, [str])])
        if isinstance(value, str):
            query = """SELECT 
                          schema_name 
                       FROM v_catalog.schemata 
                       WHERE schema_name = '{}' LIMIT 1;""".format(
                value.replace("'", "''")
            )
            res = executeSQL(
                query, title="Checking if the schema exists.", method="fetchrow"
            )
            if res:
                verticapy.options["temp_schema"] = str(value)
            else:
                raise ParameterError(f"The schema '{value}' could not be found.")
    elif option == "time_on":
        check_types([("value", value, [bool])])
        if value in (True, False, None):
            verticapy.options["time_on"] = value
    elif option == "tqdm":
        check_types([("value", value, [bool])])
        if value in (True, False, None):
            verticapy.options["tqdm"] = value
    else:
        raise ParameterError(f"Option '{option}' does not exist.")


# ---#
class tablesample:
    """
---------------------------------------------------------------------------
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
The tablesample attributes are the same than the parameters.
	"""

    #
    # Special Methods
    #
    # ---#
    def __init__(
        self,
        values: dict = {},
        dtype: dict = {},
        count: int = 0,
        offset: int = 0,
        percent: dict = {},
        max_columns: int = -1,
    ):
        check_types(
            [
                ("values", values, [dict]),
                ("dtype", dtype, [dict]),
                ("count", count, [int]),
                ("offset", offset, [int]),
                ("percent", percent, [dict]),
                ("max_columns", max_columns, [int]),
            ]
        )
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
        columns = self.values
        return (elem for elem in columns)

    # ---#
    def __getitem__(self, key):
        all_cols = [elem for elem in self.values]
        for elem in all_cols:
            if quote_ident(str(elem).lower()) == quote_ident(str(key).lower()):
                key = elem
                break
        return self.values[key]

    # ---#
    def _repr_html_(self):
        if len(self.values) == 0:
            return ""
        n = len(self.values)
        dtype = self.dtype
        max_columns = self.max_columns if self.max_columns > 0 else verticapy.options["max_columns"]
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
        formatted_text = print_table(
            data_columns,
            is_finished=(self.count <= len(data_columns[0]) + self.offset),
            offset=self.offset,
            repeat_first_column=("index" in self.values),
            return_html=True,
            dtype=dtype,
            percent=percent,
        )
        start, end = self.offset + 1, len(data_columns[0]) - 1 + self.offset
        formatted_text += '<div style="margin-top:6px; font-size:1.02em">'
        if (self.offset == 0) and (len(data_columns[0]) - 1 == self.count):
            rows = self.count
        else:
            if start > end:
                rows = "0{}".format(
                    " of {}".format(self.count) if (self.count > 0) else ""
                )
            else:
                rows = "{}-{}{}".format(
                    start, end, " of {}".format(self.count) if (self.count > 0) else "",
                )
        if len(self.values) == 1:
            column = list(self.values.keys())[0]
            if self.offset > self.count:
                formatted_text += "<b>Column:</b> {} | <b>Type:</b> {}".format(
                    column, self.dtype[column]
                )
            else:
                formatted_text += "<b>Rows:</b> {} | <b>Column:</b> {} | <b>Type:</b> {}".format(
                    rows, column, self.dtype[column]
                )
        else:
            if self.offset > self.count:
                formatted_text += "<b>Columns:</b> {}".format(n)
            else:
                formatted_text += "<b>Rows:</b> {} | <b>Columns:</b> {}".format(rows, n)
        formatted_text += "</div>"
        return formatted_text

    # ---#
    def __repr__(self):
        if len(self.values) == 0:
            return ""
        n = len(self.values)
        dtype = self.dtype
        max_columns = self.max_columns if self.max_columns > 0 else verticapy.options["max_columns"]
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
                rows = "0{}".format(
                    " of {}".format(self.count) if (self.count > 0) else ""
                )
            else:
                rows = "{}-{}{}".format(
                    start, end, " of {}".format(self.count) if (self.count > 0) else "",
                )
        if len(self.values) == 1:
            column = list(self.values.keys())[0]
            if self.offset > self.count:
                formatted_text += "Column: {} | Type: {}".format(
                    column, self.dtype[column]
                )
            else:
                formatted_text += "Rows: {} | Column: {} | Type: {}".format(
                    rows, column, self.dtype[column]
                )
        else:
            if self.offset > self.count:
                formatted_text += "Columns: {}".format(n)
            else:
                formatted_text += "Rows: {} | Columns: {}".format(rows, n)
        return formatted_text

    #
    # Methods
    #
    # ---#
    def append(self, tbs):
        """
        ---------------------------------------------------------------------------
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
        check_types([("tbs", tbs, [tablesample])])
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
    ---------------------------------------------------------------------------
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
    def merge(self, tbs):
        """
        ---------------------------------------------------------------------------
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
        check_types([("tbs", tbs, [tablesample])])
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
    ---------------------------------------------------------------------------
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
    def sort(self, column: str, desc: bool = False):
        """
        ---------------------------------------------------------------------------
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
        check_types([("column", column, [str]), ("desc", desc, [bool])])
        column = column.replace('"', "").lower()
        columns = [col for col in self.values]
        idx = None
        for i, col in enumerate(columns):
            col_tmp = col.replace('"', "").lower()
            if column == col_tmp:
                idx = i
                column = col
        if idx is None:
            raise MissingColumn(
                "The Column '{}' doesn't exist.".format(column.lower().replace('"', ""))
            )
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
	---------------------------------------------------------------------------
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
    ---------------------------------------------------------------------------
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
    ---------------------------------------------------------------------------
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
	---------------------------------------------------------------------------
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
        import pandas as pd

        if "index" in self.values:
            df = pd.DataFrame(data=self.values, index=self.values["index"])
            return df.drop(columns=["index"])
        else:
            return pd.DataFrame(data=self.values)

    # ---#
    def to_sql(self):
        """
	---------------------------------------------------------------------------
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
        sql = []
        n = len(self.values[list(self.values.keys())[0]])
        for i in range(n):
            row = []
            for column in self.values:
                val = self.values[column][i]
                if isinstance(val, str):
                    val = "'" + val.replace("'", "''") + "'"
                elif val == None:
                    val = "NULL"
                elif isinstance(val, bytes):
                    val = str(val)[2:-1]
                    val = "'{}'::binary({})".format(val, len(val))
                elif isinstance(val, datetime.datetime):
                    val = "'{}'::datetime".format(val)
                elif isinstance(val, datetime.date):
                    val = "'{}'::date".format(val)
                try:
                    if math.isnan(val):
                        val = "NULL"
                except:
                    pass
                row += ["{} AS {}".format(val, '"' + column.replace('"', "") + '"')]
            sql += ["(SELECT {})".format(", ".join(row))]
        sql = " UNION ALL ".join(sql)
        return sql

    # ---#
    def to_vdf(self):
        """
	---------------------------------------------------------------------------
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
        relation = "({}) sql_relation".format(self.to_sql())
        return vDataFrameSQL(relation)


# ---#
def to_tablesample(query: str, title: str = "", max_columns: int = -1,):
    """
	---------------------------------------------------------------------------
	Returns the result of a SQL query as a tablesample object.

	Parameters
	----------
	query: str, optional
		SQL Query.
	title: str, optional
		Query title when the query is displayed.
    max_columns: int, optional
        Maximum number of columns to display.

 	Returns
 	-------
 	tablesample
 		Result of the query.

	See Also
	--------
	tablesample : Object in memory created for rendering purposes.
	"""
    check_types([("query", query, [str]), ("max_columns", max_columns, [int]),])
    if verticapy.options["sql_on"]:
        print_query(query, title)
    start_time = time.time()
    cursor = executeSQL(query, print_time_sql=False)
    description, dtype = cursor.description, {}
    for elem in description:
        dtype[elem[0]] = type_code_to_dtype(
            type_code=elem[1], display_size=elem[2], precision=elem[4], scale=elem[5]
        )
    elapsed_time = time.time() - start_time
    if verticapy.options["time_on"]:
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
    return tablesample(values=values, dtype=dtype, max_columns=max_columns,).decimal_to_float()


# ---#
def vDataFrameSQL(
    relation: str,
    name: str = "VDF",
    schema: str = "public",
    history: list = [],
    saving: list = [],
    vdf=None,
):
    """
---------------------------------------------------------------------------
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

    check_types(
        [
            ("relation", relation, [str]),
            ("name", name, [str]),
            ("schema", schema, [str]),
            ("history", history, [list]),
            ("saving", saving, [list]),
        ]
    )
    if isinstance(vdf, vDataFrame):
        vdf.__init__("", empty=True)
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
            warning_message = (
                'A double quote " was found in the column {0}, its '
                "alias was changed using underscores '_' to {1}"
            ).format(column, column.replace('"', "_"))
            warnings.warn(warning_message, Warning)
        from verticapy.vcolumn import vColumn

        new_vColumn = vColumn(
            '"{}"'.format(column.replace('"', "_")),
            parent=vdf,
            transformations=[
                (
                    '"{}"'.format(column.replace('"', '""')),
                    ctype,
                    get_category_from_vertica_type(ctype),
                )
            ],
        )
        setattr(vdf, '"{}"'.format(column.replace('"', "_")), new_vColumn)
        setattr(vdf, column.replace('"', "_"), new_vColumn)

    return vdf

vdf_from_relation = vDataFrameSQL
# ---#
def version(condition: list = []):
    """
---------------------------------------------------------------------------
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
    check_types([("condition", condition, [list])])
    if condition:
        condition = condition + [0 for elem in range(4 - len(condition))]
    if not (verticapy.options["vertica_version"]):
        version = executeSQL(
            "SELECT version();", title="Getting the version.", method="fetchfirstelem"
        ).split("Vertica Analytic Database v")[1]
        version = version.split(".")
        result = []
        try:
            result += [int(version[0])]
            result += [int(version[1])]
            result += [int(version[2].split("-")[0])]
            result += [int(version[2].split("-")[1])]
        except:
            pass
        verticapy.options["vertica_version"] = result
    else:
        result = verticapy.options["vertica_version"]
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
            raise VersionError(
                (
                    "This Function is not available for Vertica version {0}.\n"
                    "Please upgrade your Vertica version to at least {1} to "
                    "get this functionality."
                ).format(
                    version[0] + "." + version[1] + "." + version[2].split("-")[0],
                    ".".join([str(elem) for elem in condition[:3]]),
                )
            )
    return result
