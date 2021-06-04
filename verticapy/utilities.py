# (c) Copyright [2018-2021] Micro Focus or one of its affiliates.
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
# VerticaPy is a Python library with scikit-like functionality to use to conduct
# data science projects on data stored in Vertica, taking advantage Vertica’s
# speed and built-in analytics and machine learning features. It supports the
# entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize
# data transformation operations, and offers beautiful graphical options.
#
# VerticaPy aims to solve all of these problems. The idea is simple: instead
# of moving data around for processing, VerticaPy brings the logic to the data.
#
#
# Modules
#
# Standard Python Modules
import os, math, shutil, re, time, decimal, warnings

# VerticaPy Modules
import verticapy
import vertica_python
from verticapy.toolbox import *
from verticapy.connect import read_auto_connect, vertica_conn
from verticapy.errors import *

# Other Modules
try:
    from IPython.core.display import display
except:
    pass

#
# ---#
def create_verticapy_schema(cursor=None):
    """
---------------------------------------------------------------------------
Creates a schema named 'verticapy' used to store VerticaPy extended models.

Parameters
----------
cursor: DBcursor, optional
    Vertica database cursor.
    """
    cursor, conn = check_cursor(cursor)[0:2]
    sql = "CREATE SCHEMA verticapy;"
    cursor.execute(sql)
    sql = "CREATE TABLE verticapy.models (model_name VARCHAR(128), category VARCHAR(128), model_type VARCHAR(128), create_time TIMESTAMP, size INT);"
    cursor.execute(sql)
    sql = "CREATE TABLE verticapy.attr (model_name VARCHAR(128), attr_name VARCHAR(128), value VARCHAR(65000));"
    cursor.execute(sql)
    if conn:
        conn.close()


# ---#
def drop(
    name: str = "", cursor=None, raise_error: bool = False, method: str = "auto",
):
    """
---------------------------------------------------------------------------
Drops the input relation. This can be a model, view,table, text index,
schema, or geo index.

Parameters
----------
name: str, optional
    Relation name. If empty, it will drop all VerticaPy temporary elements.
cursor: DBcursor, optional
    Vertica database cursor.
raise_error: bool, optional
    If the geo index couldn't be dropped, raises the entire error instead of
    displaying a warning.
method: str, optional
    Method used to drop.
        auto   : identifies the table/view/index/model to drop. It will never
                 drop an entire schema unless the method is set to 'schema'.
        model  : drops the input model.
        table  : drops the input table.
        view   : drops the input view.        
        geo    : drops the input geo index.
        text   : drops the input text index.
        schema : drops the input schema.

Returns
-------
bool
    True if the relation was dropped, False otherwise.
    """
    if isinstance(method, str):
        method = method.lower()
    check_types(
        [
            ("name", name, [str],),
            ("raise_error", raise_error, [bool],),
            (
                "method",
                method,
                ["table", "view", "model", "geo", "text", "auto", "schema",],
            ),
        ]
    )
    cursor, conn = check_cursor(cursor)[0:2]
    schema, relation = schema_relation(name)
    schema, relation = schema[1:-1], relation[1:-1]
    if not (name):
        method = "temp"
    if method == "auto":
        fail, end_conditions = False, False
        query = f"SELECT * FROM columns WHERE table_schema = '{schema}' AND table_name = '{relation}'"
        cursor.execute(query)
        result = cursor.fetchone()
        if not (result):
            query = f"SELECT * FROM view_columns WHERE table_schema = '{schema}' AND table_name = '{relation}'"
            cursor.execute(query)
            result = cursor.fetchone()
        elif not (end_conditions):
            method = "table"
            end_conditions = True
        if not (result):
            try:
                sql = "SELECT model_type FROM verticapy.models WHERE LOWER(model_name) = '{}'".format(
                    str_column(name).lower()
                )
                cursor.execute(sql)
                result = cursor.fetchone()
            except:
                result = []
        elif not (end_conditions):
            method = "view"
            end_conditions = True
        if not (result):
            query = f"SELECT * FROM models WHERE schema_name = '{schema}' AND model_name = '{relation}'"
            cursor.execute(query)
            result = cursor.fetchone()
        elif not (end_conditions):
            method = "model"
            end_conditions = True
        if not (result):
            query = f"SELECT * FROM (SELECT STV_Describe_Index () OVER ()) x  WHERE name IN ('{schema}.{relation}', '{relation}', '\"{schema}\".\"{relation}\"', '\"{relation}\"', '{schema}.\"{relation}\"', '\"{schema}\".{relation}')"
            cursor.execute(query)
            result = cursor.fetchone()
        elif not (end_conditions):
            method = "model"
            end_conditions = True
        if not (result):
            try:
                query = f'SELECT * FROM "{schema}"."{relation}" LIMIT 1;'
                cursor.execute(query)
                method = "text"
            except:
                fail = True
        elif not (end_conditions):
            method = "geo"
            end_conditions = True
        if fail:
            warning_message = "No relation / index / view / model named '{}' was detected.".format(
                name
            )
            warnings.warn(warning_message, Warning)
            return False
    query = ""
    if method == "model":
        try:
            sql = "SELECT model_type FROM verticapy.models WHERE LOWER(model_name) = '{}'".format(
                str_column(name).lower()
            )
            cursor.execute(sql)
            result = cursor.fetchone()
        except:
            result = []
        if result:
            model_type = result[0]
            if model_type in ("DBSCAN", "LocalOutlierFactor"):
                drop(name, cursor, method="table")
            elif model_type in ("CountVectorizer"):
                drop(name, cursor, method="text")
                sql = "SELECT value FROM verticapy.attr WHERE LOWER(model_name) = '{}' AND attr_name = 'countvectorizer_table'".format(
                    str_column(name).lower()
                )
                cursor.execute(sql)
                drop(cursor.fetchone()[0], cursor, method="table")
            elif model_type in ("KernelDensity",):
                drop(
                    name.replace('"', "") + "_KernelDensity_Map", cursor, method="table"
                )
                drop(
                    "{}_KernelDensity_Tree".format(name.replace('"', "")),
                    cursor,
                    method="model",
                )
            sql = "DELETE FROM verticapy.models WHERE LOWER(model_name) = '{}';".format(
                str_column(name).lower()
            )
            cursor.execute(sql)
            cursor.execute("COMMIT;")
            sql = "DELETE FROM verticapy.attr WHERE LOWER(model_name) = '{}';".format(
                str_column(name).lower()
            )
            cursor.execute(sql)
            cursor.execute("COMMIT;")
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
            cursor.execute(query)
            result = True
        except:
            if raise_error:
                raise
            if method in ("geo", "text"):
                method += " index"
            warning_message = f"The {method} '{name}' doesn't exist or can not be dropped !\nUse parameter: raise_error = True to get more information."
            warnings.warn(warning_message, Warning)
            result = False
    elif method == "temp":
        sql = "SELECT table_schema, table_name FROM columns WHERE LOWER(table_name) LIKE '%verticapy%' GROUP BY 1, 2;"
        cursor.execute(sql)
        all_tables = cursor.fetchall()
        for elem in all_tables:
            table = '"{}"."{}"'.format(
                elem[0].replace('"', '""'), elem[1].replace('"', '""')
            )
            with warnings.catch_warnings(record=True) as w:
                drop(
                    table, cursor, method="table",
                )
        sql = "SELECT table_schema, table_name FROM view_columns WHERE LOWER(table_name) LIKE '%verticapy%' GROUP BY 1, 2;"
        cursor.execute(sql)
        all_views = cursor.fetchall()
        for elem in all_views:
            view = '"{}"."{}"'.format(
                elem[0].replace('"', '""'), elem[1].replace('"', '""')
            )
            with warnings.catch_warnings(record=True) as w:
                drop(
                    view, cursor, method="view",
                )
        result = True
    else:
        result = True
    if conn:
        conn.close()
    return result


# ---#
def readSQL(
    query: str, cursor=None, dsn: str = "", time_on: bool = False, limit: int = 100,
):
    """
	---------------------------------------------------------------------------
	Returns the result of a SQL query as a tablesample object.

	Parameters
	----------
	query: str, optional
		SQL Query. 
	cursor: DBcursor, optional
		Vertica database cursor.
	dsn: str, optional
		Vertica DB DSN.
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
            ("query", query, [str],),
            ("dsn", dsn, [str],),
            ("time_on", time_on, [bool],),
            ("limit", limit, [int, float],),
        ]
    )
    conn = False
    if not (cursor) and not (dsn):
        conn = read_auto_connect()
        cursor = conn.cursor()
    elif not (cursor):
        cursor = vertica_conn(dsn).cursor()
    while len(query) > 0 and query[-1] in (";", " "):
        query = query[:-1]
    cursor.execute("SELECT COUNT(*) FROM ({}) VERTICAPY_SUBTABLE".format(query))
    count = cursor.fetchone()[0]
    query_on_init = verticapy.options["query_on"]
    time_on_init = verticapy.options["time_on"]
    try:
        verticapy.options["time_on"] = time_on
        verticapy.options["query_on"] = False
        result = to_tablesample(
            "SELECT * FROM ({}) VERTICAPY_SUBTABLE LIMIT {}".format(query, limit),
            cursor,
        )
    except:
        verticapy.options["time_on"] = time_on_init
        verticapy.options["query_on"] = query_on_init
        raise
    verticapy.options["time_on"] = time_on_init
    verticapy.options["query_on"] = query_on_init
    result.count = count
    if verticapy.options["percent_bar"]:
        vdf = vdf_from_relation("({}) VERTICAPY_SUBTABLE".format(query), cursor=cursor)
        percent = vdf.agg(["percent"]).transpose().values
        for column in result.values:
            result.dtype[column] = vdf[column].ctype()
            result.percent[column] = percent[vdf_columns_names([column], vdf)[0]][0]
    return result


# ---#
def get_data_types(
    expr: str, cursor=None, column_name: str = "", schema_writing: str = ""
):
    """
---------------------------------------------------------------------------
Returns customized relation columns and the respective data types.
This process creates a temporary table.

Parameters
----------
expr: str
	An expression in pure SQL.
cursor: DBcursor, optional
	Vertica database cursor.
column_name: str, optional
	If not empty, it will return only the data type of the input column if it
	is in the relation.
schema_writing: str, optional
	Schema to use to create the temporary table. If empty, the function will 
	create a local temporary table.

Returns
-------
list of tuples
	The list of the different columns and their respective type.
	"""
    cursor, conn = check_cursor(cursor)[0:2]

    if isinstance(cursor, vertica_python.vertica.cursor.Cursor):
        try:
            if column_name:
                cursor.execute(expr)
                description = cursor.description[0]
                return type_code_dtype(
                    type_code=description[1],
                    display_size=description[2],
                    precision=description[4],
                    scale=description[5],
                )
            else:
                cursor.execute(expr)
                description, ctype = cursor.description, []
                for elem in description:
                    ctype += [
                        [
                            elem[0],
                            type_code_dtype(
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
    tmp_name = "_VERTICAPY_TEMPORARY_TABLE_{}".format(get_session(cursor))
    schema = "v_temp_schema" if not (schema_writing) else schema_writing
    try:
        cursor.execute("DROP TABLE IF EXISTS {}.{}".format(schema, tmp_name))
    except:
        pass
    try:
        if schema == "v_temp_schema":
            cursor.execute(
                "CREATE LOCAL TEMPORARY TABLE {} ON COMMIT PRESERVE ROWS AS {}".format(
                    tmp_name, expr
                )
            )
        else:
            cursor.execute(
                "CREATE TEMPORARY TABLE {}.{} ON COMMIT PRESERVE ROWS AS {}".format(
                    schema, tmp_name, expr
                )
            )
    except:
        cursor.execute("DROP TABLE IF EXISTS {}.{}".format(schema, tmp_name))
        raise
    query = "SELECT column_name, data_type FROM columns WHERE {}table_name = '{}' AND table_schema = '{}' ORDER BY ordinal_position".format(
        "column_name = '{}' AND ".format(column_name) if (column_name) else "",
        tmp_name,
        schema,
    )
    cursor.execute(query)
    if column_name:
        ctype = cursor.fetchone()[1]
    else:
        ctype = cursor.fetchall()
    cursor.execute("DROP TABLE IF EXISTS {}.{}".format(schema, tmp_name))
    if conn:
        conn.close()
    return ctype


# ---#
def pandas_to_vertica(
    df, cursor=None,
):
    """
---------------------------------------------------------------------------
Ingests a pandas DataFrame into the Vertica database by creating a CSV file 
and then using flex tables. This creates a temporary local table that
will be dropped at the end of the local session. Use vDataFrame.to_db 
to store it inside the database.

Parameters
----------
df: pandas.DataFrame
	The pandas.DataFrame to ingest.
cursor: DBcursor, optional
	Vertica database cursor.
	
Returns
-------
vDataFrame
	vDataFrame of the new relation.

See Also
--------
read_csv  : Ingests a CSV file into the Vertica database.
read_json : Ingests a JSON file into the Vertica database.
	"""
    cursor, conn = check_cursor(cursor)[0:2]
    name = "verticapy_df_{}".format(random.randint(10e5, 10e6))
    path = "{}.csv".format(name)
    try:
        df.to_csv(path, index=False)
        read_csv(path, cursor, table_name=name, temporary_local_table=True, parse_n_lines=10000)
        os.remove(path)
    except:
        os.remove(path)
        raise
    from verticapy import vDataFrame

    return vDataFrame(input_relation=name, schema="v_temp_schema", cursor=cursor)


# ---#
def pcsv(
    path: str,
    cursor=None,
    sep: str = ",",
    header: bool = True,
    header_names: list = [],
    na_rep: str = "",
    quotechar: str = '"',
    escape: str = "\\",
):
    """
---------------------------------------------------------------------------
Parses a CSV file using flex tables. It will identify the columns and their
respective types.

Parameters
----------
path: str
	Absolute path where the CSV file is located.
cursor: DBcursor, optional
	Vertica database cursor. 
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

Returns
-------
dict
	dictionary containing for each column its type.

See Also
--------
read_csv  : Ingests a CSV file into the Vertica database.
read_json : Ingests a JSON file into the Vertica database.
	"""
    cursor, conn = check_cursor(cursor)[0:2]
    flex_name = "VERTICAPY_{}_FLEX".format(get_session(cursor))
    cursor.execute(
        "CREATE FLEX LOCAL TEMP TABLE {}(x int) ON COMMIT PRESERVE ROWS;".format(
            flex_name
        )
    )
    header_names = (
        ""
        if not (header_names)
        else "header_names = '{}',".format(sep.join(header_names))
    )
    try:
        with open(path, "r") as fs:
            cursor.copy(
                "COPY {} FROM STDIN PARSER FCSVPARSER(type = 'traditional', delimiter = '{}', header = {}, {} enclosed_by = '{}', escape = '{}') NULL '{}';".format(
                    flex_name, sep, header, header_names, quotechar, escape, na_rep
                ),
                fs,
            )
    except:
        cursor.execute(
            "COPY {} FROM LOCAL '{}' PARSER FCSVPARSER(type = 'traditional', delimiter = '{}', header = {}, {} enclosed_by = '{}', escape = '{}') NULL '{}';".format(
                flex_name, path, sep, header, header_names, quotechar, escape, na_rep
            )
        )
    cursor.execute("SELECT compute_flextable_keys('{}');".format(flex_name))
    cursor.execute("SELECT key_name, data_type_guess FROM {}_keys".format(flex_name))
    result = cursor.fetchall()
    dtype = {}
    for column_dtype in result:
        try:
            query = 'SELECT (CASE WHEN "{}"=\'{}\' THEN NULL ELSE "{}" END)::{} AS "{}" FROM {} WHERE "{}" IS NOT NULL LIMIT 1000'.format(
                column_dtype[0],
                na_rep,
                column_dtype[0],
                column_dtype[1],
                column_dtype[0],
                flex_name,
                column_dtype[0],
            )
            cursor.execute(query)
            dtype[column_dtype[0]] = column_dtype[1]
        except:
            dtype[column_dtype[0]] = "Varchar(100)"
    cursor.execute("DROP TABLE {}".format(flex_name))
    if conn:
        conn.close()
    return dtype


# ---#
def pjson(path: str, cursor=None):
    """
---------------------------------------------------------------------------
Parses a JSON file using flex tables. It will identify the columns and their
respective types.

Parameters
----------
path: str
	Absolute path where the JSON file is located.
cursor: DBcursor, optional
	Vertica database cursor. 

Returns
-------
dict
	dictionary containing for each column its type.

See Also
--------
read_csv  : Ingests a CSV file into the Vertica database.
read_json : Ingests a JSON file into the Vertica database.
	"""
    cursor, conn = check_cursor(cursor)[0:2]
    flex_name = "VERTICAPY_{}_FLEX".format(get_session(cursor))
    cursor.execute(
        "CREATE FLEX LOCAL TEMP TABLE {}(x int) ON COMMIT PRESERVE ROWS;".format(
            flex_name
        )
    )
    if isinstance(cursor, vertica_python.vertica.cursor.Cursor):
        with open(path, "r") as fs:
            cursor.copy(
                "COPY {} FROM STDIN PARSER FJSONPARSER();".format(flex_name), fs
            )
    else:
        cursor.execute(
            "COPY {} FROM LOCAL '{}' PARSER FJSONPARSER();".format(
                flex_name, path.replace("'", "''")
            )
        )
    cursor.execute("SELECT compute_flextable_keys('{}');".format(flex_name))
    cursor.execute("SELECT key_name, data_type_guess FROM {}_keys".format(flex_name))
    result = cursor.fetchall()
    dtype = {}
    for column_dtype in result:
        dtype[column_dtype[0]] = column_dtype[1]
    cursor.execute("DROP TABLE " + flex_name)
    if conn:
        conn.close()
    return dtype


# ---#
def read_csv(
    path: str,
    cursor=None,
    schema: str = "",
    table_name: str = "",
    sep: str = ",",
    header: bool = True,
    header_names: list = [],
    dtype: dict = {},
    na_rep: str = "",
    quotechar: str = '"',
    escape: str = "\\",
    genSQL: bool = False,
    parse_n_lines: int = -1,
    insert: bool = False,
    temporary_table: bool = False,
    temporary_local_table: bool = True,
):
    """
---------------------------------------------------------------------------
Ingests a CSV file using flex tables.

Parameters
----------
path: str
	Absolute path where the CSV file is located.
cursor: DBcursor, optional
	Vertica database cursor.
schema: str, optional
	Schema where the CSV file will be ingested.
table_name: str, optional
	Final relation name.
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
	If set to True, the SQL code to use to create the final table will be 
	generated but not executed. It is a good way to change the final
	relation types or to customize the data ingestion.
parse_n_lines: int, optional
	If this parameter is greater than 0. A new file of 'parse_n_lines' lines
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
    must to be empty, otherwise this parameter is ignored.

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
            ("path", path, [str],),
            ("schema", schema, [str],),
            ("table_name", table_name, [str],),
            ("sep", sep, [str],),
            ("header", header, [bool],),
            ("header_names", header_names, [list],),
            ("na_rep", na_rep, [str],),
            ("dtype", dtype, [dict],),
            ("quotechar", quotechar, [str],),
            ("escape", escape, [str],),
            ("genSQL", genSQL, [bool],),
            ("parse_n_lines", parse_n_lines, [int, float],),
            ("insert", insert, [bool],),
            ("temporary_table", temporary_table, [bool],),
            ("temporary_local_table", temporary_local_table, [bool],),
        ]
    )
    if (schema):
        temporary_local_table = False
    elif temporary_local_table:
        schema = "v_temp_schema"
    else:
        schema = "public"
    assert dtype or not(header_names), ParameterError("dtype must be empty when using parameter 'header_names'.")
    assert not(temporary_table) or not(temporary_local_table), ParameterError("Parameters 'temporary_table' and 'temporary_local_table' can not be both set to True.")
    cursor = check_cursor(cursor)[0]
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
    table_name = table_name if (table_name) else path.split("/")[-1].split(".csv")[0]
    query = "SELECT column_name FROM columns WHERE table_name = '{}' AND table_schema = '{}' ORDER BY ordinal_position".format(
        table_name.replace("'", "''"), schema.replace("'", "''")
    )
    cursor.execute(query)
    result = cursor.fetchall()
    if (result != []) and not (insert):
        raise NameError(
            'The table "{}"."{}" already exists !'.format(schema, table_name)
        )
    elif (result == []) and (insert):
        raise MissingRelation(
            'The table "{}"."{}" doesn\'t exist !'.format(schema, table_name)
        )
    else:
        if not(temporary_local_table):
            input_relation = '"{}"."{}"'.format(schema, table_name)
        else:
            input_relation = '"{}"'.format(table_name)
        f = open(path, "r")
        file_header = f.readline().replace("\n", "").replace('"', "").split(sep)
        f.close()
        if (header_names == []) and (header):
            header_names = file_header
            for idx in range(len(header_names)):
                h = header_names[idx]
                n = len(h)
                while n > 0 and h[0] == ' ':
                    h = h[1:]
                    n -= 1
                while n > 0 and h[-1] == ' ':
                    h = h[:-1]
                    n -= 1
                header_names[idx] = h
        elif len(file_header) > len(header_names):
            header_names += [
                "ucol{}".format(i + len(header_names))
                for i in range(len(file_header) - len(header_names))
            ]
        if (parse_n_lines > 0) and not (insert):
            f = open(path, "r")
            f2 = open(path[0:-4] + "VERTICAPY_COPY.csv", "w")
            for i in range(parse_n_lines + int(header)):
                line = f.readline()
                f2.write(line)
            f.close()
            f2.close()
            path_test = path[0:-4] + "VERTICAPY_COPY.csv"
        else:
            path_test = path
        query1 = ""
        if not (insert):
            if not(dtype):
                dtype = pcsv(
                    path_test, cursor, sep, header, header_names, na_rep, quotechar, escape
                )
            if parse_n_lines > 0:
                os.remove(path[0:-4] + "VERTICAPY_COPY.csv")
            temp = "TEMPORARY " if temporary_table else ""
            temp = "LOCAL TEMPORARY " if temporary_local_table else ""
            query1 = "CREATE {}TABLE {}({}){};".format(
                temp,
                input_relation,
                ", ".join(
                    ['"{}" {}'.format(column, dtype[column]) for column in header_names]
                ),
                " ON COMMIT PRESERVE ROWS" if temp else ""
            )
        skip = " SKIP 1" if (header) else ""
        query2 = "COPY {}({}) FROM {} DELIMITER '{}' NULL '{}' ENCLOSED BY '{}' ESCAPE AS '{}'{};".format(
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
            print(query1 + "\n" + query2)
        else:
            if query1:
                cursor.execute(query1)
            if isinstance(cursor, vertica_python.vertica.cursor.Cursor):
                with open(path, "r") as fs:
                    cursor.copy(query2.format("STDIN"), fs)
            else:
                cursor.execute(query2.format("LOCAL '{}'".format(path)))
            if query1 and not(temporary_local_table):
                print(
                    "The table {} has been successfully created.".format(input_relation)
                )
            from verticapy import vDataFrame

            return vDataFrame(table_name, cursor, schema=schema)


# ---#
def read_json(
    path: str,
    cursor=None,
    schema: str = "public",
    table_name: str = "",
    usecols: list = [],
    new_name: dict = {},
    insert: bool = False,
    temporary_table: bool = False,
    temporary_local_table: bool = False,
    print_info: bool = True,
):
    """
---------------------------------------------------------------------------
Ingests a JSON file using flex tables.

Parameters
----------
path: str
	Absolute path where the JSON file is located.
cursor: DBcursor, optional
	Vertica database cursor.
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
    If set to True, a temporary local table will be created and the parameter
    'schema' is ignored.

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
            ("schema", schema, [str],),
            ("table_name", table_name, [str],),
            ("usecols", usecols, [list],),
            ("new_name", new_name, [dict],),
            ("insert", insert, [bool],),
            ("temporary_table", temporary_table, [bool],),
            ("temporary_local_table", temporary_local_table, [bool],),
        ]
    )
    if (schema):
        temporary_local_table = False
    elif temporary_local_table:
        schema = "v_temp_schema"
    else:
        schema = "public"
    assert not(temporary_table) or not(temporary_local_table), ParameterError("Parameters 'temporary_table' and 'temporary_local_table' can not be both set to True.")
    cursor = check_cursor(cursor)[0]
    file = path.split("/")[-1]
    file_extension = file[-4 : len(file)]
    if file_extension != "json":
        raise ExtensionError("The file extension is incorrect !")
    if not (table_name):
        table_name = path.split("/")[-1].split(".json")[0]
    query = "SELECT column_name, data_type FROM columns WHERE table_name = '{}' AND table_schema = '{}' ORDER BY ordinal_position".format(
        table_name.replace("'", "''"), schema.replace("'", "''")
    )
    cursor.execute(query)
    column_name = cursor.fetchall()
    if (column_name != []) and not (insert):
        raise NameError(
            'The table "{}"."{}" already exists !'.format(schema, table_name)
        )
    elif (column_name == []) and (insert):
        raise MissingRelation(
            'The table "{}"."{}" doesn\'t exist !'.format(schema, table_name)
        )
    else:
        if not(temporary_local_table):
            input_relation = '"{}"."{}"'.format(schema, table_name)
        else:
            input_relation = '"{}"'.format(table_name)
        flex_name = "VERTICAPY_" + str(get_session(cursor)) + "_FLEX"
        cursor.execute(
            "CREATE FLEX LOCAL TEMP TABLE {}(x int) ON COMMIT PRESERVE ROWS;".format(
                flex_name
            )
        )
        if isinstance(cursor, vertica_python.vertica.cursor.Cursor):
            with open(path, "r") as fs:
                cursor.copy(
                    "COPY {} FROM STDIN PARSER FJSONPARSER();".format(flex_name), fs
                )
        else:
            cursor.execute(
                "COPY {} FROM LOCAL '{}' PARSER FJSONPARSER();".format(
                    flex_name, path.replace("'", "''")
                )
            )
        cursor.execute("SELECT compute_flextable_keys('{}');".format(flex_name))
        cursor.execute(
            "SELECT key_name, data_type_guess FROM {}_keys".format(flex_name)
        )
        result = cursor.fetchall()
        dtype = {}
        for column_dtype in result:
            try:
                cursor.execute(
                    'SELECT "{}"::{} FROM {} LIMIT 1000'.format(
                        column_dtype[0], column_dtype[1], flex_name
                    )
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
            cursor.execute(
                "CREATE {}TABLE {} AS SELECT {} FROM {}".format(
                    temp, input_relation, ", ".join(cols), flex_name
                )
            )
            if not(temporary_local_table):
                print("The table {} has been successfully created.".format(input_relation))
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
            cursor.execute(
                "INSERT INTO {} SELECT {} FROM {}".format(
                    input_relation, ", ".join(final_transformation), flex_name
                )
            )
        cursor.execute("DROP TABLE {}".format(flex_name))
        from verticapy import vDataFrame

        return vDataFrame(table_name, cursor, schema=schema)


# ---#
def read_shp(
    path: str, cursor=None, schema: str = "public", table_name: str = "",
):
    """
---------------------------------------------------------------------------
Ingests a SHP file. For the moment, only files located in the Vertica server 
can be ingested.

Parameters
----------
path: str
    Absolute path where the SHP file is located.
cursor: DBcursor, optional
    Vertica database cursor.
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
            ("path", path, [str],),
            ("schema", schema, [str],),
            ("table_name", table_name, [str],),
        ]
    )
    file = path.split("/")[-1]
    file_extension = file[-3 : len(file)]
    if file_extension != "shp":
        raise ExtensionError("The file extension is incorrect !")
    cursor = check_cursor(cursor)[0]
    query = f"SELECT STV_ShpCreateTable(USING PARAMETERS file='{path}') OVER() AS create_shp_table;"
    cursor.execute(query)
    result = cursor.fetchall()
    if not (table_name):
        table_name = file[:-4]
    result[0] = [f'CREATE TABLE "{schema}"."{table_name}"(']
    result = [elem[0] for elem in result]
    result = "".join(result)
    cursor.execute(result)
    query = f'COPY "{schema}"."{table_name}" WITH SOURCE STV_ShpSource(file=\'{path}\') PARSER STV_ShpParser();'
    cursor.execute(query)
    print(f'The table "{schema}"."{table_name}" has been successfully created.')
    from verticapy import vDataFrame

    return vDataFrame(table_name, cursor, schema=schema)


# ---#
def set_option(option: str, value: (bool, int, str) = None):
    """
    ---------------------------------------------------------------------------
    Sets VerticaPy options.

    Parameters
    ----------
    option: str
        Option to use.
        cache        : bool
            If set to True, the vDataFrame will save in memory the computed
            aggregations.
        colors       : list
            List of the colors used to draw the graphics.
        max_rows     : int
            Maximum number of rows to display. If the parameter is incorrect, 
            nothing will be changed.
        max_columns  : int
            Maximum number of columns to display. If the parameter is incorrect, 
            nothing will be changed.
        mode         : str
            How to display VerticaPy outputs.
                full  : VerticaPy regular display mode.
                light : Minimalist display mode.
        percent_bar  : bool
            If set to True, it displays the percent of non-missing values.
        print_info   : bool
            If set to True, information will be printed each time the vDataFrame is 
            modified.
        random_state : int
            Integer used to seed the random number generation in VerticaPy.
        sql_on       : bool
            If set to True, displays all the SQL queries.
        time_on      : bool
            If set to True, displays all the SQL queries elapsed time.
    value: object, optional
        New value of option.
    """
    try:
        option = option.lower()
    except:
        pass
    check_types(
        [
            (
                "option",
                option,
                [
                    "cache",
                    "colors",
                    "color_style",
                    "max_rows",
                    "max_columns",
                    "percent_bar",
                    "print_info",
                    "random_state",
                    "sql_on",
                    "time_on",
                    "mode",
                ],
            ),
        ]
    )
    if option == "cache":
        check_types([("value", value, [bool])])
        if isinstance(value, bool):
            verticapy.options["cache"] = value
    elif option == "color_style":
        check_types([("value", value, ["rgb", "sunset", "retro", "shimbg", "swamp", "med", "orchid", "magenta", "orange", "vintage", "vivid", "berries", "refreshing", "summer", "tropical", "india", "default",])])
        if isinstance(value, str):
            verticapy.options["color_style"] = value
    elif option == "colors":
        check_types([("value", value, [list])])
        if isinstance(value, list):
            verticapy.options["colors"] = [str(elem) for elem in value]
    elif option == "max_rows":
        check_types([("value", value, [int, float])])
        if value >= 0:
            verticapy.options["max_rows"] = int(value)
    elif option == "max_columns":
        check_types([("value", value, [int, float])])
        if value > 0:
            verticapy.options["max_columns"] = int(value)
    elif option == "print_info":
        check_types([("value", value, [bool])])
        if isinstance(value, bool):
            verticapy.options["print_info"] = value
    elif option == "percent_bar":
        check_types([("value", value, [bool])])
        if value in (True, False, None):
            verticapy.options["percent_bar"] = value
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
        if isinstance(value, bool):
            verticapy.options["query_on"] = value
    elif option == "time_on":
        check_types([("value", value, [bool])])
        if isinstance(value, bool):
            verticapy.options["time_on"] = value
    elif option == "mode":
        check_types([("value", value, ["light", "full"])])
        if value.lower() in ["light", "full", None]:
            verticapy.options["mode"] = value.lower()
    else:
        raise ParameterError("Option '{}' does not exist.".format(option))


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
    ):
        check_types(
            [
                ("values", values, [dict],),
                ("dtype", dtype, [dict],),
                ("count", count, [int],),
                ("offset", offset, [int],),
                ("percent", percent, [dict],),
            ]
        )
        self.values = values
        self.dtype = dtype
        self.count = count
        self.offset = offset
        self.percent = percent
        for column in values:
            if column not in dtype:
                self.dtype[column] = "undefined"

    # ---#
    def __getitem__(self, key):
        all_cols = [elem for elem in self.values]
        for elem in all_cols:
            if str_column(str(elem).lower()) == str_column(str(key).lower()):
                key = elem
                break
        return self.values[key]

    # ---#
    def _repr_html_(self):
        if len(self.values) == 0:
            return ""
        n = len(self.values)
        dtype = self.dtype
        if n < verticapy.options["max_columns"]:
            data_columns = [[column] + self.values[column] for column in self.values]
        else:
            k = int(verticapy.options["max_columns"] / 2)
            columns = [elem for elem in self.values]
            values0 = [[columns[i]] + self.values[columns[i]] for i in range(k)]
            values1 = [["..." for i in range(len(self.values[columns[0]]) + 1)]]
            values2 = [
                [columns[i]] + self.values[columns[i]]
                for i in range(n - verticapy.options["max_columns"] + k, n)
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
        if n < verticapy.options["max_columns"]:
            data_columns = [[column] + self.values[column] for column in self.values]
        else:
            k = int(verticapy.options["max_columns"] / 2)
            columns = [elem for elem in self.values]
            values0 = [[columns[i]] + self.values[columns[i]] for i in range(k)]
            values1 = [["..." for i in range(len(self.values[columns[0]]) + 1)]]
            values2 = [
                [columns[i]] + self.values[columns[i]]
                for i in range(n - verticapy.options["max_columns"] + k, n)
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
                columns[idx] += [item]
        columns = [index] + columns
        values = {}
        for item in columns:
            values[item[0]] = item[1 : len(item)]
        return tablesample(values, self.dtype, self.count, self.offset, self.percent,)

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
        try:
            import pandas as pd
        except:
            raise ImportError(
                "The pandas module seems to not be installed in your environment.\nTo be able to use this method, you'll have to install it."
            )
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
                elif math.isnan(val):
                    val = "NULL"
                row += ["{} AS {}".format(val, '"' + column.replace('"', "") + '"')]
            sql += ["(SELECT {})".format(", ".join(row))]
        sql = " UNION ALL ".join(sql)
        return sql

    # ---#
    def to_vdf(self, cursor=None, dsn: str = ""):
        """
	---------------------------------------------------------------------------
	Converts the tablesample to a vDataFrame.

	Parameters
	----------
	cursor: DBcursor, optional
		Vertica database cursor. 
	dsn: str, optional
		Database DSN.

 	Returns
 	-------
 	vDataFrame
 		vDataFrame of the tablesample.

	See Also
	--------
	tablesample.to_pandas : Converts the tablesample to a pandas DataFrame.
	tablesample.to_sql    : Generates the SQL query associated to the tablesample.
		"""
        check_types([("dsn", dsn, [str],)])
        if not (cursor) and not (dsn):
            cursor = read_auto_connect().cursor()
        elif not (cursor):
            cursor = vertica_conn(dsn).cursor()
        else:
            check_cursor(cursor)
        relation = "({}) sql_relation".format(self.to_sql())
        return vdf_from_relation(relation, cursor=cursor, dsn=dsn)


# ---#
def to_tablesample(
    query: str, cursor=None, title: str = "",
):
    """
	---------------------------------------------------------------------------
	Returns the result of a SQL query as a tablesample object.

	Parameters
	----------
	query: str, optional
		SQL Query. 
	cursor: DBcursor, optional
		Vertica database cursor.
	title: str, optional
		Query title when the query is displayed.

 	Returns
 	-------
 	tablesample
 		Result of the query.

	See Also
	--------
	tablesample : Object in memory created for rendering purposes.
	"""
    check_types([("query", query, [str],)])
    cursor, conn = check_cursor(cursor)[0:2]
    if verticapy.options["query_on"]:
        print_query(query, title)
    start_time = time.time()
    cursor.execute(query)
    description, dtype = cursor.description, {}
    for elem in description:
        dtype[elem[0]] = type_code_dtype(
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
    for elem in values:
        if elem != "index":
            for idx in range(len(values[elem])):
                if isinstance(values[elem][idx], decimal.Decimal):
                    values[elem][idx] = float(values[elem][idx])
    if conn:
        conn.close()
    return tablesample(values=values, dtype=dtype)


# ---#
def vdf_from_relation(
    relation: str,
    name: str = "VDF",
    cursor=None,
    dsn: str = "",
    schema: str = "public",
    schema_writing: str = "",
    history: list = [],
    saving: list = [],
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
cursor: DBcursor, optional
	Vertica database cursor. 
    For a cursor designed by Vertica, see vertica_python
    For ODBC, see pyodbc.
    For JDBC, see jaydebeapi.
    For help, see utilities.vHelp.
dsn: str, optional
	Database DSN. OS File including the DB credentials.
    VerticaPy will try to create a vertica_python cursor first.
	If it didn't find the library, it will try to create a pyodbc cursor.
	For help, see utilities.vHelp.
schema: str, optional
	Relation schema. It can be to use to be less ambiguous and allow to create schema 
	and relation name with dots '.' inside.
schema_writing: str, optional
	Schema to use to create the temporary table. If empty, the function will create 
	a local temporary table.
history: list, optional
	vDataFrame history (user modifications). to use to keep the previous vDataFrame
	history.
saving: list, optional
	List to use to reconstruct the vDataFrame from previous transformations.

Returns
-------
vDataFrame
	The vDataFrame associated to the input relation.
	"""
    check_types(
        [
            ("relation", relation, [str],),
            ("name", name, [str],),
            ("dsn", dsn, [str],),
            ("schema", schema, [str],),
            ("history", history, [list],),
            ("saving", saving, [list],),
        ]
    )
    name = gen_name([name])
    from verticapy import vDataFrame

    vdf = vDataFrame("", empty=True)
    vdf._VERTICAPY_VARIABLES_["dsn"] = dsn
    if not (cursor) and not (dsn):
        cursor = read_auto_connect().cursor()
    elif not (cursor):
        cursor = vertica_conn(dsn).cursor()
    else:
        check_cursor(cursor)
    vdf._VERTICAPY_VARIABLES_["input_relation"] = name
    vdf._VERTICAPY_VARIABLES_["main_relation"] = relation
    vdf._VERTICAPY_VARIABLES_["schema"] = schema
    vdf._VERTICAPY_VARIABLES_["schema_writing"] = schema_writing
    vdf._VERTICAPY_VARIABLES_["cursor"] = cursor
    vdf._VERTICAPY_VARIABLES_["where"] = []
    vdf._VERTICAPY_VARIABLES_["order_by"] = {}
    vdf._VERTICAPY_VARIABLES_["exclude_columns"] = []
    vdf._VERTICAPY_VARIABLES_["history"] = history
    vdf._VERTICAPY_VARIABLES_["saving"] = saving
    try:
        cursor.execute(
            "DROP TABLE IF EXISTS v_temp_schema.VERTICAPY_{}_TEST;".format(name)
        )
    except:
        pass
    cursor.execute(
        "CREATE LOCAL TEMPORARY TABLE VERTICAPY_{}_TEST ON COMMIT PRESERVE ROWS AS SELECT * FROM {} LIMIT 10;".format(
            name, relation
        )
    )
    cursor.execute(
        "SELECT column_name, data_type FROM columns WHERE table_name = 'VERTICAPY_{}_TEST' AND table_schema = 'v_temp_schema' ORDER BY ordinal_position".format(
            name
        )
    )
    result = cursor.fetchall()
    cursor.execute("DROP TABLE IF EXISTS v_temp_schema.VERTICAPY_{}_TEST;".format(name))
    vdf._VERTICAPY_VARIABLES_["columns"] = ['"' + item[0] + '"' for item in result]
    for column, ctype in result:
        if '"' in column:
            warning_message = "A double quote \" was found in the column {}, its alias was changed using underscores '_' to {}".format(
                column, column.replace('"', "_")
            )
            warnings.warn(warning_message, Warning)
        from verticapy.vcolumn import vColumn

        new_vColumn = vColumn(
            '"{}"'.format(column.replace('"', "_")),
            parent=vdf,
            transformations=[
                (
                    '"{}"'.format(column.replace('"', '""')),
                    ctype,
                    category_from_type(ctype),
                )
            ],
        )
        setattr(vdf, '"{}"'.format(column.replace('"', "_")), new_vColumn)
        setattr(vdf, column.replace('"', "_"), new_vColumn)
    return vdf


# ---#
def version(cursor=None, condition: list = []):
    """
---------------------------------------------------------------------------
Returns the Vertica Version.

Parameters
----------
cursor: DBcursor, optional
    Vertica database cursor.
condition: list, optional
    List of the minimal version information. If the current version is not
    greater or equal to this one, it will raise an error.

Returns
-------
list
    List containing the version information.
    [MAJOR, MINOR, PATCH, POST]
    """
    check_types(
        [("condition", condition, [list],),]
    )
    cursor, conn = check_cursor(cursor)[0:2]
    if condition:
        condition = condition + [0 for elem in range(4 - len(condition))]
    version = (
        cursor.execute("SELECT version();")
        .fetchone()[0]
        .split("Vertica Analytic Database v")[1]
    )
    version = version.split(".")
    result = []
    try:
        result += [int(version[0])]
        result += [int(version[1])]
        result += [int(version[2].split("-")[0])]
        result += [int(version[2].split("-")[1])]
    except:
        pass
    if conn:
        conn.close()
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
                "This Function is not available for Vertica version {}.\nPlease upgrade your Vertica version to at least {} to get this functionality.".format(
                    version[0] + "." + version[1] + "." + version[2].split("-")[0],
                    ".".join([str(elem) for elem in condition[:3]]),
                )
            )
    return result


# ---#
def vHelp():
    """
---------------------------------------------------------------------------
VERTICAPY Interactive Help (FAQ).
	"""
    import verticapy

    try:
        from IPython.core.display import HTML, display, Markdown
    except:
        pass
    path = os.path.dirname(verticapy.__file__)
    img1 = '<center><img src="data:image/png;base64,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" style="width:80px;"></center>'
    img2 = "              ____________       ______\n"
    img2 += "             / __          `\\     /     /\n"
    img2 += "            |  \\/         /    /     /\n"
    img2 += "            |______      /    /     /\n"
    img2 += "                   |____/    /     /\n"
    img2 += "          _____________     /     /\n"
    img2 += "          \\           /    /     /\n"
    img2 += "           \\         /    /     /\n"
    img2 += "            \\_______/    /     /\n"
    img2 += "             ______     /     /\n"
    img2 += "             \\    /    /     /\n"
    img2 += "              \\  /    /     /\n"
    img2 += "               \\/    /     /\n"
    img2 += "                    /     /\n"
    img2 += "                   /     /\n"
    img2 += "                   \\    /\n"
    img2 += "                    \\  /\n"
    img2 += "                     \\/\n"
    message = img1 if (isnotebook()) else img2
    message += "\n\n&#128226; Welcome to the <b>VERTICAPY</b> help Module. You are about to use a new fantastic way to analyze your data !\n\nYou can learn quickly how to set up a connection, how to create a Virtual DataFrame and much more.\n\nWhat do you want to know?\n - <b>[Enter  0]</b> Do you want to know why you should use this library ?\n - <b>[Enter  1]</b> You don't have data to play with and you want to load an available dataset ?\n - <b>[Enter  2]</b> Do you want to look at a quick example ?\n - <b>[Enter  3]</b> Do you want to get a link to the VERTICAPY github ?\n - <b>[Enter  4]</b> Do you want to know how to display the Virtual DataFrame SQL code generation and the time elapsed to run the query ?\n - <b>[Enter  5]</b> Do you want to know how to load your own dataset inside Vertica ?\n - <b>[Enter 6]</b> Do you want to know how you writing direct SQL queries in Jupyter ?\n - <b>[Enter -1]</b> Exit"
    if not (isnotebook()):
        message = (
            message.replace("<b>", "")
            .replace("</b>", "")
            .replace("&#128226;", "\u26A0")
        )
    display(Markdown(message)) if (isnotebook()) else print(message)
    try:
        response = int(input())
    except:
        print("The choice is incorrect.\nPlease enter a number between 0 and 11.")
        try:
            response = int(input())
        except:
            print(
                "The choice is still incorrect.\nRerun the help function when you need help."
            )
            return
    if response == 0:
        message = "# VerticaPy\nNowadays, The 'Big Data' (Tb of data) is one of the main topics in the Data Science World. Data Scientists are now very important for any organisation. Becoming Data-Driven is mandatory to survive. Vertica is the first real analytic columnar Database and is still the fastest in the market. However, SQL is not enough flexible to be very popular for Data Scientists. Python flexibility is priceless and provides to any user a very nice experience. The level of abstraction is so high that it is enough to think about a function to notice that it already exists. Many Data Science APIs were created during the last 15 years and were directly adopted by the Data Science community (examples: pandas and scikit-learn). However, Python is only working in-memory for a single node process. Even if some famous highly distributed programming languages exist to face this challenge, they are still in-memory and most of the time they can not process on all the data. Besides, moving the data can become very expensive. Data Scientists must also find a way to deploy their data preparation and their models. We are far away from easiness and the entire process can become time expensive. \nThe idea behind VERTICAPY is simple: Combining the Scalability of VERTICA with the Flexibility of Python to give to the community what they need *Bringing the logic to the data and not the opposite*. This version 1.0 is the work of 3 years of new ideas and improvement.\nMain Advantages:\n - easy Data Exploration.\n - easy Data Preparation.\n - easy Data Modeling.\n - easy Model Evaluation.\n - easy Model Deployment.\n - most of what pandas.DataFrame can do, verticapy.vdataframe can do (and even much more)\n - easy ML model creation and evaluation.\n - many scikit functions and algorithms are available (and scalable!).\n\n&#9888; Please read the VERTICAPY Documentation. If you do not have time just read below.\n\n&#9888; The previous API is really nothing compare to the new version and many methods and functions totally changed. Consider this API as a totally new one.\nIf you have any feedback about the library, please contact me: <a href=\"mailto:badr.ouali@vertica.com\">badr.ouali@vertica.com</a>"
    elif response == 1:
        message = "In VERTICAPY many datasets (titanic, iris, smart_meters, amazon, winequality) are already available to be ingested in your Vertica Database.\n\nTo ingest a dataset you can use the associated load function.\n\n<b>Example:</b>\n\n```python\nfrom vertica_python.learn.datasets import load_titanic\nvdf = load_titanic(db_cursor)\n```"
    elif response == 2:
        message = '## Quick Start\nInstall the library using the <b>pip</b> command.\n```\nroot@ubuntu:~$ pip3 install verticapy\n```\nInstall vertica_python to create a database cursor.\n```shell\nroot@ubuntu:~$ pip3 install vertica_python\n```\nCreate a vertica connection\n```python\nfrom verticapy import vertica_conn\ncur = vertica_conn("VerticaDSN").cursor()\n```\nCreate the Virtual DataFrame of your relation:\n```python\nfrom verticapy import vDataFrame\nvdf = vDataFrame("my_relation", cursor = cur)\n```\nIf you don\'t have data to play with, you can easily load well known datasets\n```python\nfrom verticapy.datasets import load_titanic\nvdf = load_titanic(cursor = cur)\n```\nExamine your data:\n```python\nvdf.describe()\n# Output\n               min       25%        50%        75%   \nage           0.33      21.0       28.0       39.0   \nbody           1.0     79.25      160.5      257.5   \nfare           0.0    7.8958    14.4542    31.3875   \nparch          0.0       0.0        0.0        0.0   \npclass         1.0       1.0        3.0        3.0   \nsibsp          0.0       0.0        0.0        1.0   \nsurvived       0.0       0.0        0.0        1.0   \n                   max    unique  \nage               80.0        96  \nbody             328.0       118  \nfare          512.3292       277  \nparch              9.0         8  \npclass             3.0         3  \nsibsp              8.0         7  \nsurvived           1.0         2 \n```\nPrint the SQL query with the <b>set_display_parameters</b> method:\n```python\nset_option(\'sql_on\', True)\nvdf.describe()\n# Output\n## Compute the descriptive statistics of all the numerical columns ##\nSELECT\n\tSUMMARIZE_NUMCOL("age","body","survived","pclass","parch","fare","sibsp") OVER ()\nFROM public.titanic\n```\nWith VerticaPy, it is now possible to solve a ML problem with few lines of code.\n```python\nfrom verticapy.learn.model_selection import cross_validate\nfrom verticapy.learn.ensemble import RandomForestClassifier\n# Data Preparation\nvdf["sex"].label_encode()["boat"].fillna(method = "0ifnull")["name"].str_extract(\' ([A-Za-z]+)\\.\').eval("family_size", expr = "parch + sibsp + 1").drop(columns = ["cabin", "body", "ticket", "home.dest"])["fare"].fill_outliers().fillna().to_db("titanic_clean")\n# Model Evaluation\ncross_validate(RandomForestClassifier("rf_titanic", cur, max_leaf_nodes = 100, n_estimators = 30), "titanic_clean", ["age", "family_size", "sex", "pclass", "fare", "boat"], "survived", cutoff = 0.35)\n# Output\n                           auc               prc_auc   \n1-fold      0.9877114427860691    0.9530465915039339   \n2-fold      0.9965555014605642    0.7676485351425721   \n3-fold      0.9927239216549301    0.6419135521132449   \navg             0.992330288634        0.787536226253   \nstd           0.00362128464093         0.12779562393   \n                     accuracy              log_loss   \n1-fold      0.971291866028708    0.0502052541223871   \n2-fold      0.983253588516746    0.0298167751798457   \n3-fold      0.964824120603015    0.0392745694400433   \navg            0.973123191716       0.0397655329141   \nstd           0.0076344236729      0.00833079837099   \n                     precision                recall   \n1-fold                    0.96                  0.96   \n2-fold      0.9556962025316456                   1.0   \n3-fold      0.9647887323943662    0.9383561643835616   \navg             0.960161644975        0.966118721461   \nstd           0.00371376912311        0.025535200301   \n                      f1-score                   mcc   \n1-fold      0.9687259282082884    0.9376119402985075   \n2-fold      0.9867172675521821    0.9646971010878469   \n3-fold      0.9588020287309097    0.9240569687684576   \navg              0.97141507483        0.942122003385   \nstd            0.0115538960753       0.0168949813163   \n                  informedness            markedness   \n1-fold      0.9376119402985075    0.9376119402985075   \n2-fold      0.9737827715355807    0.9556962025316456   \n3-fold      0.9185148945422918    0.9296324823943662   \navg             0.943303202125        0.940980208408   \nstd            0.0229190954261       0.0109037699717   \n                           csi  \n1-fold      0.9230769230769231  \n2-fold      0.9556962025316456  \n3-fold      0.9072847682119205  \navg             0.928685964607  \nstd            0.0201579224026\n```\nEnjoy!'
    elif response == 3:
        if not (isnotebook()):
            message = "Please go to https://github.com/vertica/VerticaPy/"
        else:
            message = "Please go to <a href='https://github.com/vertica/VerticaPy/wiki'>https://github.com/vertica/VerticaPy/</a>"
    elif response == 4:
        message = "You can Display the SQL Code generation & elapsed time of the Virtual DataFrame using the <b>set_option</b> function.\nIt is also possible to print the current Virtual DataFrame relation using the <b>current_relation</b> method.\n"
    elif response == 5:
        message = "VERTICAPY allows you many ways to ingest data file. It is using Vertica Flex Tables to identify the columns types and store the data inside Vertica. These functions will also return the associated Virtual DataFrame.\n\nLet's load the data from the 'data.csv' file.\n\n\n```python\nfrom verticapy import read_csv\nvdf = read_csv('data.csv', db_cursor)\n```\n\nThe same applies to json. Let's consider the file 'data.json'.\n\n\n```python\nfrom verticapy import read_json\nvdf = read_json('data.json', db_cursor)\n```\n\n"
    elif response == 6:
        message = "VerticaPy SQL Magic offers you a nice way to interact with Vertica. You can load the extension using the following command:\n```\n%load_ext verticapy.sql\n```\nYou can then run your own SQL queries.\n```\n%%sql\nSELECT * FROM public.titanic\n```"
    elif response == -1:
        message = "Thank you for using the VERTICAPY help."
    elif response == 666:
        message = "Thank you so much for using this library. My only purpose is to solve real Big Data problems in the context of Data Science. I worked years to be able to create this API and give you a real way to analyse your data.\n\nYour devoted Data Scientist: <i>Badr Ouali</i>"
    else:
        message = "The choice is incorrect.\nPlease enter a number between 0 and 11."
    if not (isnotebook()):
        message = (
            message.replace("<b>", "")
            .replace("</b>", "")
            .replace("&#128226;", "\u26A0")
            .replace("&#128540;", ";p")
            .replace("&#9888;", "\u26A0")
        )
        message = message.replace(
            '<a href="mailto:badr.ouali@vertica.com">badr.ouali@vertica.com</a>',
            "badr.ouali@vertica.com",
        )
    display(Markdown(message)) if (isnotebook()) else print(message)


# DEPRECATED
#########################
def drop_table(
    name: str, cursor=None, raise_error: bool = False,
):
    warnings.warn(
        "utilities.drop_table has been deprecated. It will be removed in v0.5.1. Use utilities.drop with parameter 'method' set to 'table' instead",
        Warning,
    )
    return drop(name, cursor, raise_error, "table")


def drop_view(
    name: str, cursor=None, raise_error: bool = False,
):
    warnings.warn(
        "utilities.drop_view has been deprecated. It will be removed in v0.5.1. Use utilities.drop_view with parameter 'method' set to 'view' instead",
        Warning,
    )
    return drop(name, cursor, raise_error, "view")


def drop_text_index(
    name: str, cursor=None, raise_error: bool = False,
):
    warnings.warn(
        "utilities.drop_text_index has been deprecated. It will be removed in v0.5.1. Use utilities.drop_text_index with parameter 'method' set to 'text' instead",
        Warning,
    )
    return drop(name, cursor, raise_error, "text")


def drop_model(
    name: str, cursor=None, raise_error: bool = False,
):
    warnings.warn(
        "utilities.drop_model has been deprecated. It will be removed in v0.5.1. Use utilities.drop with parameter 'method' set to 'model' instead",
        Warning,
    )
    return drop(name, cursor, raise_error, "model")


#########################
