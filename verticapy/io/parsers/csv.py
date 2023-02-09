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

#
#
# Modules
#
# Standard Python Modules
import os, warnings

# VerticaPy Modules
import verticapy as vp
from verticapy.utils._decorators import save_verticapy_logs
from verticapy.utils._toolbox import *
from verticapy.errors import ExtensionError, ParameterError, MissingRelation
from ..flex import compute_flextable_keys
from verticapy.io.sql._utils._format import format_schema_table, clean_query
from verticapy.io.parsers._utils import extract_compression, get_first_file


def guess_sep(file_str: str):
    sep = ","
    max_occur = file_str.count(",")
    for s in ("|", ";"):
        total_occurences = file_str.count(s)
        if total_occurences > max_occur:
            max_occur = total_occurences
            sep = s
    return sep


def erase_space_start_end_in_list_values(L: list):
    L_tmp = [elem for elem in L]
    for idx in range(len(L_tmp)):
        L_tmp[idx] = L_tmp[idx].strip()
    return L_tmp


def get_header_name_csv(path: str, sep: str):
    f = open(path, "r")
    file_header = f.readline().replace("\n", "").replace('"', "")
    if not (sep):
        sep = guess_sep(file_header)
    file_header = file_header.split(sep)
    f.close()
    for idx, col in enumerate(file_header):
        if col == "":
            if idx == 0:
                position = "beginning"
            elif idx == len(file_header) - 1:
                position = "end"
            else:
                position = "middle"
            file_header[idx] = f"col{idx}"
            warning_message = (
                f"An inconsistent name was found in the {position} of the "
                "file header (isolated separator). It will be replaced "
                f"by col{idx}."
            )
            if idx == 0:
                warning_message += (
                    "\nThis can happen when exporting a pandas DataFrame "
                    "to CSV while retaining its indexes.\nTip: Use "
                    "index=False when exporting with pandas.DataFrame.to_csv."
                )
            warnings.warn(warning_message, Warning)
    return erase_space_start_end_in_list_values(file_header)


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
    from ..sql.drop import drop

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
    from ..sql.create import create_table

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
                and vp.OPTIONS["print_info"]
            ):
                print(f"The table {input_relation} has been successfully created.")
            return vDataFrame(table_name, schema=schema)
