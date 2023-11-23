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
from typing import Optional

import verticapy._config.config as conf
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import (
    clean_query,
    format_schema_table,
    format_type,
    quote_ident,
)
from verticapy._utils._gen import gen_tmp_name
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._vertica_version import check_minimum_version
from verticapy.errors import ExtensionError

from verticapy.core.parsers._utils import extract_col_dt_from_query, extract_compression
from verticapy.core.parsers.csv import read_csv
from verticapy.core.vdataframe.base import vDataFrame

from verticapy.sql.drop import drop


@check_minimum_version
@save_verticapy_logs
def read_file(
    path: str,
    schema: Optional[str] = None,
    table_name: Optional[str] = None,
    dtype: Optional[dict] = None,
    unknown: str = "varchar",
    varchar_varbinary_length: int = 80,
    numeric_precision_scale: tuple[int, int] = (37, 15),
    insert: bool = False,
    temporary_table: bool = False,
    temporary_local_table: bool = True,
    gen_tmp_table_name: bool = True,
    ingest_local: bool = False,
    genSQL: bool = False,
    max_files: int = 100,
) -> vDataFrame:
    """
    Inspects and ingests a file in CSV, Parquet, ORC,
    JSON, or Avro format.
    This function uses the Vertica complex data type.
    For new table creation, the file must be located
    in the server.

    Parameters
    ----------
    path: str
        Path to a file or glob. Valid  paths include  any
        path that is valid  for COPY  and that uses a  file
        format supported by this function.
        When inferring  the data type, only  one file  will
        be read,  even if a glob specifies  multiple files.
        However,  in the case of  JSON, more than one  file
        may be read to infer the data type.
    schema: str, optional
        Schema in which to create the table.
    table_name: str, optional
        Name of the table to create. If empty, the file name
        is used.
    dtype: dict, optional
        Dictionary of customised  data  type.  The predicted
        data  types  will  be  replaced  by  the input  data
        types. The  dictionary must include the name  of the
        column as key and the new data type as value.
    unknown: str, optional
        Type used to replace unknown data types.
    varchar_varbinary_length: int, optional
        Default  length  of  varchar  and  varbinary columns.
    insert: bool, optional
        If set to True, the  data is ingested into  the input
        relation.
        When  you  set this  parameter to True, most  of  the
        parameters are ignored.
    temporary_table: bool, optional
        If set to True, a temporary table is created.
    temporary_local_table: bool, optional
        If set to True, a  temporary local table is  created.
        The  parameter  'schema'  must  be  empty,  otherwise
        this parameter is ignored.
    gen_tmp_table_name: bool, optional
        Sets the name of the temporary table.  This parameter
        is     only      used      when     the     parameter
        'temporary_local_table'  is  set  to  True  and   the
        parameters "table_name" and "schema" are unspecified.
    ingest_local: bool, optional
        If set to True,  the  file is ingested from  the local
        machine. This currently only works for data insertion.
    genSQL: bool, optional
        If set to True,  the SQL  code for creating the  final
        table  is generated but  not executed. This is a  good
        way to change the final relation types or to customize
        the data ingestion.
    max_files: int, optional
        (JSON only.)  If  path  is a  glob, specifies  maximum
        number of files in path to inspect. Use this parameter
        to increase the amount of data  the function considers.
        This can be beneficial  if you suspect variation among
        files.  Files  are  chosen  arbitrarily  from the glob.
        The default value is 100.

    Returns
    -------
    vDataFrame
        The vDataFrame of the relation.
    """
    dtype = format_type(dtype, dtype=dict)
    assert not ingest_local or insert, ValueError(
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
        if not table_name:
            raise ValueError(
                "Parameter 'table_name' must be defined when parameter 'insert' is set to True."
            )
        if not schema and temporary_local_table:
            schema = "v_temp_schema"
        elif not schema:
            schema = conf.get_option("temp_schema")
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
        _executeSQL(query, title="Inserting the data.")
        return vDataFrame(table_name, schema=schema)
    if schema:
        temporary_local_table = False
    elif temporary_local_table:
        schema = "v_temp_schema"
    else:
        schema = conf.get_option("temp_schema")
    basename = ".".join(path.split("/")[-1].split(".")[0:-1])
    if gen_tmp_table_name and temporary_local_table and not table_name:
        table_name = gen_tmp_name(name=basename)
    if not table_name:
        table_name = basename
    result = _executeSQL(
        query=f"""
            SELECT INFER_TABLE_DDL ('{path}' 
                                    USING PARAMETERS
                                    format='{file_format}',
                                    table_name='y_verticapy',
                                    table_schema='x_verticapy',
                                    table_type='native',
                                    with_copy_statement=true,
                                    one_line_result=true,
                                    max_files={max_files},
                                    max_candidates=1);""",
        title="Generating the CREATE and COPY statement.",
        method="fetchfirstelem",
    )
    result = result.replace("UNKNOWN", unknown)
    result = "create" + "create".join(result.split("create")[1:])
    relation = format_schema_table(schema, table_name)
    if temporary_local_table:
        create_statement = f"CREATE LOCAL TEMPORARY TABLE {quote_ident(table_name)}"
    else:
        if not schema:
            schema = conf.get_option("temp_schema")
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
        result[1] = "copy" + result[1].replace(
            '"x_verticapy"."y_verticapy"',
            relation,
        )
    else:
        if temporary_local_table:
            end = result.split(")")[-1]
            result = result.split(")")[0:-1] + ") ON COMMIT PRESERVE ROWS" + end
        result = [result]
    if varchar_varbinary_length != 80:
        result[0] = result[0].replace(
            " varchar", f" varchar({varchar_varbinary_length})"
        )
        result[0] = result[0].replace(
            " varbinary", f" varbinary({varchar_varbinary_length})"
        )
        result[0] = result[0].replace(
            "precision, scale",
            f"{numeric_precision_scale[0], numeric_precision_scale[1]}",
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
        _executeSQL(
            result,
            title="Creating the table and ingesting the data.",
        )
    else:
        _executeSQL(
            result[0],
            title="Creating the table.",
        )
        try:
            _executeSQL(
                result[1],
                title="Ingesting the data.",
            )
        except:
            drop(f'"{schema}"."{table_name}"', method="table")
            raise
    return vDataFrame(input_relation=table_name, schema=schema)
