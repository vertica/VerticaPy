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
import os
from typing import Optional

from vertica_python.errors import QueryError

import verticapy._config.config as conf
from verticapy._typing import NoneType
from verticapy._utils._gen import gen_tmp_name
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import (
    quote_ident,
    format_schema_table,
    format_type,
    clean_query,
)
from verticapy._utils._sql._sys import _executeSQL
from verticapy.errors import ExtensionError, MissingRelation

from verticapy.core.parsers._utils import extract_compression
from verticapy.core.parsers.all import read_file
from verticapy.core.vdataframe.base import vDataFrame

from verticapy.sql.drop import drop
from verticapy.sql.flex import compute_flextable_keys


def pjson(path: str, ingest_local: bool = True) -> dict[str, str]:
    """
    Parses a JSON file using flex tables. It identifies
    the columns and their respective types.

    Parameters
    ----------
    path: str
        Absolute path where the JSON file is located.
    ingest_local: bool, optional
        If  set to  True,  the file will be  ingested
        from the local machine.

    Returns
    -------
    dict
        dictionary containing column names and their SQL
        data type.
    """
    flex_name = gen_tmp_name(name="flex")[1:-1]
    _executeSQL(
        query=f"""
            CREATE FLEX LOCAL TEMP TABLE {flex_name}
            (x int) ON COMMIT PRESERVE ROWS;""",
        title="Creating a flex table.",
    )
    path_str = path.replace("'", "''")
    local = " LOCAL" if ingest_local else ""
    _executeSQL(
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


@save_verticapy_logs
def read_json(
    path: str,
    schema: Optional[str] = None,
    table_name: Optional[str] = None,
    usecols: Optional[list] = None,
    new_name: Optional[dict] = None,
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
) -> vDataFrame:
    """
    Ingests a JSON file using flex tables.

    Parameters
    ----------
    path: str
        Absolute path where the JSON file is located.
    schema: str, optional
        Schema  where the JSON file will be ingested.
    table_name: str, optional
        Final relation name.
    usecols: list, optional
        List of the JSON parameters to ingest. The
        other parameters will be ignored. If empty,
        all the JSON parameters will be ingested.
    new_name: dict, optional
        Dictionary of the new column names. If the JSON
        file is nested, it is recommended to change the
        final names because special characters will be
        included in the new column names.
        For example, {"param": {"age": 3,
                                "name": Badr},
                      "date": 1993-03-11}
        will create 3 columns: "param.age", "param.name"
        and "date".  You can  rename these columns using
        the  'new_name'  parameter  with  the  following
        dictionary: {"param.age": "age",
                     "param.name": "name"}
    insert: bool, optional
        If set to True, the data  is ingested  into  the
        input relation. The JSON parameters must  be the
        same  as the input  relation otherwise they will
        not be ingested. If set to True, table_name
        cannot be empty.
    start_point: str, optional
        String,  name  of a key in the JSON load data  at
        which to  begin  parsing. The  parser ignores all
        data  before the start_point value.  The value is
        loaded  for each object in the file.  The  parser
        processes data after  the first  instance, and up
        to the second, ignoring any remaining data.
    record_terminator: str, optional
        When  set,  any invalid JSON records are  skipped
        and parsing continues with the next record.
        Records must be terminated uniformly. For example,
        if your input file has  JSON records terminated by
        newline  characters,  set this parameter to '\n').
        If  any  invalid   JSON   records  exist,  parsing
        continues after the next record_terminator.
        Even if the data does not contain invalid records,
        specifying  an  explicit   record  terminator  can
        improve  load performance by allowing  cooperative
        parse   and  apportioned  load  to  operate   more
        efficiently.
        When you omit this parameter, parsing ends at the
        first invalid JSON record.
    suppress_nonalphanumeric_key_chars: bool, optional
        Boolean,  whether  to   suppress  non-alphanumeric
        characters in JSON key values. The parser replaces
        these characters with  an underscore (_) when this
        parameter is true.
    reject_on_materialized_type_error: bool, optional
        Boolean, whether to reject a data row that contains
        a materialized column  value that cannot be coerced
        into a compatible data type.  If the value is false
        and the type cannot be coerced, the parser sets the
        value in that column to null.
        If the column is a  strongly-typed complex type, as
        opposed  to a  flexible  complex type, then a  type
        mismatch  anywhere  in the complex type causes  the
        entire  column  to  be treated as a  mismatch.  The
        parser does not partially load complex types.
    reject_on_duplicate: bool, optional
        Boolean, whether to ignore duplicate records (false),
        or to reject  duplicates (true).  In either case, the
        load continues.
    reject_on_empty_key: bool, optional
        Boolean, whether to reject any row containing a field
        key without a value.
    flatten_maps: bool, optional
        Boolean, whether to flatten sub-maps within the JSON
        data, separating map levels  with a period (.). This
        value affects all data in the load, including nested
        maps.
    flatten_arrays: bool, optional
        Boolean,  whether  to convert lists to sub-maps with
        integer keys.
        When lists are flattened,  key names are concatenated
        in the same way as maps. Lists are not flattened by
        default. This value affects all data  in the load,
        including nested lists.
    temporary_table: bool, optional
        If set to True, a temporary table will be created.
    temporary_local_table: bool, optional
        If  set  to  True, a temporary  local table  will  be
        created.  The  parameter  'schema'  must   be  empty,
        otherwise this parameter is ignored.
    gen_tmp_table_name: bool, optional
        Sets  the  name of the temporary table. This  parameter
        is only used when the parameter 'temporary_local_table'
        is  set to True and if the parameters  "table_name" and
        "schema" are unspecified.
    ingest_local: bool, optional
        If  set  to  True, the file will be ingested  from  the
        local machine.
    genSQL: bool, optional
        If  set to True,  the SQL  code for creating the  final
        table is generated but not executed. This is a good way
        to change the  final relation types or to customize the
        data ingestion.
    materialize: bool, optional
        If  set to True, the flex table is materialized into a
        table.
        Otherwise,  it  will  remain a flex table. Flex tables
        simplify the data ingestion but have worse  performace
        compared to regular tables.
    use_complex_dt: bool, optional
        Boolean,  whether  the  input data  file  has  complex
        structure. If set to true, most of the other parameters
        are ignored.

    Returns
    -------
    vDataFrame
        The vDataFrame of the relation.
    """
    new_name = format_type(new_name, dtype=dict)
    usecols = format_type(usecols, dtype=list)
    if use_complex_dt:
        if new_name:
            raise ValueError(
                "You cannot use the parameter 'new_name' with 'use_complex_dt'."
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
        schema = conf.get_option("temp_schema")
    assert not temporary_table or not temporary_local_table, ValueError(
        "Parameters 'temporary_table' and 'temporary_local_table' can not be both set to True."
    )
    file_extension = path.split(".")[-1].lower()
    compression = extract_compression(path)
    if (
        (file_extension not in ("json",) and not is_avro)
        or (file_extension not in ("avro",) and (is_avro))
    ) and (compression == "UNCOMPRESSED"):
        raise ExtensionError("The file extension is incorrect !")
    basename = ".".join(path.split("/")[-1].split(".")[0:-1])
    if gen_tmp_table_name and temporary_local_table and not table_name:
        table_name = gen_tmp_name(name=basename)
    if not table_name:
        table_name = basename
    if is_avro:
        label = "read_avro"
        parser = "FAVROPARSER"
    else:
        label = "read_json"
        parser = "FJSONPARSER"
    if not genSQL:
        table_name_str = table_name.replace("'", "''")
        schema_str = schema.replace("'", "''")
        column_name = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('{label}')*/ 
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
    if not genSQL and (column_name != []) and not insert:
        raise NameError(f"The table {input_relation} already exists !")
    elif not genSQL and (column_name == []) and (insert):
        raise MissingRelation(f"The table {input_relation} doesn't exist !")
    else:
        if temporary_local_table:
            input_relation = quote_ident(table_name)
        all_queries = []
        if not materialize:
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
        if not insert:
            all_queries += [clean_query(query)]
        options = []
        if start_point and not is_avro:
            options += [f"start_point='{start_point}'"]
        if record_terminator and not is_avro:
            prefix = ""
            if "\\" in record_terminator.__repr__():
                prefix = "E"
            options += [f"record_terminator={prefix}'{record_terminator}'"]
        if suppress_nonalphanumeric_key_chars and not is_avro:
            options += ["suppress_nonalphanumeric_key_chars=true"]
        elif not is_avro:
            options += ["suppress_nonalphanumeric_key_chars=false"]
        if reject_on_materialized_type_error:
            assert materialize, ValueError(
                "When using complex data types the table has to "
                "be materialized. Set materialize to True"
            )
            options += ["reject_on_materialized_type_error=true"]
        else:
            options += ["reject_on_materialized_type_error=false"]
        if reject_on_duplicate and not is_avro:
            options += ["reject_on_duplicate=true"]
        elif not is_avro:
            options += ["reject_on_duplicate=false"]
        if reject_on_empty_key and not is_avro:
            options += ["reject_on_empty_key=true"]
        elif not is_avro:
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
        if genSQL and insert and not materialize:
            return [clean_query(query2)]
        elif genSQL and not materialize:
            return all_queries
        if not insert:
            _executeSQL(
                query,
                title="Creating flex table.",
            )
        _executeSQL(
            query2,
            title="Ingesting the data in the flex table.",
        )
        if not materialize:
            return vDataFrame(table_name, schema=schema)
        result = compute_flextable_keys(flex_name)
        dtype = {}
        for column_dtype in result:
            try:
                _executeSQL(
                    query=f"""
                        SELECT 
                            /*+LABEL('{label}')*/ 
                            \"{column_dtype[0]}\"::{column_dtype[1]} 
                        FROM {flex_name} 
                        LIMIT 1000""",
                    print_time_sql=False,
                )
                dtype[column_dtype[0]] = column_dtype[1]
            except QueryError:
                dtype[column_dtype[0]] = "Varchar(100)"
        if not insert:
            cols = (
                [column for column in dtype]
                if not usecols
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
                        /*+LABEL('{label}')*/ 
                        {", ".join(cols)} 
                    FROM {flex_name}"""
            all_queries = all_queries + [clean_query(query3)]
            if genSQL:
                return all_queries
            _executeSQL(
                query3,
                title="Creating table.",
            )
            if not temporary_local_table and conf.get_option("print_info"):
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
                if isinstance(final_cols[column], NoneType):
                    final_transformation += [f'NULL AS "{column}"']
                else:
                    final_transformation += [
                        f'"{final_cols}"::{column_name_dtype[column]} AS "{column}"'
                    ]
            query = f"""
                INSERT 
                    /*+LABEL('{label}')*/ 
                INTO {input_relation} 
                SELECT 
                    {", ".join(final_transformation)} 
                FROM {flex_name}"""
            if genSQL:
                return [clean_query(query)]
            _executeSQL(
                query,
                title="Inserting data into table.",
            )
        drop(name=flex_name, method="table")
        return vDataFrame(table_name, schema=schema)
