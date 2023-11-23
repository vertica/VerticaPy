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
import csv
import os
from typing import Optional

import pandas as pd

import verticapy._config.config as conf
from verticapy._typing import NoneType
from verticapy._utils._gen import gen_tmp_name
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import format_schema_table, format_type, quote_ident
from verticapy._utils._sql._sys import _executeSQL


from verticapy.core.parsers.csv import read_csv
from verticapy.core.vdataframe.base import vDataFrame


@save_verticapy_logs
def read_pandas(
    df: pd.DataFrame,
    name: Optional[str] = None,
    schema: Optional[str] = None,
    dtype: Optional[dict] = None,
    parse_nrows: int = 10000,
    temp_path: Optional[str] = None,
    insert: bool = False,
) -> vDataFrame:
    """
    Ingests a pandas DataFrame into the Vertica
    database  by  creating a CSV file and  then
    using flex tables to load the data.

    Parameters
    ----------
    df: pandas.DataFrame
        The pandas.DataFrame to ingest.
    name: str, optional
        Name  of  the new  relation or the  relation
        in which to insert the data. If unspecified,
        a  temporary  local  table is created.  This
        temporary table is dropped at the end of the
        local session.
    schema: str, optional
        Schema  of  the new  relation.  If  empty, a
        temporary  schema  is  used.  To modify  the
        temporary   schema,  use  the   'set_option'
        function.
    dtype: dict, optional
        Dictionary   of  input  types.  Providing  a
        dictionary can increase  ingestion speed and
        precision. If specified, rather than parsing
        the intermediate CSV and  guessing the input
        types,  VerticaPy uses  the specified  input
        types instead.
    parse_nrows: int, optional
        If    this parameter  is  greater  than  zero,
        VerticaPy  creates  and  ingests  a  temporary
        file containing  'parse_nrows'  number of rows
        to  determine  the  input  data types   before
        ingesting the intermediate CSV file containing
        the rest of the data. This method of data type
        identification  is less accurate, but is  much
        faster for large datasets.
    temp_path: str, optional
        The path to which to write the intermediate CSV
        file.  This is  useful in cases where the  user
        does not have write  permissions on the current
        directory.
    insert: bool, optional
        If set to True, the data are ingested into the
        input relation. The column names of your table
        and the pandas.DataFrame must match.

    Returns
    -------
    vDataFrame
        vDataFrame of the new relation.
    """
    dtype = format_type(dtype, dtype=dict)
    if not schema:
        schema = conf.get_option("temp_schema")
    if insert and not name:
        raise ValueError(
            "Parameter 'name' can not be empty when "
            "parameter 'insert' is set to True."
        )
    if not name:
        tmp_name = gen_tmp_name(name="df")[1:-1]
    else:
        tmp_name = ""
    sep = "/" if ((temp_path) and temp_path[-1] != "/") else ""
    path = f"{temp_path}{sep}{name}.csv"
    clear = False
    try:
        # Adding the quotes to STR pandas columns in order
        # to simplify the ingestion.
        # Not putting them can lead to wrong data ingestion.
        str_cols, null_columns = [], []
        for c in df.columns:
            if isinstance(df[c].first_valid_index(), NoneType):
                null_columns += [c]
            elif df[c].dtype == object and isinstance(
                df[c].loc[df[c].first_valid_index()], str
            ):
                str_cols += [c]
        if len(df.columns) == len(null_columns):
            names = ", ".join([f"NULL AS {quote_ident(col)}" for col in df.columns])
            q = " UNION ALL ".join([f"(SELECT {names})" for i in range(len(df))])
            return vDataFrame(q)
        if len(str_cols) > 0 or len(null_columns) > 0:
            tmp_df = df.copy()
            for c in str_cols:
                tmp_df[c] = '"' + tmp_df[c].str.replace('"', '""') + '"'
            for c in null_columns:
                tmp_df[c] = '""'
            clear = True
        else:
            tmp_df = df
        tmp_df.to_csv(
            path,
            index=False,
            quoting=csv.QUOTE_NONE,
            quotechar="",
            escapechar="\027",
        )

        if len(str_cols) > 0 or len(null_columns) > 0:
            # to_csv is adding an undesired special character
            # we remove it
            with open(path, "r", encoding="utf-8") as f:
                filedata = f.read()
            filedata = filedata.replace(",", ",").replace('""', "")
            with open(path, "w", encoding="utf-8") as f:
                f.write(filedata)
        if insert:
            input_relation = format_schema_table(schema, name)
            tmp_df_columns_str = ", ".join(
                ['"' + col.replace('"', '""') + '"' for col in tmp_df.columns]
            )
            _executeSQL(
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
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        if clear:
            del tmp_df
    return vdf
