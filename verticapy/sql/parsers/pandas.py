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
import os, csv
import pandas as pd

from verticapy._config.config import _options
from verticapy._utils._collect import save_verticapy_logs
from verticapy._utils._gen import gen_tmp_name
from verticapy._utils._sql._execute import _executeSQL
from verticapy._utils._sql._format import format_schema_table
from verticapy.errors import ParameterError

from verticapy.sql.parsers.csv import read_csv


@save_verticapy_logs
def read_pandas(
    df: pd.DataFrame,
    name: str = "",
    schema: str = "",
    dtype: dict = {},
    parse_nrows: int = 10000,
    temp_path: str = "",
    insert: bool = False,
):
    """
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
        schema = _options["temp_schema"]
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
            from verticapy.core.vdataframe.base import vDataFrame

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
