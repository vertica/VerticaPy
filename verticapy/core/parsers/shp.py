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

import verticapy._config.config as conf
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._sys import _executeSQL
from verticapy.errors import ExtensionError

from verticapy.core.vdataframe.base import vDataFrame


@save_verticapy_logs
def read_shp(
    path: str,
    schema: Optional[str] = None,
    table_name: Optional[str] = None,
) -> vDataFrame:
    """
    Ingests a SHP file.  At the moment, only files
    located in  the Vertica server can be ingested.

    Parameters
    ----------
    path: str
        Absolute path where the SHP file is located.
    schema: str, optional
        Schema where the SHP  file will be ingested.
    table_name: str, optional
        Final relation name.

    Returns
    -------
    vDataFrame
        The vDataFrame of the relation.
    """
    if not (schema):
        schema = conf.get_option("temp_schema")
    file = path.split("/")[-1]
    file_extension = file[-3 : len(file)]
    if file_extension != "shp":
        raise ExtensionError("The file extension is incorrect !")
    result = _executeSQL(
        query=f"""
            SELECT 
                /*+LABEL('read_shp')*/ 
                STV_ShpCreateTable(USING PARAMETERS file='{path}')
                OVER() AS create_shp_table;""",
        title="Getting SHP definition.",
        method="fetchall",
    )
    if not table_name:
        table_name = file[:-4]
    result[0] = [f'CREATE TABLE "{schema}"."{table_name}"(']
    result = [elem[0] for elem in result]
    result = "".join(result)
    _executeSQL(result, title="Creating the relation.")
    _executeSQL(
        query=f"""
            COPY "{schema}"."{table_name}" 
            WITH SOURCE STV_ShpSource(file=\'{path}\')
            PARSER STV_ShpParser();""",
        title="Ingesting the data.",
    )
    if conf.get_option("print_info"):
        print(f'The table "{schema}"."{table_name}" has been successfully created.')
    return vDataFrame(table_name, schema=schema)
