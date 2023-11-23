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

from verticapy._utils._sql._collect import save_verticapy_logs

from verticapy.core.parsers.json import read_json
from verticapy.core.vdataframe.base import vDataFrame


@save_verticapy_logs
def read_avro(
    path: str,
    schema: Optional[str] = None,
    table_name: Optional[str] = None,
    usecols: Optional[list] = None,
    new_name: Optional[dict] = None,
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
) -> vDataFrame:
    """
    Ingests an AVRO file using flex tables.

    Parameters
    ----------
    path: str
        Absolute path where the AVRO file is located.
    schema: str, optional
        Schema  where the AVRO file will be ingested.
    table_name: str, optional
        Final relation name.
    usecols: list, optional
        List  of the  AVRO parameters to ingest.  The
        other ones will be ignored.  If empty, all the
        AVRO parameters will be ingested.
    new_name: dict, optional
        Dictionary  of the new  columns name. If  the
        AVRO file is nested,  it is advised to change
        the final names as special characters will be
        included.
        For example, {"param": {"age": 3,
                                "name": Badr},
                      "date": 1993-03-11}
        will create 3 columns: "param.age", "param.name"
        and "date".  You can  rename these columns using
        the  'new_name'  parameter  with  the  following
        dictionary: {"param.age": "age",
                     "param.name": "name"}
    insert: bool, optional
        If set to True, the  data will be ingested to the
        input relation.  The AVRO  parameters must be the
        same  as the input  relation otherwise they  will
        not be ingested. If set to True, table_name cannot
        be empty.
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
    flatten_maps: bool, optional
        Boolean, whether to flatten sub-maps within the AVRO
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
        table. Otherwise,  it  will  remain a flex table. Flex
        tables simplify the data ingestion but have worse
        performace compared to regular tables.
    use_complex_dt: bool, optional
        Boolean,  whether  the  input data  file  has  complex
        structure.  When  this  is  true,  most of  the  other
        parameters will be ignored.

    Returns
    -------
    vDataFrame
        The vDataFrame of the relation.
    """
    return read_json(
        **locals(),
        is_avro=True,
    )
