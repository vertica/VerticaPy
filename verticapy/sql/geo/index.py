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
import warnings
from typing import Optional, Union

from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._sys import _executeSQL
from verticapy._typing import SQLRelation

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame


@save_verticapy_logs
def create_index(
    vdf: SQLRelation,
    gid: str,
    g: str,
    index: str,
    overwrite: bool = False,
    max_mem_mb: int = 256,
    skip_nonindexable_polygons: bool = False,
) -> TableSample:
    """
    Creates  a spatial  index on a set of polygons  to
    speed up spatial intersection with a set of points.

    Parameters
    ----------
    vdf: SQLRelation
        vDataFrame used to compute the spatial join.
    gid: str
        Name  of  an   integer  column  that  uniquely
        identifies the polygon.
        The gid cannot be NULL.
    g: str
        Name of a geometry or geography (WGS84) column
        or  expression  that   contains  polygons  and
        multipolygons.  Only polygon  and multipolygon
        can be indexed. Other shape types are excluded
        from the index.
    index: str
        Name of the index.
    overwrite: bool, optional
        BOOLEAN  value   that  specifies   whether  to
        overwrite the index, if an index exists.
    max_mem_mb: int, optional
        A positive integer that  assigns a limit to the
        amount of memory in megabytes that create_index
        can allocate during index construction.
    skip_nonindexable_polygons: bool, optional
        In rare cases, intricate polygons (for instance,
        those with too high resolution or anomalous
        spikes) cannot be indexed. These polygons are
        considered non-indexable.
        When set to False,  non-indexable polygons cause
        the  index creation  to fail. When set to  True,
        index   creation  can   succeed   by   excluding
        non-indexable polygons from the index.

    Returns
    -------
    TableSample
        geospatial indexes.
    """
    gid, g = vdf.format_colnames(gid, g)
    query = f"""
        SELECT 
            STV_Create_Index({gid}, {g} 
            USING PARAMETERS 
                index='{index}', 
                overwrite={overwrite} , 
                max_mem_mb={max_mem_mb}, 
                skip_nonindexable_polygons={skip_nonindexable_polygons}) 
            OVER() 
        FROM {vdf}"""
    return TableSample.read_sql(query)


@save_verticapy_logs
def describe_index(
    name: Optional[str] = None, list_polygons: bool = False
) -> Union[TableSample, vDataFrame]:
    """
    Retrieves  information about an index that
    contains a set of polygons.  If you do not
    pass any parameters, this function returns
    all defined indexes.

    Parameters
    ----------
    name: str, optional
        Index name.
    list_polygons: bool, optional
        Boolean that specifies whether to list
        the polygons  in the index.  If set to
        True,   the  function  returns  a
        vDataFrame instead of a TableSample.

    Returns
    -------
    TableSample
        geospatial indexes.
    """
    if not name:
        query = f"SELECT STV_Describe_Index () OVER ()"
    else:
        query = f"""
            SELECT 
                STV_Describe_Index (
                USING PARAMETERS 
                    index='{name}', 
                    list_polygons={list_polygons}) 
            OVER()"""

    if list_polygons:
        res = vDataFrame(query)
    else:
        res = TableSample.read_sql(query)

    return res


@save_verticapy_logs
def rename_index(source: str, dest: str, overwrite: bool = False) -> bool:
    """
    Renames a spatial index.

    Parameters
    ----------
    source: str
        Current  name of the spatial  index.
    dest: str
        New name of the spatial index.
    overwrite: bool, optional
        BOOLEAN value that specifies whether
        to overwrite the  index, if an index
        exists.

    Returns
    -------
    bool
        True if the index was renamed, False
        otherwise.
    """

    try:
        _executeSQL(
            query=f"""
                SELECT /*+LABEL(rename_index)*/ 
                    STV_Rename_Index(
                    USING PARAMETERS 
                        source = '{source}', 
                        dest = '{dest}', 
                        overwrite = {overwrite}) 
                    OVER ();""",
            title="Renaming Index.",
        )

    except Exception as e:
        warnings.warn(str(e), Warning)
        return False

    return True
