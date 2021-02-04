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
import warnings

# VerticaPy Modules
import verticapy, vertica_python
from verticapy.toolbox import *
from verticapy.utilities import *
from verticapy.stats import *
from verticapy import vDataFrame

# ---#
def create_index(
    vdf: vDataFrame,
    gid: str,
    g: str,
    index: str,
    overwrite: bool = False,
    max_mem_mb: int = 256,
    skip_nonindexable_polygons: bool = False,
):
    """
---------------------------------------------------------------------------
Creates a spatial index on a set of polygons to speed up spatial intersection 
with a set of points.

Parameters
----------
vdf: vDataFrame
    vDataFrame to use to compute the spatial join.
gid: str
    Name of an integer column that uniquely identifies the polygon. The gid 
    cannot be NULL.
g: str
    Name of a geometry or geography (WGS84) column or expression that contains 
    polygons and multipolygons. Only polygon and multipolygon can be indexed. 
    Other shape types are excluded from the index.
index: str
    Name of the index.
overwrite: bool, optional
    BOOLEAN value that specifies whether to overwrite the index, if an index exists.
max_mem_mb: int, optional
    A positive integer that assigns a limit to the amount of memory in megabytes 
    that create_index can allocate during index construction.
skip_nonindexable_polygons: bool, optional
    In rare cases, intricate polygons (for instance, with too high resolution or 
    anomalous spikes) cannot be indexed. These polygons are considered non-indexable. 
    When set to False, non-indexable polygons cause the index creation to fail. 
    When set to True, index creation can succeed by excluding non-indexable polygons 
    from the index.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    check_types(
        [
            ("vdf", vdf, [vDataFrame],),
            ("gid", gid, [str],),
            ("index", index, [str],),
            ("g", g, [str],),
            ("overwrite", overwrite, [bool],),
            ("max_mem_mb", max_mem_mb, [int],),
            ("skip_nonindexable_polygons", skip_nonindexable_polygons, [bool],),
        ]
    )
    columns_check([gid, g], vdf)
    gid, g = vdf_columns_names([gid, g], vdf)
    query = "SELECT STV_Create_Index({}, {} USING PARAMETERS index='{}', overwrite={} , max_mem_mb={}, skip_nonindexable_polygons={}) OVER() FROM {}"
    query = query.format(
        gid,
        g,
        index,
        overwrite,
        max_mem_mb,
        skip_nonindexable_polygons,
        vdf.__genSQL__(),
    )
    return to_tablesample(query, vdf._VERTICAPY_VARIABLES_["cursor"])


# ---#
def describe_index(
    name: str = "", cursor=None, list_polygons: bool = False,
):
    """
---------------------------------------------------------------------------
Retrieves information about an index that contains a set of polygons. 
If you do not pass any parameters, the function returns all of the 
defined indexes.

Parameters
----------
name: str, optional
    Index name.
cursor: DBcursor, optional
    Vertica DB cursor.
list_polygons: bool, optional
    Boolean that specifies whether to list the polygons in the index.
    If set to True, the function will return a vDataFrame instead of
    a tablesample.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    check_types(
        [("name", name, [str],), ("list_polygons", list_polygons, [bool],),]
    )
    cursor, conn = check_cursor(cursor)[0:2]
    if not (name):
        query = f"SELECT STV_Describe_Index () OVER ()"
    else:
        query = f"SELECT STV_Describe_Index (USING PARAMETERS index='{name}', list_polygons={list_polygons}) OVER ()"
    if list_polygons:
        result = vdf_from_relation(f"({query}) x", cursor=cursor)
    else:
        result = to_tablesample(query, cursor,)
        if conn:
            conn.close()
    return result


# ---#
def intersect(
    vdf: vDataFrame, index: str, gid: str, g: str = "", x: str = "", y: str = "",
):
    """
---------------------------------------------------------------------------
Spatially intersects a point or points with a set of polygons.

Parameters
----------
vdf: vDataFrame
    vDataFrame to use to compute the spatial join.
index: str
    Name of the index.
gid: str
    An integer column or integer that uniquely identifies the spatial object(s) 
    of g or x and y.
g: str, optional
    A geometry or geography (WGS84) column that contains points. 
    The g column can contain only point geometries or geographies.
x: str, optional
    x-coordinate or longitude.
y: str, optional
    y-coordinate or latitude.

Returns
-------
vDataFrame
    object containing the result of the intersection.
    """
    check_types(
        [
            ("vdf", vdf, [vDataFrame],),
            ("gid", gid, [str],),
            ("g", g, [str],),
            ("x", x, [str],),
            ("y", y, [str],),
            ("index", index, [str],),
        ]
    )
    table = vdf.__genSQL__()
    columns_check([gid], vdf)
    if g:
        columns_check([g], vdf)
        g = vdf_columns_names([g], vdf)[0]
        query = f"(SELECT STV_Intersect({gid}, {g} USING PARAMETERS index='{index}') OVER (PARTITION BEST) AS (point_id, polygon_gid) FROM {table}) x"
    elif x and y:
        columns_check([x, y], vdf)
        x, y = vdf_columns_names([x, y], vdf)
        query = f"(SELECT STV_Intersect({gid}, {x}, {y} USING PARAMETERS index='{index}') OVER (PARTITION BEST) AS (point_id, polygon_gid) FROM {table}) x"
    else:
        raise ParameterError("Either 'x' and 'y' or 'g' must not be empty.")
    return vdf_from_relation(query, cursor=vdf._VERTICAPY_VARIABLES_["cursor"])


# ---#
def rename_index(
    source: str, dest: str, cursor=None, overwrite: bool = False,
):
    """
---------------------------------------------------------------------------
Renames a spatial index.

Parameters
----------
source: str
    Current name of the spatial index.
dest: str
    New name of the spatial index.
cursor: DBcursor, optional
    Vertica DB cursor.
overwrite: bool, optional
    BOOLEAN value that specifies whether to overwrite the index, if an index 
    exists.

Returns
-------
bool
    True if the index was renamed, False otherwise.
    """
    check_types(
        [
            ("source", source, [str],),
            ("dest", dest, [str],),
            ("overwrite", overwrite, [bool],),
        ]
    )
    cursor, conn = check_cursor(cursor)[0:2]
    query = f"SELECT STV_Rename_Index (USING PARAMETERS source = '{source}', dest = '{dest}', overwrite = {overwrite}) OVER ();"
    try:
        cursor.execute(query)
    except Exception as e:
        warnings.warn(str(e), Warning)
        return False
    if conn:
        conn.close()
    return True
