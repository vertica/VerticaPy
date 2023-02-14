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
import warnings
from typing import Union

# VerticaPy Modules
import verticapy.stats as st
from verticapy.datasets import gen_meshgrid
from verticapy.vdataframe import vDataFrame
from verticapy._utils._collect import save_verticapy_logs
from verticapy.core.tablesample import tablesample
from verticapy.sql.read import to_tablesample, vDataFrameSQL
from verticapy._utils._sql import _executeSQL


@save_verticapy_logs
def create_index(
    vdf: vDataFrame,
    gid: str,
    g: str,
    index: str,
    overwrite: bool = False,
    max_mem_mb: int = 256,
    skip_nonindexable_polygons: bool = False,
) -> tablesample:
    """
Creates a spatial index on a set of polygons to speed up spatial 
intersection with a set of points.

Parameters
----------
vdf: vDataFrame
    vDataFrame to use to compute the spatial join.
gid: str
    Name of an integer column that uniquely identifies the polygon. 
    The gid cannot be NULL.
g: str
    Name of a geometry or geography (WGS84) column or expression that 
    contains polygons and multipolygons. Only polygon and multipolygon 
    can be indexed. Other shape types are excluded from the index.
index: str
    Name of the index.
overwrite: bool, optional
    BOOLEAN value that specifies whether to overwrite the index, if an 
    index exists.
max_mem_mb: int, optional
    A positive integer that assigns a limit to the amount of memory in 
    megabytes that create_index can allocate during index construction.
skip_nonindexable_polygons: bool, optional
    In rare cases, intricate polygons (for instance, with too high 
    resolution or anomalous spikes) cannot be indexed. These polygons 
    are considered non-indexable. 
    When set to False, non-indexable polygons cause the index creation 
    to fail. When set to True, index creation can succeed by excluding 
    non-indexable polygons from the index.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
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
        FROM {vdf.__genSQL__()}"""
    return to_tablesample(query)


@save_verticapy_logs
def coordinate_converter(
    vdf: vDataFrame,
    x: str,
    y: str,
    x0: float = 0.0,
    earth_radius: Union[int, float] = 6371,
    reverse: bool = False,
) -> vDataFrame:
    """
Converts between geographic coordinates (latitude and longitude) and 
Euclidean coordinates (x,y).

Parameters
----------
vdf: vDataFrame
    input vDataFrame.
x: str
    vDataColumn used as the abscissa (longitude).
y: str
    vDataColumn used as the ordinate (latitude).
x0: float, optional
    The initial abscissa.
earth_radius: int / float, optional
    Earth radius in km.
reverse: bool, optional
    If set to True, the Euclidean coordinates are converted to latitude 
    and longitude.

Returns
-------
vDataFrame
    result of the transformation.
    """
    x, y = vdf.format_colnames(x, y)

    result = vdf.copy()

    if reverse:

        result[x] = result[x] / earth_radius * 180 / st.PI + x0
        result[y] = (
            (st.atan(st.exp(result[y] / earth_radius)) - st.PI / 4) / st.PI * 360
        )

    else:

        result[x] = earth_radius * ((result[x] - x0) * st.PI / 180)
        result[y] = earth_radius * st.ln(st.tan(result[y] * st.PI / 360 + st.PI / 4))

    return result


@save_verticapy_logs
def describe_index(name: str = "", list_polygons: bool = False) -> tablesample:
    """
Retrieves information about an index that contains a set of polygons. If 
you do not pass any parameters, this function returns all defined indexes.

Parameters
----------
name: str, optional
    Index name.
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

    if not (name):
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
        result = vDataFrameSQL(f"({query}) x")
    else:
        result = to_tablesample(query)

    return result


@save_verticapy_logs
def intersect(
    vdf: vDataFrame, index: str, gid: str, g: str = "", x: str = "", y: str = ""
) -> vDataFrame:
    """
Spatially intersects a point or points with a set of polygons.

Parameters
----------
vdf: vDataFrame
    vDataFrame to use to compute the spatial join.
index: str
    Name of the index.
gid: str
    An integer column or integer that uniquely identifies the spatial 
    object(s) of g or x and y.
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
    x, y, gid, g = vdf.format_colnames(x, y, gid, g)

    if g:
        params = f"{gid}, {g}"

    elif x and y:
        params = f"{gid}, {x}, {y}"

    else:

        raise ParameterError("Either 'x' and 'y' or 'g' must not be empty.")

    query = f"""
        (SELECT 
            STV_Intersect({params} 
            USING PARAMETERS 
                index='{index}') 
            OVER (PARTITION BEST) AS (point_id, polygon_gid) 
        FROM {vdf.__genSQL__()}) x"""

    return vDataFrameSQL(query)


@save_verticapy_logs
def rename_index(source: str, dest: str, overwrite: bool = False) -> bool:
    """
Renames a spatial index.

Parameters
----------
source: str
    Current name of the spatial index.
dest: str
    New name of the spatial index.
overwrite: bool, optional
    BOOLEAN value that specifies whether to overwrite the index, if an 
    index exists.

Returns
-------
bool
    True if the index was renamed, False otherwise.
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


@save_verticapy_logs
def split_polygon_n(p: str, nbins: int = 100) -> vDataFrame:
    """
Splits a polygon into (nbins ** 2) smaller polygons of approximately equal
total area. This process is inexact, and the split polygons have 
approximated edges; greater values for nbins produces more accurate and 
precise edge approximations.

Parameters
----------
p: str
    String representation of the polygon.
nbins: int, optional
    Number of bins used to cut the longitude and the latitude.
    Split polygons have approximated edges, and greater values for nbins
    leads to more accurate and precise edge approximations.

Returns
-------
vDataFrame
    output vDataFrame that includes the new polygons.
    """
    sql = f"""SELECT /*+LABEL(split_polygon_n)*/
                MIN(ST_X(point)), 
                MAX(ST_X(point)), 
                MIN(ST_Y(point)), 
                MAX(ST_Y(point)) 
             FROM (SELECT 
                        STV_PolygonPoint(geom) OVER() 
                   FROM (SELECT ST_GeomFromText('{p}') 
                                AS geom) x) y"""
    min_x, max_x, min_y, max_y = _executeSQL(
        sql, title="Computing min & max: x & y.", method="fetchrow"
    )

    delta_x, delta_y = (max_x - min_x) / nbins, (max_y - min_y) / nbins
    vdf = gen_meshgrid(
        {
            "x": {"type": float, "range": [min_x, max_x], "nbins": nbins},
            "y": {"type": float, "range": [min_y, max_y], "nbins": nbins},
        }
    )
    vdf["gid"] = "ROW_NUMBER() OVER (ORDER BY x, y)"
    vdf[
        "geom"
    ] = f"""
        ST_GeomFromText(
            'POLYGON ((' || x || ' ' || y 
                || ', ' || x + {delta_x} || ' ' 
                || y || ', ' || x + {delta_x} 
                || ' ' || y + {delta_y} || ', ' 
                || x || ' ' || y + {delta_y} 
                || ', ' || x || ' ' || y || '))')"""
    vdf["gid"].apply("ROW_NUMBER() OVER (ORDER BY {})")
    vdf.filter(f"ST_Intersects(geom, ST_GeomFromText('{p}'))", print_info=False)

    return vdf[["gid", "geom"]]
