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
from typing import Union

# VerticaPy Modules
import verticapy.sql.functions.math as mt
from verticapy.datasets.generators import gen_meshgrid
from verticapy.vdataframe import vDataFrame
from verticapy._utils._collect import save_verticapy_logs
from verticapy._utils._sql._execute import _executeSQL


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
    x, y = vdf._format_colnames(x, y)

    result = vdf.copy()

    if reverse:

        result[x] = result[x] / earth_radius * 180 / mt.PI + x0
        result[y] = (
            (mt.atan(mt.exp(result[y] / earth_radius)) - mt.PI / 4) / mt.PI * 360
        )

    else:

        result[x] = earth_radius * ((result[x] - x0) * mt.PI / 180)
        result[y] = earth_radius * mt.ln(mt.tan(result[y] * mt.PI / 360 + mt.PI / 4))

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
    x, y, gid, g = vdf._format_colnames(x, y, gid, g)

    if g:
        params = f"{gid}, {g}"

    elif x and y:
        params = f"{gid}, {x}, {y}"

    else:

        raise ParameterError("Either 'x' and 'y' or 'g' must not be empty.")

    query = f"""
        SELECT 
            STV_Intersect({params} 
            USING PARAMETERS 
                index='{index}') 
            OVER (PARTITION BEST) AS (point_id, polygon_gid) 
        FROM {vdf._genSQL()}"""

    return vDataFrame(query)


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
