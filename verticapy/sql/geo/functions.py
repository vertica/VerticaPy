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

from verticapy._typing import PythonNumber
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._sys import _executeSQL
from verticapy._typing import SQLRelation

from verticapy.datasets.generators import gen_meshgrid

from verticapy.core.vdataframe.base import vDataFrame

import verticapy.sql.functions.math as mt


@save_verticapy_logs
def coordinate_converter(
    vdf: SQLRelation,
    x: str,
    y: str,
    x0: float = 0.0,
    earth_radius: PythonNumber = 6371,
    reverse: bool = False,
) -> vDataFrame:
    """
    Converts between geographic coordinates (latitude
    and longitude)  and  Euclidean coordinates (x,y).

    Parameters
    ----------
    vdf: SQLRelation
        Input vDataFrame.
    x: str
        vDataColumn used as the abscissa (longitude).
    y: str
        vDataColumn used as the ordinate  (latitude).
    x0: float, optional
        The initial abscissa.
    earth_radius: PythonNumber, optional
        Earth radius in km.
    reverse: bool, optional
        If set to True, the Euclidean coordinates are
        converted to latitude and longitude.

    Returns
    -------
    vDataFrame
        result of the transformation.

    Examples
    --------
    .. code-block:: python

        from verticapy.sql.geo import coordinate_converter
        from verticapy.datasets import load_cities

        cities = load_cities()
        cities["lat"] = "ST_X(geometry)"
        cities["lon"] = "ST_Y(geometry)"
        cities

    .. ipython:: python
        :suppress:

        from verticapy.sql.geo import coordinate_converter
        from verticapy.datasets import load_cities
        from verticapy import set_option
        cities = load_cities()
        cities["lat"] = "ST_X(geometry)"
        cities["lon"] = "ST_Y(geometry)"
        #limit display rows because hit unicode decoding error
        set_option("max_rows", 20)
        html_file = open("SPHINX_DIRECTORY/figures/sql_geo_functions_coordinate_converter_1.html", "w")
        html_file.write(cities._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_geo_functions_coordinate_converter_1.html

    .. code-block:: python

        # Converting lon, lat to x, y
        convert_xy = coordinate_converter(cities, "lon", "lat")
        convert_xy

    .. ipython:: python
        :suppress:

        convert_xy = coordinate_converter(cities, "lon", "lat")
        html_file = open("SPHINX_DIRECTORY/figures/sql_geo_functions_coordinate_converter_2.html", "w")
        html_file.write(convert_xy._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_geo_functions_coordinate_converter_2.html

    .. code-block:: python

        # Converting x, y to lon, lat
        convert_reverse_xy = coordinate_converter(convert_xy, "lon", "lat", reverse=True)
        convert_reverse_xy

    .. ipython:: python
        :suppress:

        html_file = open("SPHINX_DIRECTORY/figures/sql_geo_functions_coordinate_converter_3.html", "w")
        html_file.write(coordinate_converter(convert_xy, "lon", "lat", reverse=True)._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_geo_functions_coordinate_converter_3.html
    """
    x, y = vdf.format_colnames(x, y)

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
    vdf: SQLRelation,
    index: str,
    gid: str,
    g: Optional[str] = None,
    x: Optional[str] = None,
    y: Optional[str] = None,
) -> vDataFrame:
    """
    Spatially intersects a point or points with a set
    of polygons.

    Parameters
    ----------
    vdf: SQLRelation
        vDataFrame used to compute the spatial join.
    index: str
        Name of the index.
    gid: str
        An  integer  column  or integer that  uniquely
        identifies the spatial object(s) of g or x and
        y.
    g: str, optional
        A  geometry  or  geography (WGS84) column that
        contains points. The g column can contain only
        point geometries or geographies.
    x: str, optional
        x-coordinate or longitude.
    y: str, optional
        y-coordinate or latitude.

    Returns
    -------
    vDataFrame
        object containing the result of the intersection.

    Examples
    --------
    .. code-block:: python

        from verticapy.sql.geo import intersect, create_index
        from verticapy.datasets import load_world, load_cities

        world = load_world()
        world["id"] = "ROW_NUMBER() OVER(ORDER BY country, pop_est)"
        display(world)

        cities = load_cities()
        cities["id"] = "ROW_NUMBER() OVER (ORDER BY city)"
        cities["lat"] = "ST_X(geometry)"
        cities["lon"] = "ST_Y(geometry)"
        display(cities)

    .. ipython:: python
        :suppress:

        from verticapy.sql.geo import intersect, create_index
        from verticapy.datasets import load_world, load_cities
        from verticapy import set_option
        world = load_world()
        world["id"] = "ROW_NUMBER() OVER(ORDER BY country, pop_est)"
        cities = load_cities()
        cities["id"] = "ROW_NUMBER() OVER (ORDER BY city)"
        cities["lat"] = "ST_X(geometry)"
        cities["lon"] = "ST_Y(geometry)"
        html_file = open("SPHINX_DIRECTORY/figures/sql_geo_functions_intersect_1.html", "w")
        html_file.write(world._repr_html_())
        html_file.close()
        html_file = open("SPHINX_DIRECTORY/figures/sql_geo_functions_intersect_2.html", "w")
        html_file.write(cities._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_geo_functions_intersect_1.html

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_geo_functions_intersect_2.html

    .. code-block:: python

        # Creating Index
        create_index(world, "id", "geometry", "world_polygons", True)

    .. ipython:: python
        :suppress:

        html_file = open("SPHINX_DIRECTORY/figures/sql_geo_functions_intersect_4.html", "w")
        html_file.write(create_index(world, "id", "geometry", "world_polygons", True)._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_geo_functions_intersect_4.html

    .. code-block:: python

        # Intersect using Geometry
        intersect(cities, "world_polygons", "id", "geometry")

    .. ipython:: python
        :suppress:

        html_file = open("SPHINX_DIRECTORY/figures/sql_geo_functions_intersect_3.html", "w")
        html_file.write(intersect(cities, "world_polygons", "id", "geometry")._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_geo_functions_intersect_3.html

    .. code-block:: python

        # Intersect using Latitude and Longitude
        intersect(cities, "world_polygons", "id", x="lat", y="lon")

    .. ipython:: python
        :suppress:

        html_file = open("SPHINX_DIRECTORY/figures/sql_geo_functions_intersect_4.html", "w")
        html_file.write(intersect(cities, "world_polygons", "id", x="lat", y="lon")._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_geo_functions_intersect_4.html
    """
    x, y, gid, g = vdf.format_colnames(x, y, gid, g)

    if g:
        params = f"{gid}, {g}"

    elif x and y:
        params = f"{gid}, {x}, {y}"

    else:
        raise ValueError("Either 'x' and 'y' or 'g' must not be empty.")

    query = f"""
        SELECT 
            STV_Intersect({params} 
            USING PARAMETERS 
                index='{index}') 
            OVER (PARTITION BEST) AS (point_id, polygon_gid) 
        FROM {vdf}"""

    return vDataFrame(query)


@save_verticapy_logs
def split_polygon_n(p: str, nbins: int = 100) -> vDataFrame:
    """
    Splits a polygon into  (nbins :sup:`2`) smaller
    polygons of approximately equal total area.
    This  process  is inexact,  and  the  split
    polygons  have approximated edges;  greater
    values for nbins produces more accurate and
    precise edge approximations.

    Parameters
    ----------
    p: str
        String representation of the polygon.
    nbins: int, optional
        Number of bins used to cut the longitude
        and the latitude.  Split  polygons  have
        approximated  edges, and greater  values
        for  nbins  leads to more  accurate  and
        precise edge approximations.

    Returns
    -------
    vDataFrame
        output  vDataFrame that includes the new
        polygons.

    Examples
    --------
    .. code-block:: python

        import verticapy as vp
        from verticapy.sql.geo import split_polygon_n

        p = 'POLYGON ((121.334030916 31.5081948415, 121.334030917 31.5079167872, 121.333748304 31.5081948413, 121.334030916 31.5081948415))'
        poly = vp.vDataFrame({"triangle": [p]})
        poly["triangle"].apply("ST_GeomFromText({})")
        poly["triangle"].geo_plot(color="white",
                                  edgecolor="black")

    .. ipython:: python
        :suppress:

        import verticapy as vp
        from verticapy.sql.geo import split_polygon_n

        p = 'POLYGON ((121.334030916 31.5081948415, 121.334030917 31.5079167872, 121.333748304 31.5081948413, 121.334030916 31.5081948415))'
        poly = vp.vDataFrame({"triangle": [p]})
        poly["triangle"].apply("ST_GeomFromText({})")
        poly["triangle"].geo_plot(color="white", edgecolor="black")
        @savefig sql_geo_functions_split_polygon_n.png
        poly

    .. code-block:: python

        split_p = split_polygon_n(p)

    .. ipython:: python
        :suppress:

        split_p = split_polygon_n(p)
        html_file = open("SPHINX_DIRECTORY/figures/sql_geo_functions_split_polygon_n.html", "w")
        html_file.write(split_p._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_geo_functions_split_polygon_n.html

    .. code-block:: python

        split_p["geom"].geo_plot(color="white",
                                 edgecolor="black")

    .. ipython:: python
        :suppress:

        split_p["geom"].geo_plot(color="white", edgecolor="black")
        @savefig sql_geo_functions_split_polygon_n_2.png
        split_p
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
