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
        The gid cannot be ``NULL``.
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
        amount of memory in megabytes that ``create_index``
        can allocate during index construction.
    skip_nonindexable_polygons: bool, optional
        In rare cases, intricate polygons (for instance,
        those with too high resolution or anomalous
        spikes) cannot be indexed. These polygons are
        considered non-indexable.
        When set to False,  non-indexable polygons cause
        the index creation to fail. When set to ``True``,
        index   creation  can   succeed   by   excluding
        non-indexable polygons from the index.

    Returns
    -------
    TableSample
        geospatial indexes.

    Examples
    --------
    For this example, we will use the Cities and World
    dataset.

    .. code-block:: python

        import verticapy.datasets as vpd

        cities = vpd.load_cities()
        world = vpd.load_world()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_cities.html

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_world.html

    .. note::

        VerticaPy offers a wide range of sample
        datasets that are ideal for training
        and testing purposes. You can explore
        the full list of available datasets in
        the :ref:`api.datasets`, which provides
        detailed information on each dataset and
        how to use them effectively. These datasets
        are invaluable resources for honing your
        data analysis and machine learning skills
        within the VerticaPy environment.

    Let's preprocess the datasets by extracting latitude
    and longitude values and creating an index.

    .. code-block:: python

        world["id"] = "ROW_NUMBER() OVER(ORDER BY country, pop_est)"
        display(world)

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

    Let's create the geo-index.

    .. code-block:: python

        from verticapy.sql.geo import create_index

        create_index(world, "id", "geometry", "world_polygons", True)

    .. ipython:: python
        :suppress:

        html_file = open("SPHINX_DIRECTORY/figures/sql_geo_functions_intersect_4.html", "w")
        html_file.write(create_index(world, "id", "geometry", "world_polygons", True)._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_geo_functions_intersect_4.html

    Let's calculate the intersection between the
    cities and the various countries by using the
    GEOMETRY data type.

    .. code-block:: python

        from verticapy.sql.geo import intersect

        intersect(cities, "world_polygons", "id", "geometry")

    .. ipython:: python
        :suppress:

        html_file = open("SPHINX_DIRECTORY/figures/sql_geo_functions_intersect_3.html", "w")
        html_file.write(intersect(cities, "world_polygons", "id", "geometry")._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_geo_functions_intersect_3.html

    The same can be done using directly the longitude
    and latitude.

    .. code-block:: python

        intersect(cities, "world_polygons", "id", x="lat", y="lon")

    .. ipython:: python
        :suppress:

        html_file = open("SPHINX_DIRECTORY/figures/sql_geo_functions_intersect_4.html", "w")
        html_file.write(intersect(cities, "world_polygons", "id", x="lat", y="lon")._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_geo_functions_intersect_4.html

    .. note::

        For geospatial functions, Vertica utilizes indexing to
        expedite computations, especially considering the
        potentially extensive size of polygons.
        This is a unique optimization approach employed by
        Vertica in these scenarios.

    .. seealso::

        | :py:func:`~verticapy.sql.geo.describe_index` :
            Describes the geo index.
        | :py:func:`~verticapy.sql.geo.intersect` :
            Spatially intersects a point
            or points with a set of polygons.
        | :py:func:`~verticapy.sql.geo.rename_index` :
            Renames the geo index.
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
        :py:class:`~vDataFrame` instead of a
        :py:class:`~verticapy.core.tablesample.base.TableSample`.

    Returns
    -------
    TableSample
        geospatial indexes.

    Examples
    --------
    Describes all indexes:

    .. code-block:: python

        from verticapy.sql.geo import describe_index

        describe_index()

    .. ipython:: python
        :suppress:

        from verticapy.sql.geo import describe_index

        html_file = open("SPHINX_DIRECTORY/figures/sql_geo_index_describe_index_1.html", "w")
        html_file.write(describe_index()._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_geo_index_describe_index_1.html

    Describes a specific index:

    .. code-block:: python

        describe_index("world_polygons")

    .. ipython:: python
        :suppress:

        html_file = open("SPHINX_DIRECTORY/figures/sql_geo_index_describe_index_2.html", "w")
        html_file.write(describe_index("world_polygons")._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_geo_index_describe_index_2.html

    Describes all geometries of a specific index:

    .. code-block:: python

        describe_index(
            "world_polygons",
            list_polygons = True,
        )

    .. ipython:: python
        :suppress:

        html_file = open("SPHINX_DIRECTORY/figures/sql_geo_index_describe_index_3.html", "w")
        html_file.write(describe_index("world_polygons", list_polygons=True)._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_geo_index_describe_index_3.html

    .. note::

        For geospatial functions, Vertica utilizes indexing to
        expedite computations, especially considering the
        potentially extensive size of polygons.
        This is a unique optimization approach employed by
        Vertica in these scenarios.

    .. seealso::

        | :py:func:`~verticapy.sql.geo.create_index` :
            Creates the geo index.
        | :py:func:`~verticapy.sql.geo.intersect` :
            Spatially intersects a point
            or points with a set of polygons.
        | :py:func:`~verticapy.sql.geo.rename_index` :
            Renames the geo index.
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

    Examples
    --------
    Describes all indexes:

    .. code-block:: python

        from verticapy.sql.geo import describe_index

        describe_index()

    .. ipython:: python
        :suppress:

        from verticapy.sql.geo import rename_index, describe_index

        html_file = open("SPHINX_DIRECTORY/figures/sql_geo_index_rename_index_1.html", "w")
        html_file.write(describe_index()._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_geo_index_rename_index_1.html

    Renames a specific index:

    .. ipython:: python
        :okwarning:

        from verticapy.sql.geo import rename_index

        rename_index("world_polygons", "world_polygons_new")

    Index now has the new name:

    .. code-block:: python

        describe_index()

    .. ipython:: python
        :suppress:

        html_file = open("SPHINX_DIRECTORY/figures/sql_geo_index_rename_index_2.html", "w")
        html_file.write(describe_index()._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_geo_index_rename_index_2.html

    .. note::

        For geospatial functions, Vertica utilizes indexing to
        expedite computations, especially considering the
        potentially extensive size of polygons.
        This is a unique optimization approach employed by
        Vertica in these scenarios.

    .. seealso::

        | :py:func:`~verticapy.sql.geo.create_index` :
            Creates the geo index.
        | :py:func:`~verticapy.sql.geo.describe_index` :
            Describes the geo index.
        | :py:func:`~verticapy.sql.geo.intersect` :
            Spatially intersects a point
            or points with a set of polygons.
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
