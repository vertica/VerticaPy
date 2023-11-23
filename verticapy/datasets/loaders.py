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
import os
import vertica_python
from typing import Optional

import verticapy._config.config as conf
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import format_type, quote_ident
from verticapy._utils._sql._sys import _executeSQL
from verticapy.connection import current_cursor

from verticapy.core.vdataframe.base import vDataFrame

from verticapy.sql.create import create_table
from verticapy.sql.drop import drop

"""
General Functions.
"""


def load_dataset(
    schema: Optional[str],
    name: str,
    dtype: dict,
    copy_cols: Optional[list] = None,
    dataset_name: Optional[str] = None,
) -> vDataFrame:
    """
    General Function to ingest a dataset.
    """
    copy_cols = format_type(copy_cols, dtype=list)
    if not (schema):
        schema = conf.get_option("temp_schema")

    try:
        vdf = vDataFrame(name, schema=schema)

    except:
        name = quote_ident(name)
        create_table(table_name=name, dtype=dtype, schema=schema)

        try:
            path = os.path.dirname(__file__)
            if dataset_name in ("laliga",):
                path += f"/data/{dataset_name}/*.json"
                query = f"COPY {schema}.{name} FROM {{}} PARSER FJsonParser();"
            else:
                path += f"/data/{dataset_name}.csv"
                if len(copy_cols) == 0:
                    copy_cols = quote_ident(dtype)
                query = f"""
                    COPY
                        {schema}.{name}({', '.join(copy_cols)})
                    FROM {{}} DELIMITER ','
                              NULL ''
                              ENCLOSED BY '\"'
                              ESCAPE AS '\\'
                              SKIP 1;"""

            cur = current_cursor()

            if isinstance(cur, vertica_python.vertica.cursor.Cursor) and (
                dataset_name not in ("laliga",)
            ):
                query = query.format("STDIN")
                _executeSQL(
                    query, title="Ingesting the data.", method="copy", path=path
                )

            else:
                query = query.format(f"LOCAL '{path}'")
                _executeSQL(query, title="Ingesting the data.")

            vdf = vDataFrame(name, schema=schema)

        except:
            drop(f"{schema}.{name}", method="table")
            raise

    return vdf


"""
Datasets for basic Data Exploration.
"""


@save_verticapy_logs
def load_market(schema: Optional[str] = None, name: str = "market") -> vDataFrame:
    """
    Ingests  the  market  dataset into  the  Vertica
    database.
    This  dataset  is  ideal  for  data exploration.
    If  a  table  with  the  same  name  and  schema
    already  exists, this  function  creates  a
    vDataFrame from the input relation.

    Parameters
    ----------
    schema: str, optional
        Schema of the new relation. If empty,
        the temporary schema is used.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the market vDataFrame.

    Example
    -------
    If you call this loader without any arguments, the dataset is
    loaded using the default schema (public).

    .. code-block:: python

        from verticapy.datasets import load_market

        load_market()

    .. ipython:: python
        :suppress:

        from verticapy.datasets import load_market
        html_file = open("SPHINX_DIRECTORY/figures/datasets_loaders_load_market.html", "w")
        html_file.write(load_market()._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_market.html
    """
    return load_dataset(
        schema=schema,
        name=name,
        dtype={"Form": "Varchar(32)", "Name": "Varchar(32)", "Price": "Float"},
        dataset_name="market",
    )


"""
Datasets for Classification.
"""


@save_verticapy_logs
def load_iris(schema: Optional[str] = None, name: str = "iris") -> vDataFrame:
    """
    Ingests   the  iris  dataset   into  the  Vertica
    database.
    This  dataset  is  ideal  for  classification and
    clustering  models.  If a  table  with  the  same
    name  and  schema  already exists, this  function
    creates a vDataFrame from the input relation.

    Parameters
    ----------
    schema: str, optional
        Schema of the new relation. If empty,
        the temporary schema is used.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the iris vDataFrame.

    Example
    -------
    If you call this loader without any arguments, the dataset is
    loaded using the default schema (public).

    .. code-block:: python

        from verticapy.datasets import load_iris

        load_iris()

    .. ipython:: python
        :suppress:

        from verticapy.datasets import load_iris
        html_file = open("SPHINX_DIRECTORY/figures/datasets_loaders_load_iris.html", "w")
        html_file.write(load_iris()._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_iris.html
    """
    return load_dataset(
        schema=schema,
        name=name,
        dtype={
            "SepalLengthCm": "Numeric(5,2)",
            "SepalWidthCm": "Numeric(5,2)",
            "PetalLengthCm": "Numeric(5,2)",
            "PetalWidthCm": "Numeric(5,2)",
            "Species": "Varchar(30)",
        },
        copy_cols=[
            "Id FILLER Integer",
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm",
            "Species",
        ],
        dataset_name="iris",
    )


@save_verticapy_logs
def load_titanic(schema: Optional[str] = None, name: str = "titanic") -> vDataFrame:
    """
    Ingests the titanic dataset into the Vertica
    database.
    This  dataset  is  ideal  for classification
    models.  If a table  with the  same name and
    schema  already exists,  this  function
    creates a vDataFrame from the input relation.

    Parameters
    ----------
    schema: str, optional
        Schema of the new relation. If empty,
        the temporary schema is used.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the titanic vDataFrame.

    Example
    -------
    If you call this loader without any arguments, the dataset is
    loaded using the default schema (public).

    .. code-block:: python

        from verticapy.datasets import load_titanic

        load_titanic()

    .. ipython:: python
        :suppress:

        from verticapy.datasets import load_titanic
        html_file = open("SPHINX_DIRECTORY/figures/datasets_loaders_load_titanic.html", "w")
        html_file.write(load_titanic()._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_titanic.html
    """
    return load_dataset(
        schema=schema,
        name=name,
        dtype={
            "pclass": "Integer",
            "survived": "Integer",
            "name": "Varchar(164)",
            "sex": "Varchar(20)",
            "age": "Numeric(6,3)",
            "sibsp": "Integer",
            "parch": "Integer",
            "ticket": "Varchar(36)",
            "fare": "Numeric(10,5)",
            "cabin": "Varchar(30)",
            "embarked": "Varchar(20)",
            "boat": "Varchar(100)",
            "body": "Integer",
            "home.dest": "Varchar(100)",
        },
        dataset_name="titanic",
    )


"""
Datasets for Regression.
"""


@save_verticapy_logs
def load_winequality(
    schema: Optional[str] = None, name: str = "winequality"
) -> vDataFrame:
    """
    Ingests  the winequality dataset into the Vertica
    database.
    This   dataset  is  ideal  for   regression   and
    classification  models. If a table with the  same
    name  and  schema  already exists, this  function
    creates a vDataFrame from the input relation.

    Parameters
    ----------
    schema: str, optional
        Schema of the new relation. If empty,
        the temporary schema is used.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the winequality vDataFrame.

    Example
    -------
    If you call this loader without any arguments, the dataset is
    loaded using the default schema (public).

    .. code-block:: python

        from verticapy.datasets import load_winequality

        load_winequality()

    .. ipython:: python
        :suppress:

        from verticapy.datasets import load_winequality
        html_file = open("SPHINX_DIRECTORY/figures/datasets_loaders_load_winequality.html", "w")
        html_file.write(load_winequality()._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_winequality.html
    """
    return load_dataset(
        schema=schema,
        name=name,
        dtype={
            "fixed_acidity": "Numeric(6,3)",
            "volatile_acidity": "Numeric(7,4)",
            "citric_acid": "Numeric(6,3)",
            "residual_sugar": "Numeric(7,3)",
            "chlorides": "Float",
            "free_sulfur_dioxide": "Numeric(7,2)",
            "total_sulfur_dioxide": "Numeric(7,2)",
            "density": "Float",
            "pH": "Numeric(6,3)",
            "sulphates": "Numeric(6,3)",
            "alcohol": "Float",
            "quality": "Integer",
            "good": "Integer",
            "color": "Varchar(20)",
        },
        dataset_name="winequality",
    )


"""
Datasets for Time Series.
"""


@save_verticapy_logs
def load_airline_passengers(
    schema: Optional[str] = None, name: str = "airline_passengers"
) -> vDataFrame:
    """
    Ingests  the airline passengers dataset into  the
    Vertica database.
    This  dataset  is  ideal  for  time   series  and
    regression  models.  If a  table  with  the  same
    name  and  schema  already exists, this  function
    creates a vDataFrame from the input relation.

    Parameters
    ----------
    schema: str, optional
        Schema of the new relation. If empty,
        the temporary schema is used.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the airline passengers vDataFrame.

    Example
    -------
    If you call this loader without any arguments, the dataset is
    loaded using the default schema (public).

    .. code-block:: python

        from verticapy.datasets import load_airline_passengers

        load_airline_passengers()

    .. ipython:: python
        :suppress:

        from verticapy.datasets import load_airline_passengers
        html_file = open("SPHINX_DIRECTORY/figures/datasets_loaders_load_airline_passengers.html", "w")
        html_file.write(load_airline_passengers()._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_airline_passengers.html
    """
    return load_dataset(
        schema=schema,
        name=name,
        dtype={"date": "Date", "passengers": "Integer"},
        dataset_name="airline_passengers",
    )


@save_verticapy_logs
def load_amazon(schema: Optional[str] = None, name: str = "amazon") -> vDataFrame:
    """
    Ingests  the  amazon  dataset  into  the  Vertica
    database.
    This  dataset  is  ideal  for  time   series  and
    regression  models.  If a  table  with  the  same
    name  and  schema  already exists, this  function
    creates a vDataFrame from the input relation.

    Parameters
    ----------
    schema: str, optional
        Schema of the new relation. If empty,
        the temporary schema is used.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the amazon vDataFrame.

    Example
    -------
    If you call this loader without any arguments, the dataset is
    loaded using the default schema (public).

    .. code-block:: python

        from verticapy.datasets import load_amazon

        load_amazon()

    .. ipython:: python
        :suppress:

        from verticapy.datasets import load_amazon
        html_file = open("SPHINX_DIRECTORY/figures/datasets_loaders_load_amazon.html", "w")
        html_file.write(load_amazon()._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_amazon.html
    """
    return load_dataset(
        schema=schema,
        name=name,
        dtype={"date": "Date", "state": "Varchar(32)", "number": "Integer"},
        dataset_name="amazon",
    )


@save_verticapy_logs
def load_commodities(
    schema: Optional[str] = None, name: str = "commodities"
) -> vDataFrame:
    """
    Ingests the commodities  dataset into the Vertica
    database.
    This  dataset  is  ideal  for  time   series  and
    regression  models.  If a  table  with  the  same
    name  and  schema  already exists, this  function
    creates a vDataFrame from the input relation.

    Parameters
    ----------
    schema: str, optional
        Schema of the new relation. If empty,
        the temporary schema is used.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the commodities vDataFrame.

    Example
    -------
    If you call this loader without any arguments, the dataset is
    loaded using the default schema (public).

    .. code-block:: python

        from verticapy.datasets import load_commodities

        load_commodities()

    .. ipython:: python
        :suppress:

        from verticapy.datasets import load_commodities
        html_file = open("SPHINX_DIRECTORY/figures/datasets_loaders_load_commodities.html", "w")
        html_file.write(load_commodities()._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_commodities.html
    """
    return load_dataset(
        schema=schema,
        name=name,
        dtype={
            "date": "Date",
            "Gold": "Float",
            "Oil": "Float",
            "Spread": "Float",
            "Vix": "Float",
            "Dol_Eur": "Float",
            "SP500": "Float",
        },
        dataset_name="commodities",
    )


@save_verticapy_logs
def load_gapminder(schema: Optional[str] = None, name: str = "gapminder") -> vDataFrame:
    """
    Ingests  the gapminder  dataset into the  Vertica
    database.
    This  dataset  is  ideal  for  time   series  and
    regression  models.  If a  table  with  the  same
    name  and  schema  already exists, this  function
    creates a vDataFrame from the input relation.

    Parameters
    ----------
    schema: str, optional
        Schema of the new relation. If empty,
        the temporary schema is used.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the gapminder vDataFrame.

    Example
    -------
    If you call this loader without any arguments, the dataset is
    loaded using the default schema (public).

    .. code-block:: python

        from verticapy.datasets import load_gapminder

        load_gapminder()

    .. ipython:: python
        :suppress:

        from verticapy.datasets import load_gapminder
        html_file = open("SPHINX_DIRECTORY/figures/datasets_loaders_load_gapminder.html", "w")
        html_file.write(load_gapminder()._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_gapminder.html
    """
    return load_dataset(
        schema=schema,
        name=name,
        dtype={
            "country": "Varchar(96)",
            "year": "Integer",
            "pop": "Integer",
            "continent": "Varchar(52)",
            "lifeExp": "Float",
            "gdpPercap": "Float",
        },
        dataset_name="gapminder",
    )


@save_verticapy_logs
def load_pop_growth(
    schema: Optional[str] = None, name: str = "pop_growth"
) -> vDataFrame:
    """
    Ingests  the  population growth dataset into  the
    Vertica database.
    This  dataset  is  ideal  for  time   series  and
    geospatial  models.  If a  table  with  the  same
    name  and  schema  already exists, this  function
    creates a vDataFrame from the input relation.

    Parameters
    ----------
    schema: str, optional
        Schema of the new relation. If empty,
        the temporary schema is used.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the pop growth vDataFrame.

    Example
    -------
    If you call this loader without any arguments, the dataset is
    loaded using the default schema (public).

    .. code-block:: python

        from verticapy.datasets import load_pop_growth

        load_pop_growth()

    .. ipython:: python
        :suppress:

        from verticapy.datasets import load_pop_growth
        html_file = open("SPHINX_DIRECTORY/figures/datasets_loaders_load_pop_growth.html", "w")
        html_file.write(load_pop_growth()._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_pop_growth.html
    """
    return load_dataset(
        schema=schema,
        name=name,
        dtype={
            "year": "Int",
            "continent": "Varchar(100)",
            "country": "Varchar(100)",
            "city": "Varchar(100)",
            "population": "Float",
            "lat": "Float",
            "lon": "Float",
        },
        dataset_name="pop_growth",
    )


@save_verticapy_logs
def load_smart_meters(
    schema: Optional[str] = None, name: str = "smart_meters"
) -> vDataFrame:
    """
    Ingests the smart meters dataset into the Vertica
    database.
    This  dataset  is  ideal  for  time   series  and
    regression  models.  If a  table  with  the  same
    name  and  schema  already exists, this  function
    creates a vDataFrame from the input relation.

    Parameters
    ----------
    schema: str, optional
        Schema of the new relation. If empty,
        the temporary schema is used.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the smart meters vDataFrame.

    Example
    -------
    If you call this loader without any arguments, the dataset is
    loaded using the default schema (public).

    .. code-block:: python

        from verticapy.datasets import load_smart_meters

        load_smart_meters()

    .. ipython:: python
        :suppress:

        from verticapy.datasets import load_smart_meters
        html_file = open("SPHINX_DIRECTORY/figures/datasets_loaders_load_smart_meters.html", "w")
        html_file.write(load_smart_meters()._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_smart_meters.html
    """
    return load_dataset(
        schema=schema,
        name=name,
        dtype={"time": "Timestamp", "val": "Numeric(11,7)", "id": "Integer"},
        dataset_name="smart_meters",
    )


"""
Datasets for Geospatial.
"""


@save_verticapy_logs
def load_cities(schema: Optional[str] = None, name: str = "cities") -> vDataFrame:
    """
    Ingests  the  Cities  dataset  into the  Vertica
    database.
    This  dataset  is  ideal  for  geospatial models.
    If  a  table  with   the  same  name  and  schema
    already  exists,  this  function  creates  a
    vDataFrame from the input relation.

    Parameters
    ----------
    schema: str, optional
        Schema of the new relation. If empty,
        the temporary schema is used.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the Cities vDataFrame.

    Example
    -------
    If you call this loader without any arguments, the dataset is
    loaded using the default schema (public).

    .. code-block:: python

        from verticapy.datasets import load_cities

        load_cities()

    .. ipython:: python
        :suppress:

        from verticapy.datasets import load_cities
        html_file = open("SPHINX_DIRECTORY/figures/datasets_loaders_load_cities.html", "w")
        html_file.write(load_cities()._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_cities.html
    """
    return load_dataset(
        schema=schema,
        name=name,
        dtype={"city": "Varchar(82)", "geometry": "Geometry"},
        copy_cols=[
            "city",
            "gx FILLER LONG VARCHAR(65000)",
            "geometry AS ST_GeomFromText(gx)",
        ],
        dataset_name="cities",
    )


@save_verticapy_logs
def load_world(schema: Optional[str] = None, name: str = "world") -> vDataFrame:
    """
    Ingests  the  World  dataset  into  the  Vertica
    database.
    This  dataset  is  ideal  for  geospatial models.
    If a  table  with  the  same name  and  schema
    already exists, this  function creates a
    vDataFrame from the input relation.

    Parameters
    ----------
    schema: str, optional
        Schema of the new relation. If empty,
        the temporary schema is used.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the World vDataFrame.

    Example
    -------
    If you call this loader without any arguments, the dataset is
    loaded using the default schema (public).

    .. code-block:: python

        from verticapy.datasets import load_world

        load_world()

    .. ipython:: python
        :suppress:

        from verticapy.datasets import load_world
        html_file = open("SPHINX_DIRECTORY/figures/datasets_loaders_load_world.html", "w")
        html_file.write(load_world()._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_world.html
    """
    return load_dataset(
        schema=schema,
        name=name,
        dtype={
            "pop_est": "Int",
            "continent": "Varchar(32)",
            "country": "Varchar(82)",
            "geometry": "Geometry",
        },
        copy_cols=[
            "pop_est",
            "continent",
            "country",
            "gx FILLER LONG VARCHAR(65000)",
            "geometry AS ST_GeomFromText(gx)",
        ],
        dataset_name="world",
    )


@save_verticapy_logs
def load_africa_education(
    schema: Optional[str] = None, name: str = "africa_education"
) -> vDataFrame:
    """
    Ingests  the  Africe Education  dataset  into  the  Vertica
    database.
    This  dataset  is  ideal  for  geospatial models.
    If a  table  with  the  same name  and  schema
    already exists, this  function creates a
    vDataFrame from the input relation.

    Parameters
    ----------
    schema: str, optional
        Schema of the new relation. If empty,
        the temporary schema is used.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the Africa Education vDataFrame.

    Example
    -------
    If you call this loader without any arguments, the dataset is
    loaded using the default schema (public).

    .. code-block:: python

        from verticapy.datasets import load_africa_education

        load_africa_education()

    .. ipython:: python
        :suppress:

        from verticapy.datasets import load_africa_education
        html_file = open("SPHINX_DIRECTORY/figures/datasets_loaders_load_africa_education.html", "w")
        html_file.write(load_africa_education()._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_africa_education.html
    """
    return load_dataset(
        schema=schema,
        name=name,
        dtype={
            "PABSENT": "Integer",
            "SPUPPR16": "Varchar(20)",
            "zpmealsc": "Varchar(32)",
            "PREPEAT": "Varchar(20)",
            "zpses": "Numeric(7,3)",
            "SPUPPR06": "Varchar(20)",
            "zraloct": "Float",
            "COUNTRY": "Varchar(20)",
            "XSEX": "Varchar(20)",
            "lon": "Numeric(11,7)",
            "zralocp": "Float",
            "district": "Varchar(46)",
            "SCHOOL": "Integer",
            "ZRALEVP": "Integer",
            "SPUPPR13": "Varchar(20)",
            "ZRALEVT": "Numeric(6,3)",
            "SPUPPR09": "Varchar(20)",
            "SPUPPR10": "Varchar(20)",
            "zpsit": "Varchar(54)",
            "PNURSERY": "Varchar(38)",
            "STCHPR08": "Varchar(20)",
            "country_long": "Varchar(24)",
            "XQPROFES": "Varchar(20)",
            "PTRAVEL2": "Varchar(20)",
            "PTRAVEL": "Varchar(20)",
            "lat": "Numeric(11,7)",
            "PLIGHT": "Varchar(20)",
            "REGION": "Varchar(10)",
            "PUPIL": "Integer",
            "SUPPR17": "Varchar(20)",
            "PMOTHER": "Varchar(80)",
            "STYPE": "Varchar(20)",
            "SPUPPR07": "Varchar(20)",
            "SPUPPR14": "Varchar(20)",
            "PMALIVE": "Varchar(10)",
            "zmalocp": "Numeric(11,7)",
            "STCHPR06": "Varchar(20)",
            "XNUMYRS": "Integer",
            "PFATHER": "Varchar(10)",
            "zsdist": "Numeric(9,7)",
            "PSEX": "Varchar(10)",
            "SLOCAT": "Varchar(20)",
            "ZMALEVP": "Numeric(8,5)",
            "province": "Varchar(60)",
            "zphmwkhl": "Varchar(56)",
            "SPUPPR04": "Varchar(20)",
            "SPUPPR11": "Varchar(20)",
            "STCHPR07": "Varchar(20)",
            "SQACADEM": "Varchar(22)",
            "STCHPR04": "Varchar(20)",
            "SINS2006": "Numeric(9,7)",
            "numstu": "Integer",
            "ZMALEVT": "Numeric(9,7)",
            "PFALIVE": "Boolean",
            "STCHPR09": "Varchar(20)",
            "SPUPPR15": "Varchar(20)",
            "PENGLISH": "Varchar(32)",
            "SPUPPR12": "Varchar(20)",
            "zpsibs": "Integer",
            "XAGE": "Numeric(9,7)",
            "SPUPPR08": "Varchar(20)",
            "PAGE": "Integer",
            "schoolname": "Varchar(16)",
        },
        copy_cols=[
            "PABSENT",
            "SPUPPR16",
            "zpmealsc",
            "PREPEAT",
            "zpses",
            "SPUPPR06",
            "zraloct",
            "COUNTRY",
            "XSEX",
            "lon",
            "zralocp",
            "district",
            "SCHOOL",
            "ZRALEVP",
            "SPUPPR13",
            "ZRALEVT",
            "SPUPPR09",
            "SPUPPR10",
            "zpsit",
            "PNURSERY",
            "STCHPR08",
            "country_long",
            "XQPROFES",
            "PTRAVEL2",
            "PTRAVEL",
            "lat",
            "PLIGHT",
            "REGION",
            "PUPIL",
            "SUPPR17",
            "PMOTHER",
            "STYPE",
            "SPUPPR07",
            "SPUPPR14",
            "PMALIVE",
            "zmalocp",
            "STCHPR06",
            "XNUMYRS",
            "PFATHER",
            "zsdist",
            "PSEX",
            "SLOCAT",
            "ZMALEVP",
            "province",
            "zphmwkhl",
            "SPUPPR04",
            "SPUPPR11",
            "STCHPR07",
            "SQACADEM",
            "STCHPR04",
            "SINS2006",
            "numstu",
            "ZMALEVT",
            "PFALIVE",
            "STCHPR09",
            "SPUPPR15",
            "PENGLISH",
            "SPUPPR12",
            "zpsibs",
            "XAGE",
            "SPUPPR08",
            "PAGE",
            "schoolname",
        ],
        dataset_name="africa_education",
    )


"""
Datasets for Complex Data Analysis.
"""


@save_verticapy_logs
def load_laliga(schema: Optional[str] = None, name: str = "laliga") -> vDataFrame:
    """
    Ingests the  LaLiga dataset into the Vertica
    database.
    This dataset is  ideal for testing  complex
    data types. If a table  with  the same name
    and  schema  already exists, this  function
    creates a vDataFrame from the input
    relation.

    Parameters
    ----------
    schema: str, optional
        Schema of the new relation. If empty,
        the temporary schema is used.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the LaLiga vDataFrame.

    Example
    -------
    If you call this loader without any arguments, the dataset is
    loaded using the default schema (public).

    .. code-block:: python

        from verticapy.datasets import load_laliga

        load_laliga()

    .. ipython:: python
        :suppress:

        from verticapy.datasets import load_laliga
        html_file = open("SPHINX_DIRECTORY/figures/datasets_loaders_load_laliga.html", "w")
        html_file.write(load_laliga()._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_laliga.html
    """
    return load_dataset(
        schema=schema,
        name=name,
        dtype={
            "away_score": "int",
            "away_team": """Row("away_team_gender" varchar,
                                "away_team_group" varchar,
                                "away_team_id" int,
                                "away_team_name" varchar,
                                "country" Row("id" int, "name" varchar), 
                                "managers" Array[Row("country" Row("id" int, 
                                                                   "name" varchar), 
                                                     "dob" date, 
                                                     "id" int,
                                                     "name" varchar, 
                                                     "nickname" varchar)])""",
            "competition": """Row("competition_id" int, 
                                  "competition_name" varchar,
                                  "country_name" varchar)""",
            "competition_stage": 'Row("id" int, "name" varchar)',
            "home_score": "int",
            "home_team": """Row("country" Row("id" int, "name" varchar), 
                                "home_team_gender" varchar, 
                                "home_team_group" varchar, 
                                "home_team_id" int, 
                                "home_team_name" varchar, 
                                "managers" Array[Row("country" Row("id" int,
                                                                   "name" varchar), 
                                                     "dob" date, 
                                                     "id" int, 
                                                     "name" varchar, 
                                                     "nickname" varchar)])""",
            "kick_off": "time",
            "last_updated": "date",
            "match_date": "date",
            "match_id": "int",
            "match_status": "varchar",
            "match_week": "int",
            "metadata": """Row("data_version" date, 
                               "shot_fidelity_version" int, 
                               "xy_fidelity_version" int)""",
            "season": 'Row("season_id" int, "season_name" varchar)',
        },
        dataset_name="laliga",
    )
