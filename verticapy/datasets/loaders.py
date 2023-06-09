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
import os
import vertica_python
from typing import Optional

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
    schema: str,
    name: str,
    dtype: dict,
    copy_cols: Optional[list] = None,
    dataset_name: Optional[str] = None,
) -> vDataFrame:
    """
    General Function to ingest a dataset.
    """
    copy_cols = format_type(copy_cols, dtype=list)

    try:
        vdf = vDataFrame(name, schema=schema)

    except:
        name = quote_ident(name)
        schema = "v_temp_schema" if not schema else quote_ident(schema)
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

            _executeSQL("COMMIT;", title="Commit.")
            vdf = vDataFrame(name, schema=schema)

        except:
            drop(f"{schema}.{name}", method="table")
            raise

    return vdf


"""
Datasets for basic Data Exploration.
"""


@save_verticapy_logs
def load_market(schema: str = "public", name: str = "market") -> vDataFrame:
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
        Schema of the new relation. If empty, a
        temporary local table is created.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the market vDataFrame.
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
def load_iris(schema: str = "public", name: str = "iris") -> vDataFrame:
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
        Schema  of  the  new  relation.  If  empty, a
        temporary local table is created.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the iris vDataFrame.
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
def load_titanic(schema: str = "public", name: str = "titanic") -> vDataFrame:
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
        Schema of  the new relation. If empty, a
        temporary local table is created.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the titanic vDataFrame.
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
def load_winequality(schema: str = "public", name: str = "winequality") -> vDataFrame:
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
        Schema  of  the  new  relation.  If empty, a
        temporary local table is created.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the winequality vDataFrame.
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
    schema: str = "public", name: str = "airline_passengers"
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
        Schema of the new relation. If empty, a
        temporary local table is created.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the airline passengers vDataFrame.
    """
    return load_dataset(
        schema=schema,
        name=name,
        dtype={"date": "Date", "passengers": "Integer"},
        dataset_name="airline_passengers",
    )


@save_verticapy_logs
def load_amazon(schema: str = "public", name: str = "amazon") -> vDataFrame:
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
        Schema  of  the  new  relation.  If  empty, a
        temporary local table is created.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the amazon vDataFrame.
    """
    return load_dataset(
        schema=schema,
        name=name,
        dtype={"date": "Date", "state": "Varchar(32)", "number": "Integer"},
        dataset_name="amazon",
    )


@save_verticapy_logs
def load_commodities(schema: str = "public", name: str = "commodities") -> vDataFrame:
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
        Schema  of  the  new  relation.  If  empty, a
        temporary local table is created.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the commodities vDataFrame.
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
def load_gapminder(schema: str = "public", name: str = "gapminder") -> vDataFrame:
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
        Schema  of  the  new  relation.  If  empty, a
        temporary local table is created.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the gapminder vDataFrame.
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
def load_pop_growth(schema: str = "public", name: str = "pop_growth") -> vDataFrame:
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
        Schema of the new relation. If empty, a
        temporary local table is created.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the pop growth vDataFrame.
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
def load_smart_meters(schema: str = "public", name: str = "smart_meters") -> vDataFrame:
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
        Schema  of  the  new  relation. If  empty, a
        temporary local table is created.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the smart meters vDataFrame.
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
def load_cities(schema: str = "public", name: str = "cities") -> vDataFrame:
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
        Schema  of  the  new  relation.  If empty, a
        temporary local table is created.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the Cities vDataFrame.
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
def load_world(schema: str = "public", name: str = "world") -> vDataFrame:
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
        Schema  of  the  new  relation.  If empty, a
        temporary local table is created.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the World vDataFrame.
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


"""
Datasets for Complex Data Analysis.
"""


@save_verticapy_logs
def load_laliga(schema: str = "public", name: str = "laliga") -> vDataFrame:
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
        Schema of the new relation. If empty, a
        temporary local table is created.
    name: str, optional
        Name of the new relation.

    Returns
    -------
    vDataFrame
        the LaLiga vDataFrame.
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
