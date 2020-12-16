# (c) Copyright [2018-2020] Micro Focus or one of its affiliates.
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
import os

# VerticaPy Modules
import verticapy
from verticapy import vDataFrame
from verticapy.utilities import *
from verticapy.toolbox import *

# ---#
def load_dataset(
    cursor, schema: str, name: str, str_create: str, str_copy: str, dataset_name: str
):
    """
    General Function to ingest a dataset
    """
    check_types([("schema", schema, [str],), ("name", name, [str],)])
    cursor = check_cursor(cursor)[0]
    try:
        vdf = vDataFrame(name, cursor, schema=schema)
    except:
        cursor.execute(
            "CREATE TABLE {}.{}({});".format(
                str_column(schema), str_column(name), str_create,
            )
        )
        try:
            path = os.path.dirname(verticapy.__file__) + "/learn/data/{}.csv".format(
                dataset_name
            )
            query = "COPY {}.{}({}) FROM {} DELIMITER ',' NULL '' ENCLOSED BY '\"' ESCAPE AS '\\' SKIP 1;".format(
                str_column(schema), str_column(name), str_copy, "{}"
            )
            import vertica_python

            if isinstance(cursor, vertica_python.vertica.cursor.Cursor):
                with open(path, "r") as fs:
                    cursor.copy(query.format("STDIN"), fs)
            else:
                cursor.execute(query.format("LOCAL '{}'".format(path)))
            cursor.execute("COMMIT;")
            vdf = vDataFrame(name, cursor, schema=schema)
        except:
            cursor.execute(
                "DROP TABLE {}.{}".format(str_column(schema), str_column(name))
            )
            raise
    return vdf


#
#
# ---#
def load_amazon(cursor=None, schema: str = "public", name: str = "amazon"):
    """
---------------------------------------------------------------------------
Ingests the amazon dataset in the Vertica DB (Dataset ideal for TS and
Regression). If a table with the same name and schema already exists, 
this function will create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
	Vertica DB cursor. 
schema: str, optional
	Schema of the new relation. The default schema is public.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the amazon vDataFrame.

See Also
--------
load_commodities  : Ingests the commodities dataset in the Vertica DB.
    (Time Series / Regression).
load_iris         : Ingests the iris dataset in the Vertica DB.
	(Clustering / Classification).
load_market       : Ingests the market dataset in the Vertica DB.
	(Basic Data Exploration).
load_smart_meters : Ingests the smart meters dataset in the Vertica DB.
	(Time Series / Regression).
load_titanic      : Ingests the titanic dataset in the Vertica DB.
	(Classification).
load_winequality  : Ingests the winequality dataset in the Vertica DB.
	(Regression / Classification).
	"""
    return load_dataset(
        cursor,
        schema,
        name,
        '"date" Date, "state" Varchar(32), "number" Integer',
        '"date", "state", "number"',
        "amazon",
    )


# ---#
def load_cities(cursor=None, schema: str = "public", name: str = "cities"):
    """
---------------------------------------------------------------------------
Ingests the Cities dataset in the Vertica DB (Dataset ideal for Geospatial). 
If a table with the same name and schema already exists, this function will 
create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
    Vertica DB cursor. 
schema: str, optional
    Schema of the new relation. The default schema is public.
name: str, optional
    Name of the new relation.

Returns
-------
vDataFrame
    the Cities vDataFrame.

See Also
--------
load_world : Ingests the World dataset in the Vertica DB (Geospatial).
    """
    return load_dataset(
        cursor,
        schema,
        name,
        '"city" Varchar(82), "geometry" Geometry',
        '"city", gx FILLER LONG VARCHAR(65000), "geometry" AS ST_GeomFromText(gx)',
        "cities",
    )


# ---#
def load_commodities(cursor=None, schema: str = "public", name: str = "commodities"):
    """
---------------------------------------------------------------------------
Ingests the commodities dataset in the Vertica DB (Dataset ideal for TS and
Regression). If a table with the same name and schema already exists, 
this function will create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
    Vertica DB cursor. 
schema: str, optional
    Schema of the new relation. The default schema is public.
name: str, optional
    Name of the new relation.

Returns
-------
vDataFrame
    the amazon vDataFrame.

See Also
--------
load_amazon       : Ingests the amazon dataset in the Vertica DB.
    (Time Series / Regression).
load_iris         : Ingests the iris dataset in the Vertica DB.
    (Clustering / Classification).
load_market       : Ingests the market dataset in the Vertica DB.
    (Basic Data Exploration).
load_smart_meters : Ingests the smart meters dataset in the Vertica DB.
    (Time Series / Regression).
load_titanic      : Ingests the titanic dataset in the Vertica DB.
    (Classification).
load_winequality  : Ingests the winequality dataset in the Vertica DB.
    (Regression / Classification).
    """
    return load_dataset(
        cursor,
        schema,
        name,
        '"date" Date, "Gold" Float, "Oil" Float, "Spread" Float, "Vix" Float, "Dol_Eur" Float, "SP500" Float',
        '"date", "Gold", "Oil", "Spread", "Vix", "Dol_Eur", "SP500"',
        "commodities",
    )


# ---#
def load_iris(cursor=None, schema: str = "public", name: str = "iris"):
    """
---------------------------------------------------------------------------
Ingests the iris dataset in the Vertica DB (Dataset ideal for Classification
and Clustering). If a table with the same name and schema already exists, 
this function will create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
	Vertica DB cursor. 
schema: str, optional
	Schema of the new relation. The default schema is public.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the iris vDataFrame.

See Also
--------
load_amazon       : Ingests the amazon dataset in the Vertica DB.
	(Time Series / Regression).
load_commodities  : Ingests the commodities dataset in the Vertica DB.
    (Time Series / Regression).
load_market       : Ingests the market dataset in the Vertica DB.
	(Basic Data Exploration).
load_smart_meters : Ingests the smart meters dataset in the Vertica DB.
	(Time Series / Regression).
load_titanic      : Ingests the titanic dataset in the Vertica DB.
	(Classification).
load_winequality  : Ingests the winequality dataset in the Vertica DB.
	(Regression / Classification).
	"""
    return load_dataset(
        cursor,
        schema,
        name,
        '"SepalLengthCm" Numeric(5,2), "SepalWidthCm" Numeric(5,2), "PetalLengthCm" Numeric(5,2), "PetalWidthCm" Numeric(5,2), "Species" Varchar(30)',
        '"Id" FILLER Integer, "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"',
        "iris",
    )


# ---#
def load_market(cursor=None, schema: str = "public", name: str = "market"):
    """
---------------------------------------------------------------------------
Ingests the market dataset in the Vertica DB (Dataset ideal for easy 
exploration). If a table with the same name and schema already exists, 
this function will create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
	Vertica DB cursor. 
schema: str, optional
	Schema of the new relation. The default schema is public.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the market vDataFrame.

See Also
--------
load_amazon       : Ingests the amazon dataset in the Vertica DB.
	(Time Series / Regression).
load_commodities  : Ingests the commodities dataset in the Vertica DB.
    (Time Series / Regression).
load_iris         : Ingests the iris dataset in the Vertica DB.
	(Clustering / Classification).
load_smart_meters : Ingests the smart meters dataset in the Vertica DB.
	(Time Series / Regression).
load_titanic      : Ingests the titanic dataset in the Vertica DB.
	(Classification).
load_winequality  : Ingests the winequality dataset in the Vertica DB.
	(Regression / Classification).
	"""
    return load_dataset(
        cursor,
        schema,
        name,
        '"Name" Varchar(32), "Form" Varchar(32), "Price" Float',
        '"Form", "Name", "Price"',
        "market",
    )


# ---#
def load_smart_meters(cursor=None, schema: str = "public", name: str = "smart_meters"):
    """
---------------------------------------------------------------------------
Ingests the smart meters dataset in the Vertica DB (Dataset ideal for TS
and Regression). If a table with the same name and schema already exists, 
this function will create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
	Vertica DB cursor. 
schema: str, optional
	Schema of the new relation. The default schema is public.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the smart meters vDataFrame.

See Also
--------
load_amazon       : Ingests the amazon dataset in the Vertica DB.
	(Time Series / Regression).
load_commodities  : Ingests the commodities dataset in the Vertica DB.
    (Time Series / Regression).
load_iris         : Ingests the iris dataset in the Vertica DB.
	(Clustering / Classification).
load_market       : Ingests the market dataset in the Vertica DB.
	(Basic Data Exploration).
load_titanic      : Ingests the titanic dataset in the Vertica DB.
	(Classification).
load_winequality  : Ingests the winequality dataset in the Vertica DB.
	(Regression / Classification).
	"""
    return load_dataset(
        cursor,
        schema,
        name,
        '"time" Timestamp, "val" Numeric(11,7), "id" Integer',
        '"time", "val", "id"',
        "smart_meters",
    )


# ---#
def load_titanic(cursor=None, schema: str = "public", name: str = "titanic"):
    """
---------------------------------------------------------------------------
Ingests the titanic dataset in the Vertica DB (Dataset ideal for 
Classification). If a table with the same name and schema already exists, 
this function will create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
	Vertica DB cursor. 
schema: str, optional
	Schema of the new relation. The default schema is public.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the titanic vDataFrame.

See Also
--------
load_amazon       : Ingests the amazon dataset in the Vertica DB.
	(Time Series / Regression).
load_commodities  : Ingests the commodities dataset in the Vertica DB.
    (Time Series / Regression).
load_iris         : Ingests the iris dataset in the Vertica DB.
	(Clustering / Classification).
load_market       : Ingests the market dataset in the Vertica DB.
	(Basic Data Exploration).
load_smart_meters : Ingests the smart meters dataset in the Vertica DB.
	(Time Series / Regression).
load_winequality  : Ingests the winequality dataset in the Vertica DB.
	(Regression / Classification).
	"""
    return load_dataset(
        cursor,
        schema,
        name,
        '"pclass" Integer, "survived" Integer, "name" Varchar(164), "sex" Varchar(20), "age" Numeric(6,3), "sibsp" Integer, "parch" Integer, "ticket" Varchar(36), "fare" Numeric(10,5), "cabin" Varchar(30), "embarked" Varchar(20), "boat" Varchar(100), "body" Integer, "home.dest" Varchar(100)',
        '"pclass", "survived", "name", "sex", "age", "sibsp", "parch", "ticket", "fare", "cabin", "embarked", "boat", "body", "home.dest"',
        "titanic",
    )


# ---#
def load_winequality(cursor=None, schema: str = "public", name: str = "winequality"):
    """
---------------------------------------------------------------------------
Ingests the winequality dataset in the Vertica DB (Dataset ideal for Regression
and Classification). If a table with the same name and schema already exists, 
this function will create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
	Vertica DB cursor. 
schema: str, optional
	Schema of the new relation. The default schema is public.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the winequality vDataFrame.

See Also
--------
load_amazon       : Ingests the amazon dataset in the Vertica DB.
	(Time Series / Regression).
load_commodities  : Ingests the commodities dataset in the Vertica DB.
    (Time Series / Regression).
load_iris         : Ingests the iris dataset in the Vertica DB.
	(Clustering / Classification).
load_market       : Ingests the market dataset in the Vertica DB.
	(Basic Data Exploration).
load_smart_meters : Ingests the smart meters dataset in the Vertica DB.
	(Time Series / Regression).
load_titanic      : Ingests the titanic dataset in the Vertica DB.
	(Classification).
	"""
    return load_dataset(
        cursor,
        schema,
        name,
        '"fixed_acidity" Numeric(6,3), "volatile_acidity" Numeric(7,4), "citric_acid" Numeric(6,3), "residual_sugar" Numeric(7,3), "chlorides" Float, "free_sulfur_dioxide" Numeric(7,2), "total_sulfur_dioxide" Numeric(7,2), "density" Float, "pH" Numeric(6,3), "sulphates" Numeric(6,3), "alcohol" Float, "quality" Integer, "good" Integer, "color" Varchar(20)',
        '"fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "quality", "good", "color"',
        "winequality",
    )


# ---#
def load_world(cursor=None, schema: str = "public", name: str = "world"):
    """
---------------------------------------------------------------------------
Ingests the World dataset in the Vertica DB (Dataset ideal for Geospatial). 
If a table with the same name and schema already exists, this function will 
create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
    Vertica DB cursor. 
schema: str, optional
    Schema of the new relation. The default schema is public.
name: str, optional
    Name of the new relation.

Returns
-------
vDataFrame
    the World vDataFrame.

See Also
--------
load_cities : Ingests the cities dataset in the Vertica DB (Geospatial).
    """
    return load_dataset(
        cursor,
        schema,
        name,
        '"pop_est" Int, "continent" Varchar(32), "country" Varchar(82), "geometry" Geometry',
        '"pop_est", "continent", "country", gx FILLER LONG VARCHAR(65000), "geometry" AS ST_GeomFromText(gx)',
        "world",
    )
