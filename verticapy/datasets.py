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
import os, datetime

# VerticaPy Modules
import verticapy
from verticapy import vDataFrame, vdf_from_relation
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.errors import *

# ---#
def gen_dataset(features_ranges: dict, cursor=None, nrows: int = 1000,):
    """
---------------------------------------------------------------------------
Generates a dataset using the input parameters.

Parameters
----------
features_ranges: dict,
    Dictionary including the features types and ranges.
        For str      : The subdictionary must include two keys: 'type' must
                       be set to 'str' and 'value' must include the feature
                       categories.
        For int      : The subdictionary must include two keys: 'type' must
                       be set to 'int' and 'range' must include two integers
                       that represent the lower and the upper bound.
        For float    : The subdictionary must include two keys: 'type' must
                       be set to 'float' and 'range' must include two floats
                       that represent the lower and the upper bound.
        For date     : The subdictionary must include two keys: 'type' must
                       be set to 'date' and 'range' must include the start
                       date and the number of days after.
        For datetime : The subdictionary must include two keys: 'type' must
                       be set to 'date' and 'range' must include the start
                       date and the number of days after.
cursor: DBcursor, optional
    Vertica database cursor.
nrows: int, optional
    The maximum number of rows in the dataset.

Returns
-------
vDataFrame
    Generated dataset.
    """    
    check_types([("features_ranges", features_ranges, [dict],), 
                 ("nrows", nrows, [int],),])
    cursor = check_cursor(cursor)[0]
    sql = []
    for param in features_ranges:
        if features_ranges[param]["type"] == str:
            val = features_ranges[param]["values"]
            if isinstance(val, str):
                sql += [f"'{val}' AS \"{param}\""]
            else:
                n = len(val)
                val = ", ".join(["'" + str(elem) + "'" for elem in val])
                sql += [f"(ARRAY[{val}])[RANDOMINT({n})] AS \"{param}\""]
        elif features_ranges[param]["type"] == float:
            val = features_ranges[param]["range"]
            lower, upper = val[0], val[1]
            sql += [f"({lower} + RANDOM() * ({upper} - {lower}))::FLOAT AS \"{param}\""]
        elif features_ranges[param]["type"] == int:
            val = features_ranges[param]["range"]
            lower, upper = val[0], val[1]
            sql += [f"({lower} + RANDOM() * ({upper} - {lower}))::INT AS \"{param}\""]
        elif features_ranges[param]["type"] == datetime.date:
            val = features_ranges[param]["range"]
            start_date, number_of_days = val[0], val[1]
            sql += [f"('{start_date}'::DATE + RANDOMINT({number_of_days})) AS \"{param}\""]
        elif features_ranges[param]["type"] == datetime.datetime:
            val = features_ranges[param]["range"]
            start_date, number_of_days = val[0], val[1]
            sql += [f"('{start_date}'::TIMESTAMP + {number_of_days} * RANDOM()) AS \"{param}\""]
        elif features_ranges[param]["type"] == bool:
            sql += [f"RANDOMINT(2)::BOOL AS \"{param}\""]
        else:
            ptype = features_ranges[param]["type"]
            raise ParameterError(f"Parameter {param}: Type {ptype} is not supported.")
    sql = "(SELECT " + ", ".join(sql) + f"FROM (SELECT tm FROM (SELECT '03-11-1993'::TIMESTAMP + INTERVAL '1 second' AS t UNION ALL SELECT '03-11-1993'::TIMESTAMP + INTERVAL '{nrows} seconds' AS t) x TIMESERIES tm AS '1 second' OVER(ORDER BY t)) y) z"
    return vdf_from_relation(sql, cursor=cursor)


# ---#
def gen_meshgrid(features_ranges: dict, cursor=None,):
    """
---------------------------------------------------------------------------
Generates a dataset using regular steps.

Parameters
----------
features_ranges: dict,
    Dictionary including the features types and ranges.
        For str      : The subdictionary must include two keys: 'type' must
                       be set to 'str' and 'value' must include the feature
                       categories.
        For int      : The subdictionary must include two keys: 'type' must
                       be set to 'int' and 'range' must include two integers
                       that represent the lower and the upper bound.
        For float    : The subdictionary must include two keys: 'type' must
                       be set to 'float' and 'range' must include two floats
                       that represent the lower and the upper bound.
        For date     : The subdictionary must include two keys: 'type' must
                       be set to 'date' and 'range' must include the start
                       date and the number of days after.
        For datetime : The subdictionary must include two keys: 'type' must
                       be set to 'date' and 'range' must include the start
                       date and the number of days after.
        Numerical and date-like features must have an extra key in the dictionary 
        named 'nbins' corresponding to the number of bins used to compute the 
        different categories.
cursor: DBcursor, optional
    Vertica database cursor.

Returns
-------
vDataFrame
    generated dataset.
    """    
    check_types([("features_ranges", features_ranges, [dict],),])
    cursor = check_cursor(cursor)[0]
    sql = []
    for idx, param in enumerate(features_ranges):
        if "nbins" in features_ranges[param]:
            bins = features_ranges[param]["nbins"]
        else:
            bins = 100
        if features_ranges[param]["type"] == str:
            val = features_ranges[param]["values"]
            if isinstance(val, str):
                val = [val]
            val = " UNION ALL ".join([f"(SELECT '{elem}' AS \"{param}\")" for elem in val])
            sql += [f"({val}) x{idx}"]
        elif features_ranges[param]["type"] == float:
            val = features_ranges[param]["range"]
            lower, upper = val[0], val[1]
            ts_table = f"(SELECT DAY(tm - '03-11-1993'::TIMESTAMP) AS tm FROM (SELECT '03-11-1993'::TIMESTAMP AS t UNION ALL SELECT '03-11-1993'::TIMESTAMP + INTERVAL '{bins} days' AS t) x TIMESERIES tm AS '1 day' OVER(ORDER BY t)) y"
            h = (upper - lower) / bins
            sql += [f"(SELECT ({lower} + {h} * tm)::FLOAT AS \"{param}\" FROM {ts_table}) x{idx}"]
        elif features_ranges[param]["type"] == int:
            val = features_ranges[param]["range"]
            lower, upper = val[0], val[1]
            ts_table = f"(SELECT DAY(tm - '03-11-1993'::TIMESTAMP) AS tm FROM (SELECT '03-11-1993'::TIMESTAMP AS t UNION ALL SELECT '03-11-1993'::TIMESTAMP + INTERVAL '{bins} days' AS t) x TIMESERIES tm AS '1 day' OVER(ORDER BY t)) y"
            h = (upper - lower) / bins
            sql += [f"(SELECT ({lower} + {h} * tm)::INT AS \"{param}\" FROM {ts_table}) x{idx}"]
        elif features_ranges[param]["type"] == datetime.date:
            val = features_ranges[param]["range"]
            start_date, number_of_days = val[0], val[1]
            ts_table = f"(SELECT DAY(tm - '03-11-1993'::TIMESTAMP) AS tm FROM (SELECT '03-11-1993'::TIMESTAMP AS t UNION ALL SELECT '03-11-1993'::TIMESTAMP + INTERVAL '{bins} days' AS t) x TIMESERIES tm AS '1 day' OVER(ORDER BY t)) y"
            h = number_of_days / bins
            sql += [f"(SELECT ('{start_date}'::DATE + {h} * tm)::DATE AS \"{param}\" FROM {ts_table}) x{idx}"]
        elif features_ranges[param]["type"] == datetime.datetime:
            val = features_ranges[param]["range"]
            start_date, number_of_days = val[0], val[1]
            ts_table = f"(SELECT DAY(tm - '03-11-1993'::TIMESTAMP) AS tm FROM (SELECT '03-11-1993'::TIMESTAMP AS t UNION ALL SELECT '03-11-1993'::TIMESTAMP + INTERVAL '{bins} days' AS t) x TIMESERIES tm AS '1 day' OVER(ORDER BY t)) y"
            h = number_of_days / bins
            sql += [f"(SELECT ('{start_date}'::DATE + {h} * tm)::TIMESTAMP AS \"{param}\" FROM {ts_table}) x{idx}"]
        elif features_ranges[param]["type"] == bool:
            sql += [f"((SELECT False AS \"{param}\") UNION ALL (SELECT True AS \"{param}\")) x{idx}"]
        else:
            ptype = features_ranges[param]["type"]
            raise ParameterError(f"Parameter {param}: Type {ptype} is not supported.")
    sql = "(SELECT * FROM {}) x".format(" CROSS JOIN ".join(sql))
    return vdf_from_relation(sql, cursor=cursor)


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
            path = os.path.dirname(verticapy.__file__) + "/data/{}.csv".format(
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
def load_airline_passengers(
    cursor=None, schema: str = "public", name: str = "airline_passengers"
):
    """
---------------------------------------------------------------------------
Ingests the airline passengers dataset into the Vertica database. This dataset
is ideal for time series and regression models. If a table with the same name
and schema already exists, this function will create a vDataFrame from the 
input relation.

Parameters
----------
cursor: DBcursor, optional
    Vertica database cursor. 
schema: str, optional
    Schema of the new relation. The default schema is public.
name: str, optional
    Name of the new relation.

Returns
-------
vDataFrame
    the airline passengers vDataFrame.
    """
    return load_dataset(
        cursor,
        schema,
        name,
        '"date" Date, "passengers" Integer',
        '"date", "passengers"',
        "airline_passengers",
    )


# ---#
def load_amazon(cursor=None, schema: str = "public", name: str = "amazon"):
    """
---------------------------------------------------------------------------
Ingests the amazon dataset into the Vertica database. This dataset is ideal
for time series and regression models. If a table with the same name and schema
already exists, this function will create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
	Vertica database cursor. 
schema: str, optional
	Schema of the new relation. The default schema is public.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the amazon vDataFrame.
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
Ingests the Cities dataset into the Vertica database. This dataset is ideal
for geospatial models. If a table with the same name and schema already exists,
this function will create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
    Vertica database cursor. 
schema: str, optional
    Schema of the new relation. The default schema is public.
name: str, optional
    Name of the new relation.

Returns
-------
vDataFrame
    the Cities vDataFrame.
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
Ingests the commodities dataset into the Vertica database. This dataset is
ideal for time series and regression models. If a table with the same name 
and schema already exists, this function will create a vDataFrame from the 
input relation.

Parameters
----------
cursor: DBcursor, optional
    Vertica database cursor. 
schema: str, optional
    Schema of the new relation. The default schema is public.
name: str, optional
    Name of the new relation.

Returns
-------
vDataFrame
    the amazon vDataFrame.
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
def load_gapminder(
    cursor=None, schema: str = "public", name: str = "gapminder"
):
    """
---------------------------------------------------------------------------
Ingests the gapminder dataset into the Vertica database. This dataset is ideal
for time series and regression models. If a table with the same name and schema
already exists, this function will create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
    Vertica database cursor. 
schema: str, optional
    Schema of the new relation. The default schema is public.
name: str, optional
    Name of the new relation.

Returns
-------
vDataFrame
    the gapminder vDataFrame.
    """
    return load_dataset(
        cursor,
        schema,
        name,
        '"country" Varchar(96), "year" Integer, "pop" Integer, "continent" Varchar(52), "lifeExp" Float, "gdpPercap" Float',
        '"country", "year", "pop", "continent", "lifeExp", "gdpPercap"',
        "gapminder",
    )


# ---#
def load_iris(cursor=None, schema: str = "public", name: str = "iris"):
    """
---------------------------------------------------------------------------
Ingests the iris dataset into the Vertica database. This dataset is ideal for
classification and clustering models. If a table with the same name and schema
already exists, this function will create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
	Vertica database cursor. 
schema: str, optional
	Schema of the new relation. The default schema is public.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the iris vDataFrame.
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
Ingests the market dataset into the Vertica database. This dataset is ideal
for data exploration. If a table with the same name and schema already exists,
this function will create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
	Vertica database cursor. 
schema: str, optional
	Schema of the new relation. The default schema is public.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the market vDataFrame.
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
def load_pop_growth(
    cursor=None, schema: str = "public", name: str = "pop_growth"
):
    """
---------------------------------------------------------------------------
Ingests the population growth dataset into the Vertica database. This dataset
is ideal for time series and geospatial models. If a table with the same name
and schema already exists, this function will create a vDataFrame from the 
input relation.

Parameters
----------
cursor: DBcursor, optional
    Vertica database cursor. 
schema: str, optional
    Schema of the new relation. The default schema is public.
name: str, optional
    Name of the new relation.

Returns
-------
vDataFrame
    the pop growth vDataFrame.
    """
    return load_dataset(
        cursor,
        schema,
        name,
        '"year" Int, "continent" Varchar(100), "country" Varchar(100), "city" Varchar(100), "population" Float, "lat" Float, "lon" Float',
        '"year", "continent", "country", "city", "population", "lat", "lon"',
        "pop_growth",
    )


# ---#
def load_smart_meters(cursor=None, schema: str = "public", name: str = "smart_meters"):
    """
---------------------------------------------------------------------------
Ingests the smart meters dataset into the Vertica database. This dataset is ideal
for time series and regression models. If a table with the same name and schema
already exists, this function will create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
	Vertica database cursor. 
schema: str, optional
	Schema of the new relation. The default schema is public.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the smart meters vDataFrame.
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
Ingests the titanic dataset into the Vertica database. This dataset is ideal 
for classification models. If a table with the same name and schema already 
exists, this function will create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
	Vertica database cursor. 
schema: str, optional
	Schema of the new relation. The default schema is public.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the titanic vDataFrame.
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
Ingests the winequality dataset into the Vertica database. This dataset is ideal
for regression and classification models. If a table with the same name and schema
already exists, this function will create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
	Vertica database cursor. 
schema: str, optional
	Schema of the new relation. The default schema is public.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the winequality vDataFrame.
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
Ingests the World dataset into the Vertica database. This dataset is ideal for
ideal for geospatial models. If a table with the same name and schema already
exists, this function will create a vDataFrame from the input relation.

Parameters
----------
cursor: DBcursor, optional
    Vertica database cursor. 
schema: str, optional
    Schema of the new relation. The default schema is public.
name: str, optional
    Name of the new relation.

Returns
-------
vDataFrame
    the World vDataFrame.
    """
    return load_dataset(
        cursor,
        schema,
        name,
        '"pop_est" Int, "continent" Varchar(32), "country" Varchar(82), "geometry" Geometry',
        '"pop_est", "continent", "country", gx FILLER LONG VARCHAR(65000), "geometry" AS ST_GeomFromText(gx)',
        "world",
    )
