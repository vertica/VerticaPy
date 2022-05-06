# (c) Copyright [2018-2022] Micro Focus or one of its affiliates.
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
# VerticaPy is a Python library with scikit-like functionality for conducting
# data science projects on data stored in Vertica, taking advantage Vertica’s
# speed and built-in analytics and machine learning features. It supports the
# entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize
# data transformation operations, and offers beautiful graphical options.
#
# VerticaPy aims to do all of the above. The idea is simple: instead of moving
# data around for processing, VerticaPy brings the logic to the data.
#
#
# Modules
#
# Standard Python Modules
import os, datetime

# VerticaPy Modules
import verticapy, vertica_python
from verticapy import vDataFrame
from verticapy.connect import current_cursor
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.errors import *

# ---#
def gen_dataset(features_ranges: dict, nrows: int = 1000):
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
nrows: int, optional
    The maximum number of rows in the dataset.

Returns
-------
vDataFrame
    Generated dataset.
    """
    version(condition=[9, 3, 0])
    check_types([("features_ranges", features_ranges, [dict]), ("nrows", nrows, [int])])

    sql = []

    for param in features_ranges:

        if features_ranges[param]["type"] == str:

            val = features_ranges[param]["values"]
            if isinstance(val, str):
                sql += [f"'{val}' AS \"{param}\""]
            else:
                n = len(val)
                val = ", ".join(["'" + str(elem) + "'" for elem in val])
                sql += [f'(ARRAY[{val}])[RANDOMINT({n})] AS "{param}"']

        elif features_ranges[param]["type"] == float:

            val = features_ranges[param]["range"]
            lower, upper = val[0], val[1]
            sql += [
                f"({lower} + RANDOM() * ({upper} - {lower}))::FLOAT " f'AS "{param}"'
            ]

        elif features_ranges[param]["type"] == int:

            val = features_ranges[param]["range"]
            lower, upper = val[0], val[1]
            sql += [f"({lower} + RANDOM() * ({upper} - {lower}))::INT " f'AS "{param}"']

        elif features_ranges[param]["type"] == datetime.date:

            val = features_ranges[param]["range"]
            start_date, number_of_days = val[0], val[1]
            sql += [
                f"('{start_date}'::DATE + RANDOMINT({number_of_days})) " f'AS "{param}"'
            ]

        elif features_ranges[param]["type"] == datetime.datetime:

            val = features_ranges[param]["range"]
            start_date, number_of_days = val[0], val[1]
            sql += [
                f"('{start_date}'::TIMESTAMP + {number_of_days} "
                f'* RANDOM()) AS "{param}"'
            ]

        elif features_ranges[param]["type"] == bool:

            sql += [f'RANDOMINT(2)::BOOL AS "{param}"']

        else:

            ptype = features_ranges[param]["type"]
            raise ParameterError(f"Parameter {param}: Type {ptype}" "is not supported.")

    sql = ", ".join(sql)
    sql = (
        f"(SELECT {sql} FROM (SELECT tm FROM (SELECT '03-11-1993'"
        "::TIMESTAMP + INTERVAL '1 second' AS t UNION ALL SELECT"
        f" '03-11-1993'::TIMESTAMP + INTERVAL '{nrows} seconds' AS"
        " t) x TIMESERIES tm AS '1 second' OVER(ORDER BY t)) y) z"
    )

    return vDataFrameSQL(sql)


# ---#
def gen_meshgrid(features_ranges: dict):
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
        Numerical and date-like features must have an extra key in the 
        dictionary named 'nbins' corresponding to the number of bins used to 
        compute the different categories.

Returns
-------
vDataFrame
    generated dataset.
    """
    check_types([("features_ranges", features_ranges, [dict])])

    sql = []

    for idx, param in enumerate(features_ranges):

        nbins = 100
        if "nbins" in features_ranges[param]:
            nbins = features_ranges[param]["nbins"]
        ts_table = (
            f"(SELECT DAY(tm - '03-11-1993'::TIMESTAMP) AS tm FROM "
            "(SELECT '03-11-1993'::TIMESTAMP AS t UNION ALL SELECT"
            f" '03-11-1993'::TIMESTAMP + INTERVAL '{nbins} days' AS t)"
            " x TIMESERIES tm AS '1 day' OVER(ORDER BY t)) y"
        )

        if features_ranges[param]["type"] == str:
            val = features_ranges[param]["values"]
            if isinstance(val, str):
                val = [val]
            val = " UNION ALL ".join(
                [
                    f"""(SELECT '{elem}' 
                                          AS \"{param}\")"""
                    for elem in val
                ]
            )
            sql += [f"({val}) x{idx}"]

        elif features_ranges[param]["type"] == float:
            val = features_ranges[param]["range"]
            lower, upper = val[0], val[1]
            h = (upper - lower) / nbins
            sql += [
                f'(SELECT ({lower} + {h} * tm)::FLOAT AS "{param}" '
                f"FROM {ts_table}) x{idx}"
            ]

        elif features_ranges[param]["type"] == int:
            val = features_ranges[param]["range"]
            lower, upper = val[0], val[1]
            h = (upper - lower) / nbins
            sql += [
                f'(SELECT ({lower} + {h} * tm)::INT AS "{param}" '
                f"FROM {ts_table}) x{idx}"
            ]

        elif features_ranges[param]["type"] == datetime.date:
            val = features_ranges[param]["range"]
            start_date, number_of_days = val[0], val[1]
            h = number_of_days / nbins
            sql += [
                f"(SELECT ('{start_date}'::DATE + {h} * tm)::DATE"
                f' AS "{param}" FROM {ts_table}) x{idx}'
            ]

        elif features_ranges[param]["type"] == datetime.datetime:
            val = features_ranges[param]["range"]
            start_date, number_of_days = val[0], val[1]
            h = number_of_days / nbins
            sql += [
                f"(SELECT ('{start_date}'::DATE + {h} * tm)::TIMESTAMP "
                f'AS "{param}" FROM {ts_table}) x{idx}'
            ]

        elif features_ranges[param]["type"] == bool:
            sql += [
                f'((SELECT False AS "{param}") UNION ALL '
                f'(SELECT True AS "{param}")) x{idx}'
            ]

        else:
            ptype = features_ranges[param]["type"]
            raise ParameterError(
                f"Parameter {param}: Type {ptype} " "is not supported."
            )

    sql = "(SELECT * FROM {0}) x".format(" CROSS JOIN ".join(sql))

    return vDataFrameSQL(sql)


# ---#
def load_dataset(
    schema: str, name: str, dtype: dict, copy_cols: list = [], dataset_name: str = ""
):
    """
    General Function to ingest a dataset
    """
    check_types([("schema", schema, [str]), ("name", name, [str])])

    try:

        vdf = vDataFrame(name, schema=schema)

    except:

        name = quote_ident(name)
        schema = "v_temp_schema" if not (schema) else quote_ident(schema)
        create_table(table_name=name, dtype=dtype, schema=schema)

        try:

            path = os.path.dirname(verticapy.__file__)
            path += f"/data/{dataset_name}.csv"
            if not (copy_cols):
                copy_cols = [quote_ident(col) for col in dtype]
            copy_cols = ", ".join(copy_cols)
            query = (
                "COPY {0}.{1}({2}) FROM {3} DELIMITER ',' NULL '' "
                "ENCLOSED BY '\"' ESCAPE AS '\\' SKIP 1;"
            ).format(schema, name, copy_cols, "{}")

            cur = current_cursor()

            if isinstance(cur, vertica_python.vertica.cursor.Cursor):

                query = query.format("STDIN")
                executeSQL(query, title="Ingesting the data.", method="copy", path=path)

            else:

                query = query.format("LOCAL '{0}'".format(path))
                executeSQL(query, title="Ingesting the data.")

            executeSQL("COMMIT;", title="Commit.")
            vdf = vDataFrame(name, schema=schema)

        except:

            drop(schema + "." + name, method="table")
            raise

    return vdf


#
#
# ---#
def load_airline_passengers(schema: str = "public", name: str = "airline_passengers"):
    """
---------------------------------------------------------------------------
Ingests the airline passengers dataset into the Vertica database. 
This dataset is ideal for time series and regression models. If a table 
with the same name and schema already exists, this function will create 
a vDataFrame from the input relation.

Parameters
----------
schema: str, optional
    Schema of the new relation. If empty, a temporary local table will be
    created.
name: str, optional
    Name of the new relation.

Returns
-------
vDataFrame
    the airline passengers vDataFrame.
    """
    return load_dataset(
        schema,
        name,
        {"date": "Date", "passengers": "Integer"},
        dataset_name="airline_passengers",
    )


# ---#
def load_amazon(schema: str = "public", name: str = "amazon"):
    """
---------------------------------------------------------------------------
Ingests the amazon dataset into the Vertica database. This dataset is ideal
for time series and regression models. If a table with the same name and 
schema already exists, this function will create a vDataFrame from the 
input relation.

Parameters
---------- 
schema: str, optional
	Schema of the new relation. If empty, a temporary local table will be
    created.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the amazon vDataFrame.
	"""
    return load_dataset(
        schema,
        name,
        {"date": "Date", "state": "Varchar(32)", "number": "Integer"},
        dataset_name="amazon",
    )


# ---#
def load_cities(schema: str = "public", name: str = "cities"):
    """
---------------------------------------------------------------------------
Ingests the Cities dataset into the Vertica database. This dataset is ideal
for geospatial models. If a table with the same name and schema already 
exists, this function will create a vDataFrame from the input relation.

Parameters
----------
schema: str, optional
    Schema of the new relation. If empty, a temporary local table will be
    created.
name: str, optional
    Name of the new relation.

Returns
-------
vDataFrame
    the Cities vDataFrame.
    """
    return load_dataset(
        schema,
        name,
        {"city": "Varchar(82)", "geometry": "Geometry"},
        ["city", "gx FILLER LONG VARCHAR(65000)", "geometry AS ST_GeomFromText(gx)"],
        dataset_name="cities",
    )


# ---#
def load_commodities(schema: str = "public", name: str = "commodities"):
    """
---------------------------------------------------------------------------
Ingests the commodities dataset into the Vertica database. This dataset is
ideal for time series and regression models. If a table with the same name 
and schema already exists, this function will create a vDataFrame from the 
input relation.

Parameters
----------
schema: str, optional
    Schema of the new relation. If empty, a temporary local table will be
    created.
name: str, optional
    Name of the new relation.

Returns
-------
vDataFrame
    the amazon vDataFrame.
    """
    return load_dataset(
        schema,
        name,
        {
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


# ---#
def load_gapminder(schema: str = "public", name: str = "gapminder"):
    """
---------------------------------------------------------------------------
Ingests the gapminder dataset into the Vertica database. This dataset is 
ideal for time series and regression models. If a table with the same name 
and schema already exists, this function will create a vDataFrame from the 
input relation.

Parameters
---------- 
schema: str, optional
    Schema of the new relation. If empty, a temporary local table will be
    created.
name: str, optional
    Name of the new relation.

Returns
-------
vDataFrame
    the gapminder vDataFrame.
    """
    return load_dataset(
        schema,
        name,
        {
            "country": "Varchar(96)",
            "year": "Integer",
            "pop": "Integer",
            "continent": "Varchar(52)",
            "lifeExp": "Float",
            "gdpPercap": "Float",
        },
        dataset_name="gapminder",
    )


# ---#
def load_iris(schema: str = "public", name: str = "iris"):
    """
---------------------------------------------------------------------------
Ingests the iris dataset into the Vertica database. This dataset is ideal 
for classification and clustering models. If a table with the same name and 
schema already exists, this function will create a vDataFrame from the input 
relation.

Parameters
----------
schema: str, optional
	Schema of the new relation. If empty, a temporary local table will be
    created.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the iris vDataFrame.
	"""
    return load_dataset(
        schema,
        name,
        {
            "SepalLengthCm": "Numeric(5,2)",
            "SepalWidthCm": "Numeric(5,2)",
            "PetalLengthCm": "Numeric(5,2)",
            "PetalWidthCm": "Numeric(5,2)",
            "Species": "Varchar(30)",
        },
        [
            "Id FILLER Integer",
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm",
            "Species",
        ],
        dataset_name="iris",
    )


# ---#
def load_market(schema: str = "public", name: str = "market"):
    """
---------------------------------------------------------------------------
Ingests the market dataset into the Vertica database. This dataset is ideal
for data exploration. If a table with the same name and schema already 
exists, this function will create a vDataFrame from the input relation.

Parameters
----------
schema: str, optional
	Schema of the new relation. If empty, a temporary local table will be
    created.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the market vDataFrame.
	"""
    return load_dataset(
        schema,
        name,
        {"Form": "Varchar(32)", "Name": "Varchar(32)", "Price": "Float"},
        dataset_name="market",
    )


# ---#
def load_pop_growth(schema: str = "public", name: str = "pop_growth"):
    """
---------------------------------------------------------------------------
Ingests the population growth dataset into the Vertica database. This 
dataset is ideal for time series and geospatial models. If a table with 
the same name and schema already exists, this function will create a 
vDataFrame from the input relation.

Parameters
----------
schema: str, optional
    Schema of the new relation. If empty, a temporary local table will be
    created.
name: str, optional
    Name of the new relation.

Returns
-------
vDataFrame
    the pop growth vDataFrame.
    """
    return load_dataset(
        schema,
        name,
        {
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


# ---#
def load_smart_meters(schema: str = "public", name: str = "smart_meters"):
    """
---------------------------------------------------------------------------
Ingests the smart meters dataset into the Vertica database. This dataset is 
ideal for time series and regression models. If a table with the same name 
and schema already exists, this function will create a vDataFrame from the 
input relation.

Parameters
----------
schema: str, optional
	Schema of the new relation. If empty, a temporary local table will be
    created.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the smart meters vDataFrame.
	"""
    return load_dataset(
        schema,
        name,
        {"time": "Timestamp", "val": "Numeric(11,7)", "id": "Integer"},
        dataset_name="smart_meters",
    )


# ---#
def load_titanic(schema: str = "public", name: str = "titanic"):
    """
---------------------------------------------------------------------------
Ingests the titanic dataset into the Vertica database. This dataset is 
ideal for classification models. If a table with the same name and schema 
already exists, this function will create a vDataFrame from the input 
relation.

Parameters
----------
schema: str, optional
	Schema of the new relation. If empty, a temporary local table will be
    created.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the titanic vDataFrame.
	"""
    return load_dataset(
        schema,
        name,
        {
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


# ---#
def load_winequality(schema: str = "public", name: str = "winequality"):
    """
---------------------------------------------------------------------------
Ingests the winequality dataset into the Vertica database. This dataset is 
ideal for regression and classification models. If a table with the same 
name and schema already exists, this function will create a vDataFrame from 
the input relation.

Parameters
----------
schema: str, optional
	Schema of the new relation. If empty, a temporary local table will be
    created.
name: str, optional
	Name of the new relation.

Returns
-------
vDataFrame
	the winequality vDataFrame.
	"""
    return load_dataset(
        schema,
        name,
        {
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


# ---#
def load_world(schema: str = "public", name: str = "world"):
    """
---------------------------------------------------------------------------
Ingests the World dataset into the Vertica database. This dataset is ideal 
for ideal for geospatial models. If a table with the same name and schema 
already exists, this function will create a vDataFrame from the input 
relation.

Parameters
----------
schema: str, optional
    Schema of the new relation. If empty, a temporary local table will be
    created.
name: str, optional
    Name of the new relation.

Returns
-------
vDataFrame
    the World vDataFrame.
    """
    return load_dataset(
        schema,
        name,
        {
            "pop_est": "Int",
            "continent": "Varchar(32)",
            "country": "Varchar(82)",
            "geometry": "Geometry",
        },
        [
            "pop_est",
            "continent",
            "country",
            "gx FILLER LONG VARCHAR(65000)",
            "geometry AS ST_GeomFromText(gx)",
        ],
        dataset_name="world",
    )


#
# Datasets used in the tests
#
# ---#
def load_dataset_cl(table_name: str = "dataset_cl", schema: str = "public"):
    # Classification Dataset

    data = [
        [1, "Bus", "Male", 0, "Cheap", "Low"],
        [2, "Bus", "Male", 1, "Cheap", "Med"],
        [3, "Train", "Female", 1, "Cheap", "Med"],
        [4, "Bus", "Female", 0, "Cheap", "Low"],
        [5, "Bus", "Male", 1, "Cheap", "Med"],
        [6, "Train", "Male", 0, "Standard", "Med"],
        [7, "Train", "Female", 1, "Standard", "Med"],
        [8, "Car", "Female", 1, "Expensive", "Hig"],
        [9, "Car", "Male", 2, "Expensive", "Med"],
        [10, "Car", "Female", 2, "Expensive", "Hig"],
    ]
    input_relation = "{}.{}".format(quote_ident(schema), quote_ident(table_name))

    drop(name=input_relation, method="table")
    create_table(
        table_name=table_name,
        schema=schema,
        dtype={
            "Id": "INT",
            "transportation": "VARCHAR",
            "gender": "VARCHAR",
            "owned cars": "INT",
            "cost": "VARCHAR",
            "income": "CHAR(4)",
        },
    )
    insert_into(table_name=table_name, schema=schema, data=data, copy=False)

    return vDataFrame(input_relation=input_relation)


# ---#
def load_dataset_reg(table_name: str = "dataset_reg", schema: str = "public"):
    # Regression Dataset

    data = [
        [1, 0, "Male", 0, "Cheap", "Low"],
        [2, 0, "Male", 1, "Cheap", "Med"],
        [3, 1, "Female", 1, "Cheap", "Med"],
        [4, 0, "Female", 0, "Cheap", "Low"],
        [5, 0, "Male", 1, "Cheap", "Med"],
        [6, 1, "Male", 0, "Standard", "Med"],
        [7, 1, "Female", 1, "Standard", "Med"],
        [8, 2, "Female", 1, "Expensive", "Hig"],
        [9, 2, "Male", 2, "Expensive", "Med"],
        [10, 2, "Female", 2, "Expensive", "Hig"],
    ]
    input_relation = "{}.{}".format(quote_ident(schema), quote_ident(table_name))

    drop(name=input_relation, method="table")
    create_table(
        table_name=table_name,
        schema=schema,
        dtype={
            "Id": "INT",
            "transportation": "INT",
            "gender": "VARCHAR",
            "owned cars": "INT",
            "cost": "VARCHAR",
            "income": "CHAR(4)",
        },
    )
    insert_into(table_name=table_name, schema=schema, data=data, copy=False)

    return vDataFrame(input_relation=input_relation)


# ---#
def load_dataset_num(table_name: str = "dataset_num", schema: str = "public"):
    # Numerical Dataset

    data = [
        [1, 7.2, 3.6, 6.1, 2.5],
        [2, 7.7, 2.8, 6.7, 2.0],
        [3, 7.7, 3.0, 6.1, 2.3],
        [4, 7.9, 3.8, 6.4, 2.0],
        [5, 4.4, 2.9, 1.4, 0.2],
        [6, 4.6, 3.6, 1.0, 0.2],
        [7, 4.7, 3.2, 1.6, 0.2],
        [8, 6.5, 2.8, 4.6, 1.5],
        [9, 6.8, 2.8, 4.8, 1.4],
        [10, 7.0, 3.2, 4.7, 1.4],
    ]
    input_relation = "{}.{}".format(quote_ident(schema), quote_ident(table_name))

    drop(name=input_relation, method="table")
    create_table(
        table_name=table_name,
        schema=schema,
        dtype={
            "Id": "INT",
            "col1": "FLOAT",
            "col2": "FLOAT",
            "col3": "FLOAT",
            "col4": "FLOAT",
        },
    )
    insert_into(table_name=table_name, schema=schema, data=data, copy=False)

    return vDataFrame(input_relation=input_relation)
