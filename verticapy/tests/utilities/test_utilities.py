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

# Pytest
import pytest, warnings

# Other Modules
import pandas as pd
import os

# VerticaPy
import vertica_python, verticapy
from verticapy.core.vdataframe.base import vDataFrame
from verticapy.connection import current_cursor
from verticapy.utilities import *
from verticapy.datasets import (
    load_cities,
    load_titanic,
    load_world,
    load_iris,
    load_laliga,
)
from verticapy.geo import *
from verticapy.learn.neighbors import KNeighborsClassifier
from verticapy.learn.linear_model import LinearRegression
from verticapy._config.config import set_option
from verticapy.connection.global_connection import get_global_connection

set_option("print_info", False)


@pytest.fixture(scope="module")
def cities_vd():
    cities = load_cities()
    yield cities
    drop(name="public.cities")


@pytest.fixture(scope="module")
def laliga_vd():
    laliga = load_laliga()
    yield laliga
    drop(name="public.laliga")


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(name="public.titanic")


@pytest.fixture(scope="module")
def world_vd():
    cities = load_world()
    yield cities
    drop(name="public.world")


class TestUtilities:
    @pytest.mark.skip(reason="this test will be valid for Vertica v12.0.2")
    def test_complex_elements(self, laliga_vd):
        vdf = laliga_vd.copy()
        vdf["away_team_managers"] = vdf["away_team"]["managers"]
        vdf["home_team_managers"] = vdf["home_team"]["managers"]
        vdf["away_team_managers"]._transf[-1] = (
            '"away_team"."managers"',
            "Array",
            "complex",
        )  # doing it manually because the vertica-python client does not support cdt yet
        vdf["home_team_managers"]._transf[-1] = (
            '"home_team"."managers"',
            "Array",
            "complex",
        )
        # testing isarray
        assert vdf["away_team_managers"].isarray()
        assert vdf["home_team_managers"].isarray()
        # testing concatenation
        assert (
            vdf["away_team_managers"] + vdf["home_team_managers"]
            == 'ARRAY_CAT("away_team_managers", "home_team_managers")'
        )
        vdf["all_managers"] = vdf["away_team_managers"] + vdf["home_team_managers"]
        # testing get_item and set_item
        vdf["id_test"] = vdf["away_team_managers"][0]["Country"]["id"]
        assert vdf["id_test"].avg() == pytest.approx(181.429078014184)
        assert vdf["away_team_managers"][0:1][0:1][0:1][0]["id"].avg() == pytest.approx(
            589.698581560284
        )
        # testing apply_fun - count
        vdf["all_managers"]._transf[-1] = (
            'ARRAY_CAT("away_team_managers", "home_team_managers")',
            "Array",
            "complex",
        )
        assert vdf["all_managers"].apply_fun(func="length")
        assert vdf["all_managers"].max() == 2.0
        # testing apply_fun - min
        vdf2 = TableSample({"x": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}).to_vdf()
        vdf2["x"].apply_fun(func="min")
        assert vdf2["x"].sum() == 12
        # testing apply_fun - max
        vdf2 = TableSample({"x": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}).to_vdf()
        vdf2["x"].apply_fun(func="max")
        assert vdf2["x"].sum() == 18
        # testing apply_fun - avg
        vdf2 = TableSample({"x": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}).to_vdf()
        vdf2["x"].apply_fun(func="avg")
        assert vdf2["x"].sum() == 15
        # testing apply_fun - sum
        vdf2 = TableSample({"x": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}).to_vdf()
        vdf2["x"].apply_fun(func="sum")
        assert vdf2["x"].sum() == 45
        # testing apply_fun - contain
        vdf2 = TableSample({"x": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}).to_vdf()
        vdf2["x"].apply_fun(func="contain", x=1)
        assert vdf2["x"].sum() == 1
        # testing apply_fun - len
        vdf2 = TableSample({"x": [[1, 2, 3], [4, 5, 6], [7]]}).to_vdf()
        vdf2["x"]._transf[-1] = ('"x"', "Array", "complex")
        vdf2["x"].apply_fun(func="len")
        assert vdf2["x"].sum() == 7
        # testing apply_fun - find
        vdf2 = TableSample({"x": [[1, 2, 3], [2, 5, 6], [7, 8, 9]]}).to_vdf()
        vdf2["x"].apply_fun(func="find", x=2)
        assert vdf2["x"].sum() == 0
        # testing string to array
        vdf2 = TableSample(
            {"x": ["[1, -2, 3]", "[2,    5,    6]", "[7,   8]"]}
        ).to_vdf()
        vdf2["x"].astype("array")
        vdf2["x"].apply_fun(func="len")
        assert vdf2["x"].sum() == 8
        # TableSample with ROW and arrays
        vdf2 = TableSample(
            {
                "x": [[1, 2, 3], [4, 5, 6], [7]],
                "y": [{"a": 1, "b": 2}, {"a": 2, "b": 3}, {"a": 4, "b": 5}],
            }
        ).to_vdf()
        assert vdf2["y"]["a"].sum() == 7
        vdf2["x"]._transf[-1] = ('"x"', "Array", "complex")
        vdf2["x"].apply_fun(func="len")
        assert vdf2["x"].sum() == 7.0
        # Complex to JSON
        vdf2 = TableSample(
            {
                "x": [[1, 2, 3], [4, 5, 6], [7]],
                "y": [{"a": 1, "b": 2}, {"a": 2, "b": 3}, {"a": 4, "b": 5}],
            }
        ).to_vdf()
        vdf2["x"]._transf[-1] = ('"x"', "Array", "complex")
        vdf2["x"].astype("json")
        vdf2["y"].astype("json")
        assert vdf2["x"]._transf[-1][0] == "TO_JSON({})"
        assert vdf2["y"]._transf[-1][0] == "TO_JSON({})"
        # Test get_len
        vdf2 = TableSample({"x": [[1, 2, 3], [2, 5, 6], [7, 8, 9]]}).to_vdf()
        vdf2["x"]._transf[-1] = ('"x"', "Array", "complex")
        assert vdf2["x"].get_len().sum() == 9

    def test_create_schema_table(self):
        drop("verticapy_test_create_schema", method="schema")
        create_schema("verticapy_test_create_schema")
        create_table(
            table_name="test",
            dtype={"col0": "INT", "col1": "FLOAT"},
            schema="verticapy_test_create_schema",
        )
        current_cursor().execute(
            """SELECT 
                    table_name 
               FROM columns 
               WHERE table_schema = 'verticapy_test_create_schema' 
               GROUP BY 1 ORDER BY 1;"""
        )
        result = current_cursor().fetchone()[0]
        assert result == "test"

    def test_drop(self, world_vd):
        current_cursor().execute("DROP TABLE IF EXISTS public.drop_data")
        current_cursor().execute(
            'CREATE TABLE IF NOT EXISTS public.drop_data(Id identity(2000) primary key, transportation VARCHAR, gender VARCHAR, "owned cars" INT, cost VARCHAR, income CHAR(4)) ORDER BY id SEGMENTED BY HASH(id) ALL NODES KSAFE;'
        )
        current_cursor().execute(
            "INSERT INTO drop_data(transportation, gender, \"owned cars\", cost, income) VALUES ('Bus', 'Male', 0, 'Cheap', 'Low')"
        )
        current_cursor().execute(
            "INSERT INTO drop_data(transportation, gender, \"owned cars\", cost, income) VALUES ('Bus', 'Male', 1, 'Cheap', 'Med')"
        )
        current_cursor().execute(
            "INSERT INTO drop_data(transportation, gender, \"owned cars\", cost, income) VALUES ('Train', 'Female', 1, 'Cheap', 'Med')"
        )
        current_cursor().execute(
            "INSERT INTO drop_data(transportation, gender, \"owned cars\", cost, income) VALUES ('Bus', 'Female', 0, 'Cheap', 'Low')"
        )
        current_cursor().execute(
            "INSERT INTO drop_data(transportation, gender, \"owned cars\", cost, income) VALUES ('Bus', 'Male', 1, 'Cheap', 'Med')"
        )
        current_cursor().execute(
            "INSERT INTO drop_data(transportation, gender, \"owned cars\", cost, income) VALUES ('Train', 'Male', 0, 'Standard', 'Med')"
        )
        current_cursor().execute(
            "INSERT INTO drop_data(transportation, gender, \"owned cars\", cost, income) VALUES ('Train', 'Female', 1, 'Standard', 'Med')"
        )
        current_cursor().execute(
            "INSERT INTO drop_data(transportation, gender, \"owned cars\", cost, income) VALUES ('Car', 'Female', 1, 'Expensive', 'Hig')"
        )
        current_cursor().execute(
            "INSERT INTO drop_data(transportation, gender, \"owned cars\", cost, income) VALUES ('Car', 'Male', 2, 'Expensive', 'Med')"
        )
        current_cursor().execute(
            "INSERT INTO drop_data(transportation, gender, \"owned cars\", cost, income) VALUES ('Car', 'Female', 2, 'Expensive', 'Hig')"
        )
        current_cursor().execute("COMMIT")
        # table
        current_cursor().execute("DROP TABLE IF EXISTS public.verticapy_table_test")
        current_cursor().execute("CREATE TABLE verticapy_table_test AS SELECT 1;")
        drop("verticapy_table_test")
        current_cursor().execute(
            "SELECT table_name FROM columns WHERE table_name = 'verticapy_table_test' GROUP BY 1;"
        )
        result = current_cursor().fetchall()
        assert result == []
        # view
        current_cursor().execute("DROP VIEW IF EXISTS public.verticapy_view_test")
        current_cursor().execute("CREATE VIEW verticapy_view_test AS SELECT 1;")
        drop("verticapy_view_test")
        current_cursor().execute(
            "SELECT table_name FROM view_columns WHERE table_name = 'verticapy_view_test' GROUP BY 1;"
        )
        result = current_cursor().fetchall()
        assert result == []
        # text index
        current_cursor().execute(
            "CREATE TEXT INDEX drop_index ON drop_data (id, transportation);"
        )
        drop("drop_index")
        with pytest.raises(vertica_python.errors.MissingRelation):
            current_cursor().execute("SELECT * FROM drop_index;")
        # model
        current_cursor().execute("DROP MODEL IF EXISTS public.verticapy_model_test")
        current_cursor().execute(
            "SELECT NAIVE_BAYES('public.verticapy_model_test', 'public.drop_data', 'transportation', 'gender, cost');"
        )
        drop("verticapy_model_test")
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'verticapy_model_test' GROUP BY 1;"
        )
        result = current_cursor().fetchall()
        assert result == []
        # geo index
        world_copy = world_vd.copy()
        world_copy["id"] = "ROW_NUMBER() OVER (ORDER BY pop_est)"
        result = create_index(world_copy, "id", "geometry", "world_polygons", True)
        drop("world_polygons")
        with pytest.raises(vertica_python.errors.QueryError):
            describe_index("world_polygons", True)
        drop()

    def test_flex_elements(self):
        # Creating a Flex Table
        current_cursor().execute(
            "CREATE FLEX LOCAL TEMP TABLE utilities_flex_test(x int) ON COMMIT PRESERVE ROWS;"
        )
        path = os.path.dirname(verticapy.__file__) + "/datasets/data/laliga/*.json"
        current_cursor().execute(
            f"COPY utilities_flex_test FROM LOCAL '{path}' PARSER FJSONPARSER();"
        )
        # Testing isflextable
        assert isflextable(table_name="utilities_flex_test", schema="v_temp_schema")
        # Testing compute_flextable_keys
        keys = compute_flextable_keys(
            "v_temp_schema.utilities_flex_test",
            usecols=["referee.country.id", "stadium.id", "metadata.data_version"],
        )
        assert keys == [
            ["referee.country.id", "Integer"],
            ["stadium.id", "Integer"],
            ["metadata.data_version", "Date"],
        ]
        # Testing compute_vmap_keys
        home_managers_keys = compute_vmap_keys(
            expr="v_temp_schema.utilities_flex_test",
            vmap_col="home_team.managers",
            limit=2,
        )
        assert len(home_managers_keys) == 2
        assert home_managers_keys[0][1] == home_managers_keys[1][1] == 282
        # Testing vDataFrame from Flextable
        vdf = vDataFrame("v_temp_schema.utilities_flex_test")
        # Testing vDataFrame.get_len
        vdf["away_team.managers"].get_len().avg() == pytest.approx(3.36725663716814)
        vdf["stadium.name"].get_len().avg() == pytest.approx(14.5809523809524)
        # Testing vDataFrame[].isvmap
        assert vdf["away_team.managers"].isvmap()
        # Testing astype: VMAP to str
        vdf["away_team.managers2"] = vdf["away_team.managers"]
        vdf["away_team.managers"].astype(str)
        assert vdf["away_team.managers"].category() == "text"
        assert (
            vdf["away_team.managers"]._transf[-1][0]
            == "MAPTOSTRING({} USING PARAMETERS canonical_json=false)::varchar"
        )
        vdf["away_team.managers2"].astype("json")
        assert vdf["away_team.managers2"].category() == "text"
        assert (
            vdf["away_team.managers2"]._transf[-1][0]
            == "MAPTOSTRING({} USING PARAMETERS canonical_json=true)"
        )
        # Testing vDataFrame.__set__ using VMAP sub category
        vdf["home_team.managers.0.country.id"] = vdf["home_team.managers"][
            "0.country.id"
        ]
        assert (
            vdf["home_team.managers.0.country.id"]._transf[-1][0]
            == "MAPLOOKUP(\"home_team.managers\", '0.country.id')"
        )
        # Materialising the flex table - TODO
        # vdf_table = vdf.to_db(name = "utilities_table_test", relation_type = "local")
        # read_json to get a vDataFrame with maps
        vdf_table = read_json(path)
        # Testing vDataFrame.flat_vmap
        vdf_table_flat = vdf_table.flat_vmap(
            exclude_columns=["away_score", "home_score"]
        )
        all_flat_count = vdf_table_flat.count_percent()
        assert len(all_flat_count["count"]) == 52
        assert all_flat_count["count"][-10] == 105.0
        assert all_flat_count["count"][-15] == 282.0
        assert '"away_team.managers.0.country.name"' in all_flat_count["index"]
        assert '"away_team.managers.0.country.id"' in all_flat_count["index"]
        assert '"away_team.managers.0.nickname"' in all_flat_count["index"]
        drop("v_temp_schema.utilities_flex_test")

    def test_get_data_types(self):
        result = get_data_types(
            "SELECT 1 AS col1, 'abc' AS col2, '5 minutes'::interval AS col3"
        )
        assert result == [
            ["col1", "Integer"],
            ["col2", "Varchar(3)"],
            ["col3", "Interval Day to Second"],
        ]

    def test_insert_into(self):
        # using copy
        iris = load_iris()
        result = insert_into(
            table_name="iris",
            schema="public",
            column_names=iris.get_columns(),
            data=iris.to_list(),
        )
        drop(name="public.iris", method="table")
        assert result == 150
        # using multiple inserts
        iris = load_iris()
        # generating the SQL code
        result = insert_into(
            table_name="iris",
            schema="public",
            column_names=iris.get_columns(),
            data=iris.to_list(),
            copy=False,
            genSQL=True,
        )
        assert len(result) == 150
        for elem in result:
            assert elem[0:27] == 'INSERT INTO "public"."iris"'
        # executing multiple inserts
        result = insert_into(
            table_name="iris",
            schema="public",
            column_names=iris.get_columns(),
            data=iris.to_list(),
            copy=False,
        )
        drop(name="public.iris", method="table")
        assert result == 150

    def test_read_pandas(self, titanic_vd):
        df = titanic_vd.to_pandas()
        drop("titanic_pandas")
        vdf = read_pandas(df=df, name="titanic_pandas")
        assert vdf.shape() == (1234, 14)
        drop("titanic_pandas")
        vdf = read_pandas(df=df)
        assert vdf.shape() == (1234, 14)
        drop("test_df")
        read_pandas(df, name="test_df", schema="public")
        read_pandas(df, name="test_df", schema="public", insert=True)
        vdf = read_pandas(df, name="test_df", schema="public", insert=True)
        assert vdf.shape() == (3702, 14)
        drop("test_df")
        # Problem with '\'
        # d = {"col1": [1, 2, 3, 4], "col2": ["red", 'gre"en', "b\lue", 'p\i""nk']}
        # df = pd.DataFrame(data=d)
        # vdf = read_pandas(df)
        # assert vdf.shape() == (4, 2)

    def test_pcsv(self):
        result = pcsv(
            os.path.dirname(verticapy.__file__) + "/datasets/data/titanic.csv"
        )
        assert result == {
            "age": "Numeric(6,3)",
            "boat": "Varchar(100)",
            "body": "Integer",
            "cabin": "Varchar(30)",
            "embarked": "Varchar(20)",
            "fare": "Numeric(10,5)",
            "home.dest": "Varchar(100)",
            "name": "Varchar(164)",
            "parch": "Integer",
            "pclass": "Integer",
            "sex": "Varchar(20)",
            "sibsp": "Integer",
            "survived": "Integer",
            "ticket": "Varchar(36)",
        }

    def test_pjson(self):
        result = pjson(
            os.path.dirname(verticapy.__file__)
            + "/tests/utilities/titanic-passengers.json",
        )
        assert result == {
            "datasetid": "Varchar(36)",
            "fields.age": "Float",
            "fields.cabin": "Varchar(30)",
            "fields.embarked": "Varchar(20)",
            "fields.fare": "Float",
            "fields.name": "Varchar(164)",
            "fields.parch": "Integer",
            "fields.passengerid": "Integer",
            "fields.pclass": "Integer",
            "fields.sex": "Varchar(20)",
            "fields.sibsp": "Integer",
            "fields.survived": "Boolean",
            "fields.ticket": "Varchar(36)",
            "record_timestamp": "Timestamp",
            "recordid": "Varchar(80)",
        }

    def test_read_avro(self):
        drop("public.variants", method="table")
        path = os.path.dirname(verticapy.__file__) + "/tests/utilities/variants.avro"
        result = read_avro(
            path,
            table_name="variants",
            schema="public",
        )
        assert result.shape() == (731, 34)
        assert result["end"].avg() == pytest.approx(16074719.005472)
        drop("public.variants", method="table")

    def test_read_json(self, laliga_vd):
        drop("public.titanic_verticapy_test_json", method="table")
        path = os.path.dirname(verticapy.__file__) + "/tests/utilities/"
        result = read_json(
            path + "titanic-passengers.json",
            table_name="titanic_verticapy_test_json",
            schema="public",
        )
        assert result.shape() == (891, 15)
        assert drop("public.titanic_verticapy_test_json", method="table")
        result = read_json(
            path + "titanic-passengers.json",
            table_name="titanic_verticapy_test_json",
        )
        assert result.shape() == (891, 15)
        assert drop("v_temp_schema.titanic_verticapy_test_json", method="table")
        result = read_json(
            path + "json_many/*.json",
            table_name="titanic_verticapy_test_json",
        )
        assert result.shape() == (1782, 15)
        assert drop("v_temp_schema.titanic_verticapy_test_json", method="table")

        """
        # doing an ingest_local = False does not work yet
        
        # TODO test with ingest_local = False

        # use complex dt
        laliga_vd.to_json("/home/dbadmin/laliga/", n_files=5, order_by="match_id")
        path = "/home/dbadmin/laliga/*.json"
        drop("public.laliga_verticapy_test_json", method="table")
        vdf = read_json(
            path,
            table_name="laliga_verticapy_test_json",
            schema="public",
            ingest_local=False,
            use_complex_dt=True,
        )
        assert vdf.shape() == (452, 14)
        """

        # Trying SQL
        path = os.path.dirname(verticapy.__file__) + "/datasets/data/laliga/*.json"
        drop("public.laliga_verticapy_test_json", method="table")
        queries = read_json(
            path,
            table_name="laliga_verticapy_test_json",
            schema="public",
            genSQL=True,
            ingest_local=True,
            use_complex_dt=False,
        )
        for query in queries:
            current_cursor().execute(
                query.replace("tmp_flex", "tmp_flex_test_read_json")
            )
        vdf = vDataFrame("public.laliga_verticapy_test_json")
        assert vdf.shape() == (452, 40)
        assert vdf["away_score"].ctype().lower()[0:3] == "int"
        assert vdf["away_team.away_team_id"].ctype().lower()[0:3] == "int"
        assert vdf["match_status"].ctype().lower() == "varchar(20)"
        assert vdf["away_team.away_team_gender"].ctype().lower() == "varchar(20)"
        assert not (
            isflextable(table_name="laliga_verticapy_test_json", schema="public")
        )

        """
        -- TO DO, tests on insert! - it seems to not work well
        # testing insert
        vdf = read_json(
            path,
            table_name="laliga_verticapy_test_json",
            schema="public",
            insert=True,
            ingest_local=False,
            use_complex_dt=False,
        )
        assert vdf.shape() == (904, 40)
        """
        # testing temporary table
        drop("public.laliga_verticapy_test_json", method="table")
        vdf = read_json(
            path,
            table_name="laliga_verticapy_test_json",
            schema="public",
            temporary_table=True,
            ingest_local=True,
            use_complex_dt=False,
        )
        assert drop(
            "public.laliga_verticapy_test_json",
            method="table",
        )

        # testing local temporary table
        vdf = read_json(
            path,
            table_name="laliga_verticapy_test_json2",
            temporary_local_table=True,
            ingest_local=True,
            use_complex_dt=False,
        )
        assert drop(
            "v_temp_schema.laliga_verticapy_test_json2",
            method="table",
        )

        # Checking flextables and materialize option
        path = os.path.dirname(verticapy.__file__) + "/tests/utilities/"
        drop("public.titanic_verticapy_test_json")
        result = read_json(
            path + "titanic-passengers.json",
            table_name="titanic_verticapy_test_json",
            schema="public",
            ingest_local=True,
            materialize=False,
        )
        assert isflextable(table_name="titanic_verticapy_test_json", schema="public")

        # Checking materialize, storing to database, and re-conversion to a vdataframe
        drop("public.titanic_verticapy_test_json_2")
        result.to_db("public.titanic_verticapy_test_json_2")
        result2 = vDataFrame("public.titanic_verticapy_test_json_2")
        assert result2["fields.cabin"].dtype() == result["fields.cabin"].dtype()
        assert result2["fields.age"].dtype() == result["fields.age"].dtype()
        assert result2["datasetid"].dtype() == result["datasetid"].dtype()
        assert result2["fields.fare"].dtype() == result["fields.fare"].dtype()
        assert (
            result2["fields.parch"].dtype()[0:3] == result["fields.parch"].dtype()[0:3]
        )
        assert (
            result2["fields.pclass"].dtype()[0:3]
            == result["fields.pclass"].dtype()[0:3]
        )
        assert drop("public.titanic_verticapy_test_json")
        drop("public.titanic_verticapy_test_json_2")

    def test_read_csv(self):
        path = os.path.dirname(verticapy.__file__) + "/datasets/data/titanic.csv"
        # with schema
        drop("public.titanic_verticapy_test_csv", method="table")
        result = read_csv(
            path, table_name="titanic_verticapy_test_csv", schema="public"
        )
        assert result.shape() == (1234, 14)
        drop("public.titanic_verticapy_test_csv", method="table")
        # temporary table
        result = read_csv(
            path,
            table_name="titanic_verticapy_test_csv",
            schema="public",
            temporary_table=True,
        )
        assert result.shape() == (1234, 14)
        drop("public.titanic_verticapy_test_csv", method="table")
        # parse_nrows
        result = read_csv(
            path,
            table_name="titanic_verticapy_test_csv",
            schema="public",
            parse_nrows=100,
        )
        assert result.shape() == (1234, 14)
        # insert
        result = read_csv(
            path,
            table_name="titanic_verticapy_test_csv",
            schema="public",
            insert=True,
        )
        assert result.shape() == (2468, 14)
        drop("public.titanic_verticapy_test_csv", method="table")
        # temporary local table
        result = read_csv(path, table_name="titanic_verticapy_test_csv")
        assert result.shape() == (1234, 14)
        drop("v_temp_schema.titanic_verticapy_test_csv", method="table")
        # with header names
        result = read_csv(
            path,
            table_name="titanic_verticapy_test_csv",
            header_names=["ucol{}".format(i) for i in range(14)],
        )
        assert result.shape() == (1234, 14)
        assert result.get_columns() == ['"ucol{}"'.format(i) for i in range(14)]
        drop("v_temp_schema.titanic_verticapy_test_csv", method="table")
        # with dtypes
        dtype = {
            "pclass": "int",
            "survived": "bool",
            "name": "varchar",
            "sex": "varchar",
            "age": "float",
            "sibsp": "int",
            "parch": "int",
            "ticket": "varchar",
            "fare": "float",
            "cabin": "varchar",
            "embarked": "varchar",
            "boat": "varchar",
            "body": "varchar",
            "home.dest": "varchar",
        }
        result = read_csv(
            path,
            table_name="titanic_verticapy_test_csv",
            dtype=dtype,
        )
        assert result.shape() == (1234, 14)
        drop("v_temp_schema.titanic_verticapy_test_csv", method="table")
        # genSQL
        result = read_csv(
            path,
            schema="public",
            table_name="titanic_verticapy_test_csv",
            genSQL=True,
        )
        assert result[0][0:50] == 'CREATE TABLE "public"."titanic_verticapy_test_csv"'
        assert result[1][0:42] == 'COPY "public"."titanic_verticapy_test_csv"'
        # Multiple files
        path = os.path.dirname(verticapy.__file__) + "/tests/utilities/"
        result = read_csv(
            path + "csv_many/*.csv",
            table_name="titanic_verticapy_test_csv",
            dtype=dtype,
        )
        assert result.shape() == (1870, 14)
        drop("v_temp_schema.titanic_verticapy_test_csv", method="table")

        # Checking Flextable
        path = os.path.dirname(verticapy.__file__) + "/datasets/data/"
        drop("public.titanic_verticapy_test_csv")
        result = read_csv(
            path=path + "titanic.csv",
            table_name="titanic_verticapy_test_csv",
            materialize=False,
            ingest_local=True,
            schema="public",
        )
        assert isflextable(table_name="titanic_verticapy_test_csv", schema="public")

        # Checking materialize, storing to database, and re-conversion to a vdataframe
        drop("public.titanic_verticapy_test_csv_2")
        result.to_db('"public"."titanic_verticapy_test_csv_2"')
        result2 = vDataFrame("public.titanic_verticapy_test_csv_2")
        assert result2["ticket"].dtype() == result["ticket"].dtype()
        assert result2["survived"].dtype()[0:3] == result["survived"].dtype()[0:3]
        assert result2["sibsp"].dtype()[0:3] == result["sibsp"].dtype()[0:3]
        assert result2["pclass"].dtype()[0:3] == result["pclass"].dtype()[0:3]
        assert result2["home.dest"].dtype() == result["home.dest"].dtype()

        # with compression
        path = os.path.dirname(verticapy.__file__) + "/tests/utilities/titanic.csv.gz"
        drop("public.titanic_verticapy_test_csv_gz")
        result3 = read_csv(
            path,
            table_name="titanic_verticapy_test_csv_gz",
            ingest_local=True,
            schema="public",
            header_names=[col[1:-1] for col in result2.get_columns()],
        )
        assert result3.shape() == (1234, 14)

        # auto identification of the separator
        path = os.path.dirname(verticapy.__file__) + "/tests/utilities/"
        for i in range(1, 4):
            drop(f"public.csv_test{i}")
            result3 = read_csv(
                path + f"csv_test{i}.csv",
                table_name=f"csv_test{i}",
                schema="public",
            )
            assert result3.shape() == (4, 4)
            assert result3.get_columns() == ['"a"', '"b"', '"c"', '"d"']
            assert result3["a"].avg() == 1.0
            drop(f"public.csv_test{i}")

    @pytest.mark.skip(reason="can not read files locally.")
    def test_read_file(self, laliga_vd):
        laliga_vd.to_json("/home/dbadmin/laliga/", n_files=5, order_by="match_id")
        path = "/home/dbadmin/laliga/*.json"
        drop(name="v_temp_schema.laliga_test")
        vdf = read_file(
            path=path,
            schema="",
            table_name="laliga_test",
            dtype={"away_score": "float", "away_team_id": "float"},
            unknown="varchar",
            varchar_varbinary_length=200,
            max_files=20,
        )
        assert laliga_vd.shape() == vdf.shape()
        assert vdf["away_score"].ctype().lower()[0:5] == "float"
        assert vdf["away_team"]["away_team_id"].ctype().lower()[0:5] == "float"
        assert vdf["match_status"].ctype().lower() == "varchar(200)"
        assert drop(name="v_temp_schema.laliga_test")
        # with genSQL = True
        sql = read_file(
            path=path,
            schema="",
            table_name="laliga_test",
            dtype={"away_score": "float", "away_team_id": "float"},
            unknown="varchar",
            varchar_varbinary_length=200,
            max_files=20,
            genSQL=True,
        )
        for query in sql:
            current_cursor().execute(query)
        vdf = vDataFrame("v_temp_schema.laliga_test")
        assert laliga_vd.shape() == vdf.shape()
        assert vdf["away_score"].ctype().lower() == "float"
        assert vdf["away_team"]["away_team_id"].ctype().lower()[0:5] == "float"
        assert vdf["match_status"].ctype().lower() == "varchar(200)"
        drop(name="v_temp_schema.laliga_test", method="table")
        # without any table name
        vdf = read_file(
            path=path,
            dtype={"away_score": "float", "away_team_id": "float"},
            unknown="varchar",
            varchar_varbinary_length=200,
            max_files=20,
        )
        assert laliga_vd.shape() == vdf.shape()

        # testing insert
        vdf = read_file(path)
        vdf = read_file(
            path,
            table_name=vdf._vars["main_relation"],
            insert=True,
        )
        assert vdf.shape() == (904, 14)

    def test_read_shp(self, cities_vd):
        drop(name="public.cities_test")
        cities_vd.to_shp("cities_test", "/home/dbadmin/", shape="Point")
        vdf = read_shp("/home/dbadmin/cities_test.shp")
        assert vdf.shape() == (202, 3)
        try:
            os.remove("/home/dbadmin/cities_test.shp")
            os.remove("/home/dbadmin/cities_test.shx")
            os.remove("/home/dbadmin/cities_test.dbf")
        except:
            pass
        drop(name="public.cities_test")

    def test_save_to_query_profile(self):
        model = LinearRegression(
            "model_test",
        )
        iris = load_iris()
        q = save_to_query_profile(
            name="test",
            path="test_path.test_value",
            json_dict={
                "X0": 1103,
                "X1": None,
                "X2": True,
                "X3": False,
                "X4": [
                    "x0",
                    "x1",
                    "x2",
                    "x3",
                ],
                "X5": {
                    "Y0": 3,
                    1: ["y0", "y1"],
                    None: 4,
                },
                "vdf": iris,
                "model": model,
            },
            query_label="verticapy_test_utilities_json",
            return_query=True,
            add_identifier=False,
        )
        assert (
            q
            == 'SELECT /*+LABEL(\'verticapy_test_utilities_json\')*/ \'{"verticapy_fname": "test", "verticapy_fpath": "test_path.test_value", "X0": 1103, "X1": null, "X2": true, "X3": false, "X4": "x0;x1;x2;x3", "X5": {"Y0": 3, "1": "y0;y1", "None": 4}, "vdf": "\\"public\\".\\"iris\\"", "model": "LinearRegression"}\''
        )
        # with identifier
        q2 = save_to_query_profile(
            name="test",
            path="test_path.test_value",
            json_dict={
                "X0": 1103,
                "X1": None,
                "X2": True,
                "X3": False,
                "X4": [
                    "x0",
                    "x1",
                    "x2",
                    "x3",
                ],
                "X5": {
                    "Y0": 3,
                    1: ["y0", "y1"],
                    None: 4,
                },
                "vdf": iris,
                "model": model,
            },
            query_label="verticapy_test_utilities_json",
            return_query=True,
            add_identifier=True,
        )
        gb_conn = get_global_connection()
        assert (
            q2
            == 'SELECT /*+LABEL(\'verticapy_test_utilities_json\')*/ \'{"verticapy_fname": "test", "verticapy_fpath": "test_path.test_value", "verticapy_id": "'
            + str(gb_conn.vpy_session_identifier)
            + '", "X0": 1103, "X1": null, "X2": true, "X3": false, "X4": "x0;x1;x2;x3", "X5": {"Y0": 3, "1": "y0;y1", "None": 4}, "vdf": "\\"public\\".\\"iris\\"", "model": "LinearRegression"}\''
        )
        current_cursor().execute(
            "SELECT MAPKEYS(MAPJSONEXTRACTOR(SUBSTRING('{0}', 53, 241))) OVER ();".format(
                q.replace("'", "''")
            )
        )
        all_keys = current_cursor().fetchall()
        all_keys = [elem[0] for elem in all_keys]
        assert "model" in all_keys
        assert "vdf" in all_keys
        assert "verticapy_fname" in all_keys
        assert "verticapy_fpath" in all_keys
        assert "X0" in all_keys
        assert "X1" in all_keys
        assert "X2" in all_keys
        assert "X3" in all_keys
        assert "X4" in all_keys
        assert "X5.1" in all_keys
        assert "X5.None" in all_keys
        assert "X5.Y0" in all_keys
        save_to_query_profile(
            name="test",
            path="test_path.test_value",
            json_dict={
                "X0": 1103,
                "X1": None,
                "X2": True,
                "X3": False,
                "X4": [
                    "x0",
                    "x1",
                    "x2",
                    "x3",
                ],
                "X5": {
                    "Y0": 3,
                    1: ["y0", "y1"],
                    None: 4,
                },
                "vdf": iris,
                "model": model,
            },
            query_label="verticapy_test_utilities_json",
            return_query=False,
            add_identifier=False,
        )
        current_cursor().execute(
            "SELECT query FROM v_monitor.query_profiles WHERE identifier = 'verticapy_test_utilities_json' LIMIT 1;"
        )
        q3 = current_cursor().fetchone()[0]
        assert q == q3

    def test_TableSample(self):
        result = TableSample(
            {"index": ["Apple", "Banana", "Orange"], "price": [1, 2, 3]}
        )
        assert result["index"] == ["Apple", "Banana", "Orange"]
        assert result["price"] == [1, 2, 3]
        result2 = result.transpose()
        assert result2["Apple"] == [1]
        assert result2["Banana"] == [2]
        assert result2["Orange"] == [3]
        result3 = result.to_list()
        assert result3 == [[1], [2], [3]]
        result4 = result.to_numpy()
        assert result4[0] == [1]
        assert result4[1] == [2]
        assert result4[2] == [3]
        result5 = result.to_pandas()["price"].mean()
        assert result5 == 2.0
        result6 = result.to_sql()
        assert (
            result6
            == '(SELECT \'Apple\' AS "index", 1 AS "price") UNION ALL (SELECT \'Banana\' AS "index", 2 AS "price") UNION ALL (SELECT \'Orange\' AS "index", 3 AS "price")'
        )
        result7 = result.to_vdf()["price"].mean()
        assert result7 == 2.0

    def test_tablesample_read_sql(self):
        result = TableSample.read_sql('SELECT 1 AS "verticapy test *+""";')
        assert result['verticapy test *+"'] == [1]

    def test_vDataFrame_sql(self):
        result = vDataFrame(
            '(SELECT 1 AS "verticapy test *+") x',
        )
        assert result["verticapy test *+"].avg() == 1.0

    @pytest.mark.skip(reason="this test will be implemented later")
    def test_set_option(self):
        pass

    def test_vertica_version(self):
        result = vertica_version()
        assert result[0] < 99
        try:
            vertica_version([99, 1, 1])
            fail = False
        except:
            fail = True
        assert fail
