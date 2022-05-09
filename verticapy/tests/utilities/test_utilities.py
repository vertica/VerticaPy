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

# Pytest
import pytest

# Other Modules
import pandas as pd

# VerticaPy
import vertica_python
from verticapy.connect import current_cursor
from verticapy.utilities import *
from verticapy.datasets import load_cities, load_titanic, load_world, load_iris
from verticapy.geo import *
from verticapy.learn.neighbors import KNeighborsClassifier

set_option("print_info", False)


@pytest.fixture(scope="module")
def cities_vd():
    cities = load_cities()
    yield cities
    drop(name="public.cities")


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
        drop("verticapy", method="schema")

    def test_create_verticapy_schema(self):
        drop("verticapy", method="schema")
        create_verticapy_schema()
        current_cursor().execute(
            """SELECT 
                    table_name 
               FROM columns 
               WHERE table_schema = 'verticapy' 
               GROUP BY 1 ORDER BY 1;"""
        )
        result = [elem[0] for elem in current_cursor().fetchall()]
        assert result == ["attr", "models"]
        drop("verticapy", method="schema")

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
        # verticapy model
        with warnings.catch_warnings(record=True) as w:
            drop("verticapy", method="schema")
        create_verticapy_schema()
        model = KNeighborsClassifier("verticapy_model_test")
        model.fit("public.drop_data", ["gender", "cost"], "transportation")
        drop("verticapy_model_test")
        current_cursor().execute(
            "SELECT model_name FROM verticapy.models WHERE model_name = 'verticapy_model_test' GROUP BY 1;"
        )
        result = current_cursor().fetchall()
        assert result == []
        drop("verticapy", method="schema")
        # geo index
        world_copy = world_vd.copy()
        world_copy["id"] = "ROW_NUMBER() OVER (ORDER BY pop_est)"
        result = create_index(world_copy, "id", "geometry", "world_polygons", True)
        drop("world_polygons")
        with pytest.raises(vertica_python.errors.QueryError):
            describe_index("world_polygons", True)
        drop()

    def test_readSQL(self):
        result = readSQL('SELECT 1 AS "verticapy test *+""";')
        assert result['verticapy test *+"'] == [1]

    def test_get_data_types(self):
        result = get_data_types(
            "SELECT 1 AS col1, 'abc' AS col2, '5 minutes'::interval AS col3"
        )
        assert result == [
            ["col1", "Integer"],
            ["col2", "Varchar(3)"],
            ["col3", "Interval"],
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

    def test_pandas_to_vertica(self, titanic_vd):
        df = titanic_vd.to_pandas()
        drop("titanic_pandas")
        vdf = pandas_to_vertica(df=df, name="titanic_pandas")
        assert vdf.shape() == (1234, 14)
        drop("titanic_pandas")
        vdf = pandas_to_vertica(df=df)
        assert vdf.shape() == (1234, 14)
        drop("test_df")
        pandas_to_vertica(df, name="test_df", schema="public")
        pandas_to_vertica(df, name="test_df", schema="public", insert=True)
        vdf = pandas_to_vertica(df, name="test_df", schema="public", insert=True)
        assert vdf.shape() == (3702, 14)
        drop("test_df")
        # Problem with '\'
        # d = {"col1": [1, 2, 3, 4], "col2": ["red", 'gre"en', "b\lue", 'p\i""nk']}
        # df = pd.DataFrame(data=d)
        # vdf = pandas_to_vertica(df)
        # assert vdf.shape() == (4, 2)

    def test_pcsv(self):
        result = pcsv(os.path.dirname(verticapy.__file__) + "/data/titanic.csv")
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

    def test_read_json(self):
        drop("public.titanic_verticapy_test_json", method="table")
        path = (
            os.path.dirname(verticapy.__file__)
            + "/tests/utilities/titanic-passengers.json"
        )
        result = read_json(
            path, table_name="titanic_verticapy_test_json", schema="public"
        )
        assert result.shape() == (891, 15)
        drop("public.titanic_verticapy_test_json", method="table")
        result = read_json(path, table_name="titanic_verticapy_test_json")
        assert result.shape() == (891, 15)
        drop("public.titanic_verticapy_test_json", method="table")
        # TODO
        # test the param gen_tmp_table_name

    def test_read_csv(self):
        path = os.path.dirname(verticapy.__file__) + "/data/titanic.csv"
        # with schema
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
            path, table_name="titanic_verticapy_test_csv", schema="public", insert=True
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
        result = read_csv(
            path,
            table_name="titanic_verticapy_test_csv",
            dtype={
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
            },
        )
        assert result.shape() == (1234, 14)
        drop("v_temp_schema.titanic_verticapy_test_csv", method="table")
        # genSQL
        result = read_csv(
            path, schema="public", table_name="titanic_verticapy_test_csv", genSQL=True
        )
        assert result[0][0:50] == 'CREATE TABLE "public"."titanic_verticapy_test_csv"'
        assert result[1][0:42] == 'COPY "public"."titanic_verticapy_test_csv"'
        # TODO
        # test the param gen_tmp_table_name

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

    def test_tablesample(self):
        result = tablesample(
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

    def test_to_tablesample(self):
        result = to_tablesample('SELECT 1 AS "verticapy test *+""";')
        assert result['verticapy test *+"'] == [1]

    def test_vDataFrameSQL(self):
        result = vDataFrameSQL('(SELECT 1 AS "verticapy test *+") x',)
        assert result["verticapy test *+"].avg() == 1.0

    @pytest.mark.skip(reason="this test will be implemented later")
    def test_set_option(self):
        pass

    def test_version(self):
        result = version()
        assert result[0] < 20
        try:
            version([99, 1, 1])
            fail = False
        except:
            fail = True
        assert fail
