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

import pytest, vertica_python
from verticapy import drop, set_option, vDataFrame
from verticapy.geo import *
from verticapy.learn.neighbors import KNeighborsClassifier

set_option("print_info", False)


@pytest.fixture(scope="module")
def cities_vd(base):
    from verticapy.datasets import load_cities

    cities = load_cities(cursor=base.cursor)
    yield cities
    with warnings.catch_warnings(record=True) as w:
        drop(
            name="public.cities", cursor=base.cursor,
        )


@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    with warnings.catch_warnings(record=True) as w:
        drop(
            name="public.titanic", cursor=base.cursor,
        )


@pytest.fixture(scope="module")
def world_vd(base):
    from verticapy.datasets import load_world

    cities = load_world(cursor=base.cursor)
    yield cities
    with warnings.catch_warnings(record=True) as w:
        drop(
            name="public.world", cursor=base.cursor,
        )


class TestUtilities:
    def test_create_verticapy_schema(self, base):
        with warnings.catch_warnings(record=True) as w:
            drop("verticapy", base.cursor, method="schema")
        create_verticapy_schema(base.cursor)
        base.cursor.execute(
            "SELECT table_name FROM columns WHERE table_schema = 'verticapy' GROUP BY 1 ORDER BY 1;"
        )
        result = [elem[0] for elem in base.cursor.fetchall()]
        assert result == ["attr", "models"]
        drop("verticapy", base.cursor, method="schema")

    def test_drop(self, base, world_vd):
        base.cursor.execute("DROP TABLE IF EXISTS public.drop_data")
        base.cursor.execute(
            'CREATE TABLE IF NOT EXISTS public.drop_data(Id identity(2000) primary key, transportation VARCHAR, gender VARCHAR, "owned cars" INT, cost VARCHAR, income CHAR(4)) ORDER BY id SEGMENTED BY HASH(id) ALL NODES KSAFE;'
        )
        base.cursor.execute(
            "INSERT INTO drop_data(transportation, gender, \"owned cars\", cost, income) VALUES ('Bus', 'Male', 0, 'Cheap', 'Low')"
        )
        base.cursor.execute(
            "INSERT INTO drop_data(transportation, gender, \"owned cars\", cost, income) VALUES ('Bus', 'Male', 1, 'Cheap', 'Med')"
        )
        base.cursor.execute(
            "INSERT INTO drop_data(transportation, gender, \"owned cars\", cost, income) VALUES ('Train', 'Female', 1, 'Cheap', 'Med')"
        )
        base.cursor.execute(
            "INSERT INTO drop_data(transportation, gender, \"owned cars\", cost, income) VALUES ('Bus', 'Female', 0, 'Cheap', 'Low')"
        )
        base.cursor.execute(
            "INSERT INTO drop_data(transportation, gender, \"owned cars\", cost, income) VALUES ('Bus', 'Male', 1, 'Cheap', 'Med')"
        )
        base.cursor.execute(
            "INSERT INTO drop_data(transportation, gender, \"owned cars\", cost, income) VALUES ('Train', 'Male', 0, 'Standard', 'Med')"
        )
        base.cursor.execute(
            "INSERT INTO drop_data(transportation, gender, \"owned cars\", cost, income) VALUES ('Train', 'Female', 1, 'Standard', 'Med')"
        )
        base.cursor.execute(
            "INSERT INTO drop_data(transportation, gender, \"owned cars\", cost, income) VALUES ('Car', 'Female', 1, 'Expensive', 'Hig')"
        )
        base.cursor.execute(
            "INSERT INTO drop_data(transportation, gender, \"owned cars\", cost, income) VALUES ('Car', 'Male', 2, 'Expensive', 'Med')"
        )
        base.cursor.execute(
            "INSERT INTO drop_data(transportation, gender, \"owned cars\", cost, income) VALUES ('Car', 'Female', 2, 'Expensive', 'Hig')"
        )
        base.cursor.execute("COMMIT")
        # table
        base.cursor.execute("DROP TABLE IF EXISTS public.verticapy_table_test")
        base.cursor.execute("CREATE TABLE verticapy_table_test AS SELECT 1;")
        drop("verticapy_table_test", base.cursor)
        base.cursor.execute(
            "SELECT table_name FROM columns WHERE table_name = 'verticapy_table_test' GROUP BY 1;"
        )
        result = base.cursor.fetchall()
        assert result == []
        # view
        base.cursor.execute("DROP VIEW IF EXISTS public.verticapy_view_test")
        base.cursor.execute("CREATE VIEW verticapy_view_test AS SELECT 1;")
        drop("verticapy_view_test", base.cursor)
        base.cursor.execute(
            "SELECT table_name FROM view_columns WHERE table_name = 'verticapy_view_test' GROUP BY 1;"
        )
        result = base.cursor.fetchall()
        assert result == []
        # text index
        base.cursor.execute(
            "CREATE TEXT INDEX drop_index ON drop_data (id, transportation);"
        )
        drop("drop_index", base.cursor)
        with pytest.raises(vertica_python.errors.MissingRelation):
            base.cursor.execute("SELECT * FROM drop_index;")
        # model
        base.cursor.execute("DROP MODEL IF EXISTS public.verticapy_model_test")
        base.cursor.execute(
            "SELECT NAIVE_BAYES('public.verticapy_model_test', 'public.drop_data', 'transportation', 'gender, cost');"
        )
        drop("verticapy_model_test", base.cursor)
        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'verticapy_model_test' GROUP BY 1;"
        )
        result = base.cursor.fetchall()
        assert result == []
        # verticapy model
        with warnings.catch_warnings(record=True) as w:
            drop("verticapy", base.cursor, method="schema")
        create_verticapy_schema(base.cursor)
        model = KNeighborsClassifier("verticapy_model_test", base.cursor)
        model.fit("public.drop_data", ["gender", "cost",], "transportation")
        drop("verticapy_model_test", base.cursor)
        base.cursor.execute(
            "SELECT model_name FROM verticapy.models WHERE model_name = 'verticapy_model_test' GROUP BY 1;"
        )
        result = base.cursor.fetchall()
        assert result == []
        drop("verticapy", base.cursor, method="schema")
        # geo index
        world_copy = world_vd.copy()
        world_copy["id"] = "ROW_NUMBER() OVER (ORDER BY pop_est)"
        result = create_index(world_copy, "id", "geometry", "world_polygons", True)
        drop(
            "world_polygons", base.cursor,
        )
        with pytest.raises(vertica_python.errors.QueryError):
            describe_index("world_polygons", base.cursor, True,)
        drop(cursor=base.cursor,)

    def test_readSQL(self, base):
        result = readSQL('SELECT 1 AS "verticapy test *+""";', base.cursor,)
        assert result['verticapy test *+"'] == [1]

    def test_get_data_types(self, base):
        result = get_data_types(
            "SELECT 1 AS col1, 'abc' AS col2, '5 minutes'::interval AS col3",
            base.cursor,
        )
        assert result == [
            ["col1", "Integer"],
            ["col2", "Varchar(3)"],
            ["col3", "Interval"],
        ]

    def test_pandas_to_vertica(self, titanic_vd):
        df = titanic_vd.to_pandas()
        with warnings.catch_warnings(record=True) as w:
            drop(
                "titanic_pandas", titanic_vd._VERTICAPY_VARIABLES_["cursor"],
            )
        pandas_to_vertica(
            df=df,
            cursor=titanic_vd._VERTICAPY_VARIABLES_["cursor"],
            name="titanic_pandas",
        )
        vdf = vDataFrame(
            "titanic_pandas", cursor=titanic_vd._VERTICAPY_VARIABLES_["cursor"]
        )
        assert vdf.shape() == (1234, 14)
        drop(
            "titanic_pandas", titanic_vd._VERTICAPY_VARIABLES_["cursor"],
        )

    def test_pcsv(self, base):
        result = pcsv(
            os.path.dirname(verticapy.__file__) + "/data/titanic.csv", base.cursor
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

    def test_pjson(self, base):
        result = pjson(
            os.path.dirname(verticapy.__file__)
            + "/tests/utilities/titanic-passengers.json",
            base.cursor,
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
            "recordid": "Uuid",
        }

    def test_read_json(self, base):
        with warnings.catch_warnings(record=True) as w:
            drop(
                "public.titanic_verticapy_test", base.cursor,
            )
        result = read_json(
            os.path.dirname(verticapy.__file__)
            + "/tests/utilities/titanic-passengers.json",
            base.cursor,
            table_name="titanic_verticapy_test",
        )
        assert result.shape() == (891, 15)
        drop("titanic_verticapy_test", base.cursor)

    def test_read_csv(self, base):
        with warnings.catch_warnings(record=True) as w:
            drop(
                "public.titanic_verticapy_test", base.cursor,
            )
        result = read_csv(
            os.path.dirname(verticapy.__file__) + "/data/titanic.csv",
            base.cursor,
            table_name="titanic_verticapy_test",
        )
        assert result.shape() == (1234, 14)
        drop("titanic_verticapy_test", base.cursor)

    def test_read_shp(self, cities_vd):
        with warnings.catch_warnings(record=True) as w:
            drop(
                name="public.cities_test",
                cursor=cities_vd._VERTICAPY_VARIABLES_["cursor"],
            )
        cities_vd.to_shp("cities_test", "/home/dbadmin/", shape="Point")
        vdf = read_shp(
            "/home/dbadmin/cities_test.shp", cities_vd._VERTICAPY_VARIABLES_["cursor"]
        )
        assert vdf.shape() == (202, 3)
        try:
            os.remove("/home/dbadmin/cities_test.shp")
            os.remove("/home/dbadmin/cities_test.shx")
            os.remove("/home/dbadmin/cities_test.dbf")
        except:
            pass
        with warnings.catch_warnings(record=True) as w:
            drop(
                name="public.cities_test",
                cursor=cities_vd._VERTICAPY_VARIABLES_["cursor"],
            )

    def test_tablesample(self, base):
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
        result7 = result.to_vdf(base.cursor)["price"].mean()
        assert result7 == 2.0
        

    def test_to_tablesample(self, base):
        result = to_tablesample('SELECT 1 AS "verticapy test *+""";', base.cursor,)
        assert result['verticapy test *+"'] == [1]

    def test_vdf_from_relation(self, base):
        result = vdf_from_relation(
            '(SELECT 1 AS "verticapy test *+") x', cursor=base.cursor,
        )
        assert result["verticapy test *+"].avg() == 1.0

    def test_version(self, base):
        result = version(base.cursor,)
        assert result[0] < 20
        try:
            version(base.cursor, [99, 1, 1])
            fail = False
        except:
            fail = True
        assert fail
