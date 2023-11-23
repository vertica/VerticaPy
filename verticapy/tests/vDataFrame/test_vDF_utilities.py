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
import pytest

# Standard Python Modules
import os, pickle
from math import ceil, floor

# Other Modules
import pandas

# VerticaPy
from verticapy import (
    vDataFrame,
    drop,
    set_option,
    read_shp,
    read_csv,
    read_json,
)
from verticapy.connection import current_cursor
import verticapy.stats as st
from verticapy.datasets import load_titanic, load_cities, load_amazon, load_world
from verticapy.sql.sys import current_session, username

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(name="public.titanic")


@pytest.fixture(scope="module")
def cities_vd():
    cities = load_cities()
    yield cities
    drop(name="public.cities")


@pytest.fixture(scope="module")
def amazon_vd():
    amazon = load_amazon()
    yield amazon
    drop(name="public.amazon")


@pytest.fixture(scope="module")
def world_vd():
    world = load_world()
    yield world
    drop(name="public.world")


class TestvDFUtilities:
    def test_vDF_repr(self, titanic_vd):
        # vdf
        repr_vdf = titanic_vd.__repr__()
        assert "pclass" in repr_vdf
        assert "survived" in repr_vdf
        assert 10000 < len(repr_vdf) < 1000000
        repr_html_vdf = titanic_vd._repr_html_()
        assert 10000 < len(repr_html_vdf) < 10000000
        assert "<table>" in repr_html_vdf
        assert "data:image/png;base64," in repr_html_vdf
        # vdc
        repr_vdc = titanic_vd["age"].__repr__()
        assert "age" in repr_vdc
        assert "60" in repr_vdc
        assert 500 < len(repr_vdc) < 5000
        repr_html_vdc = titanic_vd["age"]._repr_html_()
        assert 10000 < len(repr_html_vdc) < 10000000
        assert "<table>" in repr_html_vdc
        assert "data:image/png;base64," in repr_html_vdc

    def test_vDF_magic(self, titanic_vd):
        assert (
            str(titanic_vd["name"]._in(["Madison", "Ashley", None]))
            == "(\"name\") IN ('Madison', 'Ashley', NULL)"
        )
        assert str(titanic_vd["age"]._between(1, 4)) == '("age") BETWEEN (1) AND (4)'
        assert str(titanic_vd["age"]._as("age2")) == '("age") AS age2'
        assert str(titanic_vd["age"]._distinct()) == 'DISTINCT ("age")'
        assert (
            str(
                st.sum(titanic_vd["age"])._over(
                    by=[titanic_vd["pclass"], titanic_vd["sex"]],
                    order_by=[titanic_vd["fare"]],
                )
            )
            == 'SUM("age") OVER (PARTITION BY "pclass", "sex" ORDER BY "fare")'
        )
        assert str(abs(titanic_vd["age"])) == 'ABS("age")'
        assert str(ceil(titanic_vd["age"])) == 'CEIL("age")'
        assert str(floor(titanic_vd["age"])) == 'FLOOR("age")'
        assert str(round(titanic_vd["age"], 2)) == 'ROUND("age", 2)'
        assert str(-titanic_vd["age"]) == '-("age")'
        assert str(+titanic_vd["age"]) == '+("age")'
        assert str(titanic_vd["age"] % 2) == 'MOD("age", 2)'
        assert str(2 % titanic_vd["age"]) == 'MOD(2, "age")'
        assert str(titanic_vd["age"] ** 2) == 'POWER("age", 2)'
        assert str(2 ** titanic_vd["age"]) == 'POWER(2, "age")'
        assert str(titanic_vd["age"] + 3) == '("age") + (3)'
        assert str(3 + titanic_vd["age"]) == '(3) + ("age")'
        assert str(titanic_vd["age"] - 3) == '("age") - (3)'
        assert str(3 - titanic_vd["age"]) == '(3) - ("age")'
        assert str(titanic_vd["age"] * 3) == '("age") * (3)'
        assert str(3 * titanic_vd["age"]) == '(3) * ("age")'
        assert str(titanic_vd["age"] // 3) == '("age") // (3)'
        assert str(3 // titanic_vd["age"]) == '(3) // ("age")'
        assert str(titanic_vd["age"] > 3) == '("age") > (3)'
        assert str(3 > titanic_vd["age"]) == '("age") < (3)'
        assert str(titanic_vd["age"] >= 3) == '("age") >= (3)'
        assert str(3 >= titanic_vd["age"]) == '("age") <= (3)'
        assert str(titanic_vd["age"] < 3) == '("age") < (3)'
        assert str(3 < titanic_vd["age"]) == '("age") > (3)'
        assert str(titanic_vd["age"] <= 3) == '("age") <= (3)'
        assert str(3 <= titanic_vd["age"]) == '("age") >= (3)'
        assert (
            str((3 >= titanic_vd["age"]) & (titanic_vd["age"] <= 50))
            == '(("age") <= (3)) AND (("age") <= (50))'
        )
        assert (
            str((3 >= titanic_vd["age"]) | (titanic_vd["age"] <= 50))
            == '(("age") <= (3)) OR (("age") <= (50))'
        )
        assert str("Mr " + titanic_vd["name"]) == "('Mr ') || (\"name\")"
        assert str(titanic_vd["name"] + " .") == "(\"name\") || (' .')"
        assert str(3 * titanic_vd["name"]) == 'REPEAT("name", 3)'
        assert str(titanic_vd["name"] * 3) == 'REPEAT("name", 3)'
        assert str(titanic_vd["age"] == 3) == '("age") = (3)'
        assert str(3 == titanic_vd["age"]) == '("age") = (3)'
        assert str(titanic_vd["age"] != 3) == '("age") != (3)'
        assert str(None != titanic_vd["age"]) == '("age") IS NOT NULL'
        assert titanic_vd["fare"][0] >= 0
        assert titanic_vd[["fare"]][0][0] >= 0
        assert titanic_vd[titanic_vd["fare"] > 500].shape()[0] == 4
        assert titanic_vd[titanic_vd["fare"] < 500].shape()[0] == 1229
        assert titanic_vd[titanic_vd["fare"] * 4 + 2 < 500].shape()[0] == 1167
        assert titanic_vd[titanic_vd["fare"] / 4 - 2 < 500].shape()[0] == 1233

    def test_vDF_to_csv(self, titanic_vd):
        titanic_vd.copy().select(["age", "fare"]).sort({"age": "desc", "fare": "desc"})[
            0:2
        ].to_csv("verticapy_test_to_csv.csv")
        try:
            file = open("verticapy_test_to_csv.csv", "r")
            result = file.read()
            assert result == '"age","fare"\n80.000,30.00000\n76.000,78.85000'
        except:
            os.remove("verticapy_test_to_csv.csv")
            file.close()
            raise
        os.remove("verticapy_test_to_csv.csv")
        file.close()
        # multiple files
        try:
            titanic_vd.to_csv(
                "titanic_verticapy_test_to_csv",
                n_files=3,
                order_by=["name", "age", "fare"],
            )
            titanic_test = read_csv("titanic_verticapy_test_to_csv/*.csv")
            assert titanic_test.shape() == (1234, 14)
        except:
            os.remove("titanic_verticapy_test_to_csv/1.csv")
            os.remove("titanic_verticapy_test_to_csv/2.csv")
            os.remove("titanic_verticapy_test_to_csv/3.csv")
            os.rmdir("titanic_verticapy_test_to_csv")
            raise
        os.remove("titanic_verticapy_test_to_csv/1.csv")
        os.remove("titanic_verticapy_test_to_csv/2.csv")
        os.remove("titanic_verticapy_test_to_csv/3.csv")
        os.rmdir("titanic_verticapy_test_to_csv")

    def test_vDF_to_parquet(self, titanic_vd):
        session_id = f"{current_session()}_{username()}"
        name = "parquet_test_{}".format(session_id)
        result = titanic_vd.to_parquet(name)
        assert result["Rows Exported"][0] == 1234
        # TODO: erasing the folder

    def test_vDF_to_db(self, titanic_vd):
        drop("verticapy_titanic_tmp")
        # testing relation_type = view
        try:
            titanic_vd.copy().to_db(
                name="verticapy_titanic_tmp",
                usecols=["age", "fare", "survived"],
                relation_type="view",
                db_filter="age > 40",
                nb_split=3,
            )
            titanic_tmp = vDataFrame("verticapy_titanic_tmp")
            assert titanic_tmp.shape() == (220, 4)
            assert titanic_tmp["_verticapy_split_"].min() == 0
            assert titanic_tmp["_verticapy_split_"].max() == 2
            current_cursor().execute(
                "SELECT table_name FROM view_columns WHERE table_name = 'verticapy_titanic_tmp'"
            )
            result = current_cursor().fetchone()
            assert result[0] == "verticapy_titanic_tmp"
        except:
            drop("verticapy_titanic_tmp", method="view")
            raise
        drop("verticapy_titanic_tmp", method="view")
        # testing relation_type = table
        try:
            titanic_vd.copy().to_db(
                name="verticapy_titanic_tmp",
                usecols=["age", "fare", "survived"],
                relation_type="table",
                db_filter="age > 40",
                nb_split=3,
            )
            titanic_tmp = vDataFrame("verticapy_titanic_tmp")
            assert titanic_tmp.shape() == (220, 4)
            assert titanic_tmp["_verticapy_split_"].min() == 0
            assert titanic_tmp["_verticapy_split_"].max() == 2
            current_cursor().execute(
                "SELECT table_name FROM columns WHERE table_name = 'verticapy_titanic_tmp'"
            )
            result = current_cursor().fetchone()
            assert result[0] == "verticapy_titanic_tmp"
        except:
            drop("verticapy_titanic_tmp")
            raise
        drop("verticapy_titanic_tmp")
        # testing relation_type = temporary table
        try:
            titanic_vd.copy().to_db(
                name="verticapy_titanic_tmp",
                usecols=["age", "fare", "survived"],
                relation_type="temporary",
                db_filter="age > 40",
                nb_split=3,
            )
            titanic_tmp = vDataFrame("verticapy_titanic_tmp")
            assert titanic_tmp.shape() == (220, 4)
            assert titanic_tmp["_verticapy_split_"].min() == 0
            assert titanic_tmp["_verticapy_split_"].max() == 2
            current_cursor().execute(
                "SELECT table_name FROM columns WHERE table_name = 'verticapy_titanic_tmp'"
            )
            result = current_cursor().fetchone()
            assert result[0] == "verticapy_titanic_tmp"
        except:
            drop("verticapy_titanic_tmp")
            raise
        drop("verticapy_titanic_tmp")
        # testing relation_type = temporary local table
        try:
            titanic_vd.copy().to_db(
                name="verticapy_titanic_tmp",
                usecols=["age", "fare", "survived"],
                relation_type="local",
                db_filter="age > 40",
                nb_split=3,
            )
            titanic_tmp = vDataFrame("v_temp_schema.verticapy_titanic_tmp")
            assert titanic_tmp.shape() == (220, 4)
            assert titanic_tmp["_verticapy_split_"].min() == 0
            assert titanic_tmp["_verticapy_split_"].max() == 2
            current_cursor().execute(
                "SELECT table_name FROM columns WHERE table_name = 'verticapy_titanic_tmp'"
            )
            result = current_cursor().fetchone()
            assert result[0] == "verticapy_titanic_tmp"
        except:
            drop("verticapy_titanic_tmp")
            raise
        drop("verticapy_titanic_tmp")

    def test_vDF_to_json(self, titanic_vd):
        session_id = f"{current_session()}_{username()}"
        titanic_vd.copy().select(["age", "fare"]).sort({"age": "desc", "fare": "desc"})[
            0:2
        ].to_json("verticapy_test_to_json.json")
        try:
            file = open("verticapy_test_to_json.json", "r")
            result = file.read()
            assert (
                result
                == '[\n{"age": 80.000, "fare": 30.00000},\n{"age": 76.000, "fare": 78.85000}\n]'
            )
        except:
            os.remove("verticapy_test_to_json.json")
            file.close()
            raise
        os.remove("verticapy_test_to_json.json")
        file.close()
        # multiple files
        try:
            titanic_vd.to_json(
                "titanic_verticapy_test_to_json",
                n_files=3,
                order_by=["name", "age", "fare"],
            )
            titanic_test = read_json("titanic_verticapy_test_to_json/*.json")
            assert titanic_test.shape() == (1234, 14)
        except:
            os.remove("titanic_verticapy_test_to_json/1.json")
            os.remove("titanic_verticapy_test_to_json/2.json")
            os.remove("titanic_verticapy_test_to_json/3.json")
            os.rmdir("titanic_verticapy_test_to_json")
            raise
        os.remove("titanic_verticapy_test_to_json/1.json")
        os.remove("titanic_verticapy_test_to_json/2.json")
        os.remove("titanic_verticapy_test_to_json/3.json")
        os.rmdir("titanic_verticapy_test_to_json")

    def test_vDF_to_list(self, titanic_vd):
        result = titanic_vd.select(["age", "survived"])[:20].to_list()
        assert len(result) == 20
        assert len(result[0]) == 2

    def test_vDF_to_numpy(self, titanic_vd):
        result = titanic_vd.select(["age", "survived"])[:20].to_numpy()
        assert result.shape == (20, 2)

    def test_vDF_to_pandas(self, titanic_vd):
        result = titanic_vd.to_pandas()
        assert isinstance(result, pandas.DataFrame)
        assert result.shape == (1234, 14)

    def test_vDF_to_pickle(self, titanic_vd):
        result = titanic_vd.select(["age", "survived"])[:20].to_pickle("save.p")
        pickle.DEFAULT_PROTOCOL = 4
        result_tmp = pickle.load(open("save.p", "rb"))
        assert result_tmp.shape() == (20, 2)
        os.remove("save.p")

    @pytest.mark.skip(reason="Python 3.6 VE could not install proper dependencies")
    def test_vDF_to_geopandas(self, world_vd):
        import geopandas

        result = world_vd.to_geopandas(geometry="geometry")
        assert isinstance(result, geopandas.GeoDataFrame)
        assert result.shape == (177, 4)

    def test_vDF_to_shp(self, cities_vd):
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

    def test_vDF_del_catalog(self, titanic_vd):
        result = titanic_vd.copy()
        result.describe(method="numerical")
        assert "max" in result["age"]._catalog
        assert "avg" in result["age"]._catalog
        result.del_catalog()
        assert "max" not in result["age"]._catalog
        assert "avg" not in result["age"]._catalog

    def test_vDF_load(self, titanic_vd):
        result = titanic_vd.copy()
        result._vars["saving"] = []
        result.save()
        assert len(result._vars["saving"]) == 1
        result.filter("age < 40")
        result["embarked"].drop()
        assert result.shape() == (760, 13)
        result = result.load()
        assert len(result._vars["saving"]) == 0
        assert result.shape() == (1234, 14)

    def test_vDF_save(self, titanic_vd):
        result = titanic_vd.copy()
        result._vars["saving"] = []
        result.save()
        assert len(result._vars["saving"]) == 1

    def test_vDF_catcol(self, titanic_vd):
        result = [
            elem.replace('"', "").lower()
            for elem in titanic_vd.catcol(max_cardinality=4)
        ]
        result.sort()
        assert result == [
            "boat",
            "cabin",
            "embarked",
            "home.dest",
            "name",
            "pclass",
            "sex",
            "survived",
            "ticket",
        ]

    def test_vDF_category(self, titanic_vd):
        # test for category = float
        result = titanic_vd["age"].category()
        assert result == "float"

        # test for category = text
        result2 = titanic_vd["name"].category()
        assert result2 == "text"

        # test for category = int
        result3 = titanic_vd["pclass"].category()
        assert result3 == "int"

    def test_vDF_current_relation(self, titanic_vd):
        result = titanic_vd.current_relation().split(".")[1].replace('"', "")
        assert result == "titanic"

    def test_vDF_datecol(self, amazon_vd):
        result = [elem.replace('"', "") for elem in amazon_vd.datecol()]
        result.sort()
        assert result == ["date"]

    def test_vDF_dtypes(self, amazon_vd):
        result = amazon_vd.dtypes()["dtype"]
        result.sort()
        assert result == ["date", "int", "varchar(32)"]

    def test_vDF_dtype(self, amazon_vd):
        # test of dtype on int
        result = amazon_vd["number"].dtype()
        assert result == "int"

        # test of dtype on date
        result2 = amazon_vd["date"].dtype()
        assert result2 == "date"

        # test of dtype on varchar(32)
        result3 = amazon_vd["state"].dtype()
        assert result3 == "varchar(32)"

        # verify ctype is the same
        assert result == amazon_vd["number"].ctype()
        assert result2 == amazon_vd["date"].ctype()
        assert result3 == amazon_vd["state"].ctype()

    def test_vDF_empty(self, amazon_vd):
        # test for non-empty vDataFrame
        result = amazon_vd.empty()
        assert not result

        # test for empty vDataFrame
        result2 = amazon_vd.copy().drop(["number", "date", "state"]).empty()
        assert result2

    def test_vDF_expected_store_usage(self, titanic_vd):
        # test expected_size
        result = titanic_vd.expected_store_usage()["expected_size (b)"][-1]
        assert result == pytest.approx(85947.0)

        # test max_size
        result2 = titanic_vd.expected_store_usage()["max_size (b)"][-1]
        assert result2 == pytest.approx(504492.0)

    def test_vDF_explain(self, titanic_vd):
        # test with parameter digraph = False
        result = titanic_vd.explain(digraph=False)
        assert isinstance(result, str)

        # test with parameter digraph = True
        result2 = titanic_vd.explain(digraph=True)
        assert result2[0:7] == "digraph"

    def test_vDF_get_columns(self, titanic_vd):
        result = [
            elem.replace('"', "")
            for elem in titanic_vd.get_columns(exclude_columns=["sibsp", "age"])
        ]
        result.sort()
        assert result == [
            "boat",
            "body",
            "cabin",
            "embarked",
            "fare",
            "home.dest",
            "name",
            "parch",
            "pclass",
            "sex",
            "survived",
            "ticket",
        ]

    def test_vDF_head(self, titanic_vd):
        # testing vDataFrame[].head
        result = titanic_vd.copy().sort({"age": "desc"})["age"].head(2)
        assert result["age"] == [80.0, 76.0]

        # testing vDataFrame.head
        result = titanic_vd.copy().sort({"age": "desc"}).head(2)
        assert result["age"] == [80.0, 76.0]
        assert result["fare"] == [30.0, 78.85]

    def test_vDF_iloc(self, titanic_vd):
        # testing vDataFrame[].iloc
        result = titanic_vd.copy().sort({"age": "desc"})["age"].iloc(2, 1)
        assert result["age"] == [76.0, 74.0]

        # testing vDataFrame.iloc
        result = titanic_vd.copy().sort({"age": "desc"}).iloc(2, 1, ["age", "fare"])
        assert result["age"] == [76.0, 74.0]
        assert result["fare"] == [78.85, 7.775]

    def test_vDF_info(self, titanic_vd):
        result = titanic_vd.copy().filter("age > 0")
        result["fare"].drop_outliers()
        result = len(result.info().split("\n")) - 1
        assert result == 2

    def test_vDF_isdate(self, amazon_vd):
        # test for date-like vDataColumn
        result = amazon_vd["date"].isdate()
        assert result

        # test for non-date-like vDataColumn
        result2 = amazon_vd["number"].isdate()
        assert not result2

        result2 = amazon_vd["state"].isdate()
        assert not result2

    def test_vDF_isnum(self, amazon_vd):
        # test for numerical vDataColumn
        result = amazon_vd["number"].isnum()
        assert result

        # test for non-numerical vDataColumn
        result = amazon_vd["date"].isnum()
        assert not result

        result = amazon_vd["state"].isnum()
        assert not result

    @pytest.mark.skip(reason="test not stable")
    def test_vDF_memory_usage(self, amazon_vd):
        # testing vDataFrame[].memory_usage
        result = amazon_vd["number"].memory_usage()
        assert result == pytest.approx(1690, 5e-2)

        # testing vDataFrame.memory_usage
        result2 = amazon_vd.memory_usage()
        assert result2["value"][0] == pytest.approx(684, 5e-2)
        assert result2["value"][1] == pytest.approx(1688, 5e-2)
        assert result2["value"][2] == pytest.approx(1689, 5e-2)
        assert result2["value"][3] == pytest.approx(1690, 5e-2)
        assert result2["value"][4] == pytest.approx(5751, 5e-2)

    def test_vDF_numcol(self, titanic_vd):
        result = [elem.replace('"', "") for elem in titanic_vd.numcol()]
        result.sort()
        assert result == [
            "age",
            "body",
            "fare",
            "parch",
            "pclass",
            "sibsp",
            "survived",
        ]

    def test_vDF_tail(self, titanic_vd):
        # testing vDataFrame[].tail
        result = titanic_vd.copy().sort(["age"])["age"].tail(2)
        assert result["age"] == [76.0, 80.0]

        # testing vDataFrame.tail
        result = titanic_vd.copy().sort(["age"]).tail(2)
        assert result["age"] == [76.0, 80.0]
        assert result["fare"] == [78.85, 30.0]

    def test_vDF_sql(self, titanic_vd):
        sql = """-- Selecting some columns \n
                 SELECT 
                    age, 
                    fare 
                 FROM titanic 
                 WHERE age IS NOT NULL;;"""
        vdf = vDataFrame(sql)
        assert vdf.shape() == (997, 2)
        vdf = vDataFrame(sql, usecols=["age"])
        assert vdf.shape() == (997, 1)

    def test_vDF_store_usage(self, titanic_vd):
        result = titanic_vd["age"].store_usage()
        assert result == pytest.approx(5908, 1e-2)

    def test_vDF_swap(self, titanic_vd):
        result = titanic_vd.copy()
        result.swap("sex", 0)
        result.swap("pclass", 1)
        assert result.get_columns()[0].replace('"', "") == "sex"
        assert result.get_columns()[1].replace('"', "") == "pclass"
        result.swap("pclass", "sex")
        assert result.get_columns()[0].replace('"', "") == "pclass"
        assert result.get_columns()[1].replace('"', "") == "sex"
