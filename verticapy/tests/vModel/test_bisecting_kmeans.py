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

import pytest, warnings, sys, os, verticapy
from verticapy.learn.cluster import BisectingKMeans
from verticapy import vDataFrame, drop, set_option, vertica_conn
import matplotlib.pyplot as plt

set_option("print_info", False)


@pytest.fixture(scope="module")
def winequality_vd(base):
    from verticapy.datasets import load_winequality

    winequality = load_winequality(cursor=base.cursor)
    yield winequality
    drop(name="public.winequality", cursor=base.cursor)


@pytest.fixture(scope="module")
def bsk_data_vd(base):
    base.cursor.execute("DROP TABLE IF EXISTS public.bsk_data")
    base.cursor.execute(
        "CREATE TABLE IF NOT EXISTS public.bsk_data(Id INT, col1 FLOAT, col2 FLOAT, col3 FLOAT, col4 FLOAT)"
    )
    base.cursor.execute("INSERT INTO bsk_data VALUES (1, 7.2, 3.6, 6.1, 2.5)")
    base.cursor.execute("INSERT INTO bsk_data VALUES (2, 7.7, 2.8, 6.7, 2.0)")
    base.cursor.execute("INSERT INTO bsk_data VALUES (3, 7.7, 3.0, 6.1, 2.3)")
    base.cursor.execute("INSERT INTO bsk_data VALUES (4, 7.9, 3.8, 6.4, 2.0)")
    base.cursor.execute("INSERT INTO bsk_data VALUES (5, 4.4, 2.9, 1.4, 0.2)")
    base.cursor.execute("INSERT INTO bsk_data VALUES (6, 4.6, 3.6, 1.0, 0.2)")
    base.cursor.execute("INSERT INTO bsk_data VALUES (7, 4.7, 3.2, 1.6, 0.2)")
    base.cursor.execute("INSERT INTO bsk_data VALUES (8, 6.5, 2.8, 4.6, 1.5)")
    base.cursor.execute("INSERT INTO bsk_data VALUES (9, 6.8, 2.8, 4.8, 1.4)")
    base.cursor.execute("INSERT INTO bsk_data VALUES (10, 7.0, 3.2, 4.7, 1.4)")
    base.cursor.execute("COMMIT")

    bsk_data = vDataFrame(input_relation="public.bsk_data", cursor=base.cursor)
    yield bsk_data
    with warnings.catch_warnings(record=True) as w:
        drop(name="public.bsk_data", cursor=base.cursor)


@pytest.fixture(scope="module")
def model(base, bsk_data_vd):
    base.cursor.execute("DROP MODEL IF EXISTS bsk_model_test")

    base.cursor.execute(
        "SELECT BISECTING_KMEANS('bsk_model_test', 'public.bsk_data', '*', 3 USING PARAMETERS exclude_columns='id', kmeans_seed=11, id_column='id')"
    )

    model_class = BisectingKMeans(
        "bsk_model_test", cursor=base.cursor, n_cluster=3, max_iter=10
    )
    model_class.metrics_ = model_class.get_attr("Metrics")
    model_class.cluster_centers_ = model_class.get_attr("BKTree")
    model_class.X = ["col1", "col2", "col3", "col4"]

    yield model_class
    model_class.drop()


class TestBisectingKMeans:
    def test_repr(self, model):
        assert "bisecting_kmeans('bsk_model_test'" in model.__repr__()
        model_repr = BisectingKMeans("BisectingKMeans_repr")
        model_repr.drop()
        assert model_repr.__repr__() == "<BisectingKMeans>"

    def test_deploySQL(self, model):
        expected_sql = "APPLY_BISECTING_KMEANS(col1, col2, col3, col4 USING PARAMETERS model_name = 'bsk_model_test', match_by_pos = 'true')"
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self, base):
        base.cursor.execute("DROP MODEL IF EXISTS bsk_model_test_drop")
        model_test = BisectingKMeans("bsk_model_test_drop", cursor=base.cursor)
        model_test.fit("public.bsk_data", ["col1", "col2", "col3", "col4"])

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'bsk_model_test_drop'"
        )
        assert base.cursor.fetchone()[0] == "bsk_model_test_drop"

        model_test.drop()
        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'bsk_model_test_drop'"
        )
        assert base.cursor.fetchone() is None

    def test_get_attr(self, model):
        m_att = model.get_attr()

        assert m_att["attr_name"] == [
            "num_of_clusters",
            "dimensions_of_dataset",
            "num_of_clusters_found",
            "height_of_BKTree",
            "BKTree",
            "Metrics",
            "call_string",
        ]
        assert m_att["attr_fields"] == [
            "num_of_clusters",
            "dimensions_of_dataset",
            "num_of_clusters_found",
            "height_of_BKTree",
            "center_id, col1, col2, col3, col4, withinss, totWithinss, bisection_level, cluster_size, parent, left_child, right_child",
            "Measure, Value",
            "call_string",
        ]
        assert m_att["#_of_rows"] == [1, 1, 1, 1, 5, 7, 1]

        assert model.get_attr("num_of_clusters")["num_of_clusters"][0] == 3
        assert model.get_attr("dimensions_of_dataset")["dimensions_of_dataset"][0] == 4
        assert model.get_attr("num_of_clusters_found")["num_of_clusters_found"][0] == 3
        assert model.get_attr("height_of_BKTree")["height_of_BKTree"][0] == 3

        m_att_bktree = model.get_attr(attr_name="BKTree")
        assert m_att_bktree["bisection_level"] == [0, 1, 1, 2, 2]

    def test_get_params(self, model):
        assert model.get_params() == {
            "max_iter": 10,
            "tol": 0.0001,
            "n_cluster": 3,
            "init": "kmeanspp",
            "bisection_iterations": 1,
            "split_method": "sum_squares",
            "min_divisible_cluster_size": 2,
            "distance_method": "euclidean",
        }

    def test_get_predict(self, bsk_data_vd, model):
        bsk_data_copy = bsk_data_vd.copy()

        model.predict(bsk_data_copy, X=["col1", "col2", "col3", "col4"], name="pred")

        assert len(bsk_data_copy["pred"].distinct()) == 3

    def test_set_cursor(self, model):
        cur = vertica_conn(
            "vp_test_config",
            os.path.dirname(verticapy.__file__) + "/tests/verticaPy_test_tmp.conf",
        ).cursor()
        model.set_cursor(cur)
        model.cursor.execute("SELECT 1;")
        result = model.cursor.fetchone()
        assert result[0] == 1

    def test_set_params(self, model):
        model.set_params({"max_iter": 200})
        assert model.get_params()["max_iter"] == 200

    def test_model_from_vDF(self, base, bsk_data_vd):
        base.cursor.execute("DROP MODEL IF EXISTS bsk_vDF")
        model_test = BisectingKMeans("bsk_vDF", cursor=base.cursor)
        model_test.fit(bsk_data_vd, ["col1", "col2", "col3", "col4"])
        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'bsk_vDF'"
        )
        assert base.cursor.fetchone()[0] == "bsk_vDF"
        model_test.drop()

    def test_init_method(self, base):
        model_test_kmeanspp = BisectingKMeans(
            "bsk_kmeanspp_test", cursor=base.cursor, init="kmeanspp"
        )
        model_test_kmeanspp.drop()
        model_test_kmeanspp.fit("public.bsk_data", ["col1", "col2", "col3", "col4"])

        assert (
            model_test_kmeanspp.get_attr("call_string")["call_string"][0]
            == "bisecting_kmeans('bsk_kmeanspp_test', 'public.bsk_data', '\"col1\", \"col2\", \"col3\", \"col4\"', 8\nUSING PARAMETERS bisection_iterations=1, split_method='SUM_SQUARES', min_divisible_cluster_size=2, distance_method='euclidean', kmeans_center_init_method='kmeanspp', kmeans_epsilon=0.0001, kmeans_max_iterations=300, key_columns=''\"col1\", \"col2\", \"col3\", \"col4\"'')"
        )
        model_test_kmeanspp.drop()

        model_test_pseudo = BisectingKMeans(
            "bsk_pseudo_test", cursor=base.cursor, init="pseudo"
        )
        model_test_pseudo.drop()
        model_test_pseudo.fit("public.bsk_data", ["col1", "col2", "col3", "col4"])

        assert (
            model_test_pseudo.get_attr("call_string")["call_string"][0]
            == "bisecting_kmeans('bsk_pseudo_test', 'public.bsk_data', '\"col1\", \"col2\", \"col3\", \"col4\"', 8\nUSING PARAMETERS bisection_iterations=1, split_method='SUM_SQUARES', min_divisible_cluster_size=2, distance_method='euclidean', kmeans_center_init_method='pseudo', kmeans_epsilon=0.0001, kmeans_max_iterations=300, key_columns=''\"col1\", \"col2\", \"col3\", \"col4\"'')"
        )
        model_test_pseudo.drop()

    def test_get_plot(self, base, winequality_vd):
        base.cursor.execute("DROP MODEL IF EXISTS model_test_plot")
        model_test = BisectingKMeans("model_test_plot", cursor=base.cursor)
        model_test.fit(winequality_vd, ["alcohol", "quality"])
        result = model_test.plot()
        assert len(result.get_default_bbox_extra_artists()) == 16
        plt.close("all")
        model_test.drop()

    def test_plot_tree(self, model):
        result = model.plot_tree()
        assert result.by_attr()[0:3] == "[0]"

    def test_to_python(self, model):
        model.cursor.execute(
            "SELECT APPLY_BISECTING_KMEANS(5.006, 3.418, 1.464, 0.244 USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model.name
            )
        )
        prediction = model.cursor.fetchone()
        assert prediction == pytest.approx(
            model.to_python(return_str=False)([[5.006, 3.418, 1.464, 0.244]])
        )

    def test_to_sql(self, model):
        model.cursor.execute(
            "SELECT APPLY_BISECTING_KMEANS(5.006, 3.418, 1.464, 0.244 USING PARAMETERS model_name = '{}', match_by_pos=True)::float, {}::float".format(
                model.name, model.to_sql([5.006, 3.418, 1.464, 0.244])
            )
        )
        prediction = model.cursor.fetchone()
        assert prediction[0] == pytest.approx(prediction[1])
