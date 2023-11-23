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

# Other Modules
import matplotlib.pyplot as plt

# VerticaPy
import verticapy
from verticapy import (
    vDataFrame,
    drop,
    set_option,
)
from verticapy.connection import current_cursor
from verticapy.datasets import load_winequality, load_dataset_num
from verticapy.learn.cluster import BisectingKMeans

# Matplotlib skip
import matplotlib

matplotlib_version = matplotlib.__version__
skip_plt = pytest.mark.skipif(
    matplotlib_version > "3.5.2",
    reason="Test skipped on matplotlib version greater than 3.5.2",
)

set_option("print_info", False)


@pytest.fixture(scope="module")
def winequality_vd():
    winequality = load_winequality()
    yield winequality
    drop(
        name="public.winequality",
    )


@pytest.fixture(scope="module")
def bsk_data_vd():
    bsk_data = load_dataset_num(table_name="bsk_data", schema="public")
    yield bsk_data
    drop(name="public.bsk_data", method="table")


@pytest.fixture(scope="module")
def model(bsk_data_vd):
    current_cursor().execute("DROP MODEL IF EXISTS bsk_model_test")

    current_cursor().execute(
        "SELECT BISECTING_KMEANS('bsk_model_test', 'public.bsk_data', '*', 3 USING PARAMETERS exclude_columns='id', kmeans_seed=11, id_column='id')"
    )

    model_class = BisectingKMeans("bsk_model_test", n_cluster=3, max_iter=10)
    model_class.X = ["col1", "col2", "col3", "col4"]
    model_class._compute_attributes()

    yield model_class
    model_class.drop()


class TestBisectingKMeans:
    def test_repr(self, model):
        assert model.__repr__() == "<BisectingKMeans>"

    def test_deploySQL(self, model):
        expected_sql = 'APPLY_BISECTING_KMEANS("col1", "col2", "col3", "col4" USING PARAMETERS model_name = \'bsk_model_test\', match_by_pos = \'true\')'
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self):
        current_cursor().execute("DROP MODEL IF EXISTS bsk_model_test_drop")
        model_test = BisectingKMeans(
            "bsk_model_test_drop",
        )
        model_test.fit("public.bsk_data", ["col1", "col2", "col3", "col4"])

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'bsk_model_test_drop'"
        )
        assert current_cursor().fetchone()[0] == "bsk_model_test_drop"

        model_test.drop()
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'bsk_model_test_drop'"
        )
        assert current_cursor().fetchone() is None

    def test_get_vertica_attributes(self, model):
        m_att = model.get_vertica_attributes()

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

        assert (
            model.get_vertica_attributes("num_of_clusters")["num_of_clusters"][0] == 3
        )
        assert (
            model.get_vertica_attributes("dimensions_of_dataset")[
                "dimensions_of_dataset"
            ][0]
            == 4
        )
        assert (
            model.get_vertica_attributes("num_of_clusters_found")[
                "num_of_clusters_found"
            ][0]
            == 3
        )
        assert (
            model.get_vertica_attributes("height_of_BKTree")["height_of_BKTree"][0] == 3
        )

        m_att_bktree = model.get_vertica_attributes(attr_name="BKTree")
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

    def test_set_params(self, model):
        model.set_params({"max_iter": 200})
        assert model.get_params()["max_iter"] == 200

    def test_model_from_vDF(self, bsk_data_vd):
        current_cursor().execute("DROP MODEL IF EXISTS bsk_vDF")
        model_test = BisectingKMeans(
            "bsk_vDF",
        )
        model_test.fit(bsk_data_vd, ["col1", "col2", "col3", "col4"])
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'bsk_vDF'"
        )
        assert current_cursor().fetchone()[0] == "bsk_vDF"
        model_test.drop()

    def test_init_method(self):
        model_test_kmeanspp = BisectingKMeans("bsk_kmeanspp_test", init="kmeanspp")
        model_test_kmeanspp.drop()
        model_test_kmeanspp.fit("public.bsk_data", ["col1", "col2", "col3", "col4"])

        assert (
            "kmeans_center_init_method='kmeanspp'"
            in model_test_kmeanspp.get_vertica_attributes("call_string")["call_string"][
                0
            ]
        )
        model_test_kmeanspp.drop()

        model_test_pseudo = BisectingKMeans("bsk_pseudo_test", init="pseudo")
        model_test_pseudo.drop()
        model_test_pseudo.fit("public.bsk_data", ["col1", "col2", "col3", "col4"])
        assert (
            "kmeans_center_init_method='pseudo'"
            in model_test_pseudo.get_vertica_attributes("call_string")["call_string"][0]
        )
        model_test_pseudo.drop()

    @skip_plt
    def test_get_plot(self, winequality_vd):
        current_cursor().execute("DROP MODEL IF EXISTS model_test_plot")
        model_test = BisectingKMeans(
            "model_test_plot",
        )
        model_test.fit(winequality_vd, ["alcohol", "quality"])
        result = model_test.plot()
        assert len(result.get_default_bbox_extra_artists()) > 7
        plt.close("all")
        model_test.drop()

    def test_to_graphviz(self, model):
        gvz_tree_0 = model.to_graphviz(
            round_score=4,
            percent=True,
            vertical=False,
            node_style={"shape": "box", "style": "filled"},
            arrow_style={"color": "blue"},
            leaf_style={"shape": "circle", "style": "filled"},
        )
        assert 'digraph Tree{\ngraph [rankdir = "LR"];\n0' in gvz_tree_0
        assert "0 -> 1" in gvz_tree_0

    def test_plot_tree(self, model):
        result = model.plot_tree()
        assert model.to_graphviz() == result.source.strip()

    def test_to_python(self, model):
        current_cursor().execute(
            "SELECT APPLY_BISECTING_KMEANS(5.006, 3.418, 1.464, 0.244 USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model.model_name
            )
        )
        prediction = current_cursor().fetchone()
        assert prediction == pytest.approx(
            model.to_python()([[5.006, 3.418, 1.464, 0.244]])
        )

    def test_to_sql(self, model):
        current_cursor().execute(
            "SELECT APPLY_BISECTING_KMEANS(5.006, 3.418, 1.464, 0.244 USING PARAMETERS model_name = '{}', match_by_pos=True)::float, {}::float".format(
                model.model_name, model.to_sql([5.006, 3.418, 1.464, 0.244])
            )
        )
        prediction = current_cursor().fetchone()
        assert prediction[0] == pytest.approx(prediction[1])

    def test_to_memmodel(self, model):
        mmodel = model.to_memmodel()
        res = mmodel.predict([[5.006, 3.418, 1.464, 0.244], [3.0, 11.0, 1993.0, 0.0]])
        res_py = model.to_python()(
            [[5.006, 3.418, 1.464, 0.244], [3.0, 11.0, 1993.0, 0.0]]
        )
        assert res[0] == res_py[0]
        assert res[1] == res_py[1]
        vdf = vDataFrame("public.bsk_data")
        vdf["prediction_sql"] = mmodel.predict_sql(["col1", "col2", "col3", "col4"])
        model.predict(vdf, name="prediction_vertica_sql")
        score = vdf.score("prediction_sql", "prediction_vertica_sql", metric="accuracy")
        assert score == pytest.approx(1.0)

    def test_optional_name(self):
        model = BisectingKMeans()
        assert model.model_name is not None
