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
import matplotlib.pyplot as plt

# VerticaPy
from verticapy import drop, set_option
from verticapy.connect import current_cursor
from verticapy.datasets import load_iris, load_winequality
from verticapy.learn.cluster import KMeans

set_option("print_info", False)


@pytest.fixture(scope="module")
def iris_vd():
    iris = load_iris()
    yield iris
    drop(name="public.iris",)


@pytest.fixture(scope="module")
def winequality_vd():
    winequality = load_winequality()
    yield winequality
    drop(name="public.winequality",)


@pytest.fixture(scope="module")
def model(iris_vd):
    model_class = KMeans(
        "kmeans_model_test",
        n_cluster=3,
        max_iter=10,
        init=[[7.2, 3.0, 5.8, 1.6], [6.9, 3.1, 4.9, 1.5], [5.7, 4.4, 1.5, 0.4]],
    )
    model_class.drop()
    model_class.fit(
        "public.iris",
        ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
    )
    yield model_class
    model_class.drop()


class TestKMeans:
    def test_repr(self, model):
        assert "kmeans" in model.__repr__()
        model_repr = KMeans("model_repr")
        model_repr.drop()
        assert model_repr.__repr__() == "<KMeans>"

    def test_deploySQL(self, model):
        expected_sql = 'APPLY_KMEANS("SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm" USING PARAMETERS model_name = \'kmeans_model_test\', match_by_pos = \'true\')'
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self):
        current_cursor().execute("DROP MODEL IF EXISTS kmeans_model_test_drop")
        model_test = KMeans("kmeans_model_test_drop",)
        model_test.fit("public.iris", ["SepalLengthCm", "SepalWidthCm"])

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'kmeans_model_test_drop'"
        )
        assert current_cursor().fetchone()[0] == "kmeans_model_test_drop"

        model_test.drop()
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'kmeans_model_test_drop'"
        )
        assert current_cursor().fetchone() is None

    def test_get_attr(self, model):
        m_att = model.get_attr()

        assert m_att["attr_name"] == ["centers", "metrics"]
        assert m_att["attr_fields"] == [
            "sepallengthcm, sepalwidthcm, petallengthcm, petalwidthcm",
            "metrics",
        ]
        assert m_att["#_of_rows"] == [3, 1]

        m_att_centers = model.get_attr(attr_name="centers")

        assert m_att_centers["sepallengthcm"] == [
            pytest.approx(5.006),
            pytest.approx(5.90161290322581),
            pytest.approx(6.85),
        ]

    def test_get_params(self, model):
        assert model.get_params() == {
            "max_iter": 10,
            "tol": 0.0001,
            "n_cluster": 3,
            "init": [[7.2, 3.0, 5.8, 1.6], [6.9, 3.1, 4.9, 1.5], [5.7, 4.4, 1.5, 0.4]],
        }

    def test_get_predict(self, iris_vd, model):
        iris_copy = iris_vd.copy()

        model.predict(
            iris_copy,
            X=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
            name="pred",
        )

        assert iris_copy["pred"].mode() == 1
        assert iris_copy["pred"].distinct() == [0, 1, 2]

    def test_set_params(self, model):
        model.set_params({"max_iter": 20})
        assert model.get_params()["max_iter"] == 20

    def test_model_from_vDF(self, iris_vd):
        current_cursor().execute("DROP MODEL IF EXISTS kmeans_vDF")
        model_test = KMeans("kmeans_vDF", init="random")
        model_test.fit(iris_vd, ["SepalLengthCm", "SepalWidthCm"])
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'kmeans_vDF'"
        )
        assert current_cursor().fetchone()[0] == "kmeans_vDF"
        model_test.drop()

    def test_init_method(self):
        model_test_kmeanspp = KMeans("kmeanspp_test", init="kmeanspp")
        model_test_kmeanspp.drop()
        model_test_kmeanspp.fit("public.iris")

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'kmeanspp_test'"
        )
        assert current_cursor().fetchone()[0] == "kmeanspp_test"
        model_test_kmeanspp.drop()

        model_test_random = KMeans("random_test", init="random")
        model_test_random.drop()
        model_test_random.fit("public.iris")

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'random_test'"
        )
        assert current_cursor().fetchone()[0] == "random_test"
        model_test_random.drop()

    def test_get_plot(self, winequality_vd):
        current_cursor().execute("DROP MODEL IF EXISTS model_test_plot")
        model_test = KMeans("model_test_plot",)
        model_test.fit(winequality_vd, ["alcohol", "quality"])
        result = model_test.plot(color="b")
        assert len(result.get_default_bbox_extra_artists()) == 16
        plt.close("all")
        model_test.drop()

    def test_to_python(self, model):
        current_cursor().execute(
            "SELECT APPLY_KMEANS(5.006, 3.418, 1.464, 0.244 USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model.name
            )
        )
        prediction = current_cursor().fetchone()[0]
        assert prediction == pytest.approx(
            model.to_python(return_str=False)([[5.006, 3.418, 1.464, 0.244]])[0]
        )
        assert 0.0 == pytest.approx(
            model.to_python(return_str=False, return_distance_clusters=True)(
                [[5.006, 3.418, 1.464, 0.244]]
            )[0][0]
        )

    def test_to_sql(self, model):
        current_cursor().execute(
            "SELECT APPLY_KMEANS(5.006, 3.418, 1.464, 0.244 USING PARAMETERS model_name = '{}', match_by_pos=True)::float, {}::float".format(
                model.name, model.to_sql([5.006, 3.418, 1.464, 0.244])
            )
        )
        prediction = current_cursor().fetchone()
        assert prediction[0] == pytest.approx(prediction[1])

    def test_to_memmodel(self, model, iris_vd):
        mmodel = model.to_memmodel()
        res = mmodel.predict([[5.006, 3.418, 1.464, 0.244], [3.0, 11.0, 1993.0, 0.0]])
        res_py = model.to_python()(
            [[5.006, 3.418, 1.464, 0.244], [3.0, 11.0, 1993.0, 0.0]]
        )
        assert res[0] == res_py[0]
        assert res[1] == res_py[1]
        vdf = iris_vd.copy()
        vdf["prediction_sql"] = mmodel.predict_sql(
            ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        )
        model.predict(vdf, name="prediction_vertica_sql")
        score = vdf.score("prediction_sql", "prediction_vertica_sql", "accuracy")
        assert score == pytest.approx(1.0)

    def test_get_voronoi_plot(self, iris_vd):
        current_cursor().execute("DROP MODEL IF EXISTS model_test_plot")
        model_test = KMeans("model_test_plot",)
        model_test.fit(iris_vd, ["SepalLengthCm", "SepalWidthCm"])
        result = model_test.plot_voronoi(color="b")
        assert len(result.gca().get_default_bbox_extra_artists()) == 21
        plt.close("all")
        model_test.drop()
