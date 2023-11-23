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
from verticapy import drop, set_option
from verticapy.connection import current_cursor
from verticapy.datasets import load_iris, load_winequality
from verticapy.learn.cluster import KMeans

# Matplotlib skip
import matplotlib

matplotlib_version = matplotlib.__version__
skip_plt = pytest.mark.skipif(
    matplotlib_version > "3.5.2",
    reason="Test skipped on matplotlib version greater than 3.5.2",
)

set_option("print_info", False)


@pytest.fixture(scope="module")
def iris_vd():
    iris = load_iris()
    yield iris
    drop(
        name="public.iris",
    )


@pytest.fixture(scope="module")
def winequality_vd():
    winequality = load_winequality()
    yield winequality
    drop(
        name="public.winequality",
    )


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
        assert model.__repr__() == "<KMeans>"

    def test_deploySQL(self, model):
        expected_sql = 'APPLY_KMEANS("SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm" USING PARAMETERS model_name = \'kmeans_model_test\', match_by_pos = \'true\')'
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self):
        current_cursor().execute("DROP MODEL IF EXISTS kmeans_model_test_drop")
        model_test = KMeans(
            "kmeans_model_test_drop",
        )
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

    def test_get_vertica_attributes(self, model):
        m_att = model.get_vertica_attributes()

        assert m_att["attr_name"] == ["centers", "metrics"]
        assert m_att["attr_fields"] == [
            "sepallengthcm, sepalwidthcm, petallengthcm, petalwidthcm",
            "metrics",
        ]
        assert m_att["#_of_rows"] == [3, 1]

        m_att_centers = model.get_vertica_attributes(attr_name="centers")

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
            "init": [
                [7.2, 3.0, 5.8, 1.6],
                [6.9, 3.1, 4.9, 1.5],
                [5.7, 4.4, 1.5, 0.4],
            ],
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

    @skip_plt
    def test_get_plot(self, winequality_vd):
        current_cursor().execute("DROP MODEL IF EXISTS model_test_plot")
        model_test = KMeans(
            "model_test_plot",
        )
        model_test.fit(winequality_vd, ["alcohol", "quality"])
        result = model_test.plot(color="b")
        assert len(result.get_default_bbox_extra_artists()) > 8
        plt.close("all")
        model_test.drop()

    def test_to_python(self, model):
        current_cursor().execute(
            "SELECT APPLY_KMEANS(5.006, 3.418, 1.464, 0.244 USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model.model_name
            )
        )
        prediction = current_cursor().fetchone()[0]
        assert prediction == pytest.approx(
            model.to_python()([[5.006, 3.418, 1.464, 0.244]])[0]
        )
        assert 0.0 == pytest.approx(
            model.to_python(return_distance_clusters=True)(
                [[5.006, 3.418, 1.464, 0.244]]
            )[0][0]
        )

    def test_to_sql(self, model):
        current_cursor().execute(
            "SELECT APPLY_KMEANS(5.006, 3.418, 1.464, 0.244 USING PARAMETERS model_name = '{}', match_by_pos=True)::float, {}::float".format(
                model.model_name, model.to_sql([5.006, 3.418, 1.464, 0.244])
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
        score = vdf.score("prediction_sql", "prediction_vertica_sql", metric="accuracy")
        assert score == pytest.approx(1.0)

    @skip_plt
    def test_get_voronoi_plot(self, iris_vd):
        current_cursor().execute("DROP MODEL IF EXISTS model_test_plot")
        model_test = KMeans(
            "model_test_plot",
        )
        model_test.fit(iris_vd, ["SepalLengthCm", "SepalWidthCm"])
        result = model_test.plot_voronoi(color="b")
        assert len(result.gca().get_default_bbox_extra_artists()) == 21
        plt.close("all")
        model_test.drop()

    def test_overwrite_model(self, iris_vd):
        model = KMeans("test_overwrite_model")
        model.drop()  # to isulate this test from any previous left over
        model.fit(iris_vd, ["SepalLengthCm", "SepalWidthCm"])

        # overwrite_model is false by default
        with pytest.raises(NameError) as exception_info:
            model.fit(iris_vd, ["SepalLengthCm", "SepalWidthCm"])
        assert exception_info.match("The model 'test_overwrite_model' already exists!")

        # overwriting the model when overwrite_model is specified true
        model = KMeans("test_overwrite_model", overwrite_model=True)
        model.fit(iris_vd, ["SepalLengthCm", "SepalWidthCm"])

        # cleaning up
        model.drop()
