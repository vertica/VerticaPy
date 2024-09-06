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
from verticapy.tests.conftest import get_version
from verticapy import drop, set_option
from verticapy.connection import current_cursor
from verticapy.datasets import load_iris
from verticapy.learn.cluster import KPrototypes

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
def model(iris_vd):
    model_class = KPrototypes(
        "kprototypes_model_test",
        n_cluster=3,
        max_iter=10,
        init=[
            [7.2, 3.0, 5.8, 1.6, "Iris-setosa"],
            [6.9, 3.1, 4.9, 1.5, "Iris-versicolor"],
            [5.7, 4.4, 1.5, 0.4, "Iris-virginica"],
        ],
    )
    model_class.drop()
    model_class.fit(iris_vd)
    yield model_class
    model_class.drop()


version = get_version()


@pytest.mark.skipif(
    version[0] < 12 or (version[0] == 12 and version[1] == 0 and version[2] < 3),
    reason="requires vertica 12.0.3 or higher",
)
class TestKPrototypes:
    def test_repr(self, model):
        assert model.__repr__() == "<KPrototypes>"

    def test_deploySQL(self, model):
        expected_sql = 'APPLY_KPROTOTYPES("SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species" USING PARAMETERS model_name = \'kprototypes_model_test\', match_by_pos = \'true\')'
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self):
        current_cursor().execute("DROP MODEL IF EXISTS kprototypes_model_test_drop")
        model_test = KPrototypes("kprototypes_model_test_drop", n_cluster=3)
        model_test.fit("public.iris", ["SepalLengthCm", "SepalWidthCm", "Species"])

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'kprototypes_model_test_drop'"
        )
        assert current_cursor().fetchone()[0] == "kprototypes_model_test_drop"

        model_test.drop()
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'kprototypes_model_test_drop'"
        )
        assert current_cursor().fetchone() is None

    def test_get_vertica_attributes(self, model):
        m_att = model.get_vertica_attributes()

        assert m_att["attr_name"] == ["centers", "metrics"]
        assert m_att["attr_fields"] == [
            "sepallengthcm, sepalwidthcm, petallengthcm, petalwidthcm, species",
            "metrics",
        ]
        assert m_att["#_of_rows"] == [3, 1]

        m_att_centers = model.get_vertica_attributes(attr_name="centers")

        assert m_att_centers["sepallengthcm"] == [
            pytest.approx(5.006),
            pytest.approx(5.9156862745098),
            pytest.approx(6.62244897959184),
        ]

    def test_get_params(self, model):
        assert model.get_params() == {
            "max_iter": 10,
            "tol": 0.0001,
            "n_cluster": 3,
            "init": [
                [7.2, 3.0, 5.8, 1.6, "Iris-setosa"],
                [6.9, 3.1, 4.9, 1.5, "Iris-versicolor"],
                [5.7, 4.4, 1.5, 0.4, "Iris-virginica"],
            ],
            "gamma": 1.0,
        }

    def test_get_predict(self, iris_vd, model):
        iris_copy = iris_vd.copy()

        model.predict(
            iris_copy,
            X=[
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm",
                "Species",
            ],
            name="pred",
        )

        assert iris_copy["pred"].mode() == 1
        assert iris_copy["pred"].distinct() == [0, 1, 2]

    def test_set_params(self, model):
        model.set_params({"max_iter": 20})
        assert model.get_params()["max_iter"] == 20

    def test_model_from_vDF(self, iris_vd):
        current_cursor().execute("DROP MODEL IF EXISTS kprototypes_vDF")
        model_test = KPrototypes("kprototypes_vDF", n_cluster=3, init="random")
        model_test.fit(iris_vd, ["SepalLengthCm", "SepalWidthCm", "Species"])
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'kprototypes_vDF'"
        )
        assert current_cursor().fetchone()[0] == "kprototypes_vDF"
        model_test.drop()

    def test_init_method(self):
        model_test_random = KPrototypes("random_test", n_cluster=3, init="random")
        model_test_random.drop()
        model_test_random.fit("public.iris")

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'random_test'"
        )
        assert current_cursor().fetchone()[0] == "random_test"
        model_test_random.drop()

    @skip_plt
    def test_get_plot(self, iris_vd):
        current_cursor().execute("DROP MODEL IF EXISTS model_test_plot")
        model_test = KPrototypes(
            "model_test_plot",
            n_cluster=3,
        )
        model_test.fit(iris_vd, ["SepalLengthCm", "PetalWidthCm"])
        result = model_test.plot(color="b")
        assert len(result.get_default_bbox_extra_artists()) > 8
        plt.close("all")
        model_test.drop()
        # TODO: test for categorical inputs.

    def test_to_python(self, model):
        current_cursor().execute(
            "SELECT APPLY_KPROTOTYPES(5.006, 3.418, 1.464, 0.244, 'Iris-setosa' USING PARAMETERS model_name = '{0}', match_by_pos=True)".format(
                model.model_name
            )
        )
        prediction = current_cursor().fetchone()[0]
        assert prediction == pytest.approx(
            model.to_python()([[5.006, 3.418, 1.464, 0.244, "Iris-setosa"]])[0]
        )
        assert 0.0 == pytest.approx(
            model.to_python(return_distance_clusters=True)(
                [[5.006, 3.418, 1.464, 0.244]]
            )[0][0]
        )

    def test_to_sql(self, model):
        current_cursor().execute(
            "SELECT APPLY_KPROTOTYPES(5.006, 3.418, 1.464, 0.244, 'Iris-setosa' USING PARAMETERS model_name = '{0}', match_by_pos=True)::float, {1}::float".format(
                model.model_name,
                model.to_sql([5.006, 3.418, 1.464, 0.244, "'Iris-setosa'"]),
            )
        )
        prediction = current_cursor().fetchone()
        assert prediction[0] == pytest.approx(prediction[1])

    def test_to_memmodel(self, model, iris_vd):
        mmodel = model.to_memmodel()
        res = mmodel.predict(
            [
                [5.006, 3.418, 1.464, 0.244, "Iris-setosa"],
                [3.0, 11.0, 1993.0, 0.0, "Iris-setosa"],
            ]
        )
        res_py = model.to_python()(
            [
                [5.006, 3.418, 1.464, 0.244, "Iris-setosa"],
                [3.0, 11.0, 1993.0, 0.0, "Iris-setosa"],
            ]
        )
        assert res[0] == res_py[0]
        assert res[1] == res_py[1]
        vdf = iris_vd.copy()
        vdf["prediction_sql"] = mmodel.predict_sql(
            [
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm",
                "Species",
            ]
        )
        model.predict(vdf, name="prediction_vertica_sql")
        score = vdf.score("prediction_sql", "prediction_vertica_sql", metric="accuracy")
        assert score == pytest.approx(0.993333333333333)  # can we do better?

    def test_optional_name(self):
        model = KPrototypes()
        assert model.model_name is not None
