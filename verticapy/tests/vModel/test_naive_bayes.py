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
from verticapy.datasets import load_winequality, load_titanic, load_iris
from verticapy.learn.naive_bayes import *

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
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(
        name="public.titanic",
    )


@pytest.fixture(scope="module")
def model(iris_vd):
    model_class = NaiveBayes(
        "nb_model_test",
    )
    model_class.drop()
    model_class.fit(
        "public.iris",
        ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
        "Species",
    )
    yield model_class
    model_class.drop()


class TestNB:
    def test_repr(self, model):
        assert model.__repr__() == "<NaiveBayes>"

    def test_NB_subclasses(self, winequality_vd):
        model_test = BernoulliNB("model_test")
        assert model_test.parameters["nbtype"] == "bernoulli"
        model_test.drop()
        model_test.fit(winequality_vd, ["good"], "quality")
        model_test.drop()
        model_test = CategoricalNB("model_test")
        assert model_test.parameters["nbtype"] == "categorical"
        model_test.drop()
        model_test.fit(winequality_vd, ["color"], "quality")
        model_test.drop()
        model_test = GaussianNB("model_test")
        assert model_test.parameters["nbtype"] == "gaussian"
        model_test.drop()
        model_test.fit(winequality_vd, ["residual_sugar", "alcohol"], "quality")
        model_test.drop()
        model_test = MultinomialNB("model_test")
        assert model_test.parameters["nbtype"] == "multinomial"
        model_test.drop()
        model_test.fit(winequality_vd, ["good"], "quality")
        model_test.drop()

    def test_classification_report(self, model):
        cls_rep1 = model.classification_report().transpose()

        assert cls_rep1["auc"][0] == pytest.approx(1.0)
        assert cls_rep1["prc_auc"][0] == pytest.approx(1.0)
        assert cls_rep1["accuracy"][0] == pytest.approx(1.0)
        assert cls_rep1["log_loss"][0] == pytest.approx(0.0)
        assert cls_rep1["precision"][0] == pytest.approx(1.0)
        assert cls_rep1["recall"][0] == pytest.approx(1.0)
        assert cls_rep1["f1_score"][0] == pytest.approx(1.0)
        assert cls_rep1["mcc"][0] == pytest.approx(1.0)
        assert cls_rep1["informedness"][0] == pytest.approx(1.0)
        assert cls_rep1["markedness"][0] == pytest.approx(1.0)
        assert cls_rep1["csi"][0] == pytest.approx(1.0)

    def test_confusion_matrix(self, model):
        conf_mat1 = model.confusion_matrix()

        assert list(conf_mat1[:, 0]) == [50, 0, 0]
        assert list(conf_mat1[:, 1]) == [0, 47, 3]
        assert list(conf_mat1[:, 2]) == [0, 3, 47]

        conf_mat2 = model.confusion_matrix(cutoff=0.2)

        assert list(conf_mat1[:, 0]) == [50, 0, 0]
        assert list(conf_mat1[:, 1]) == [0, 47, 3]
        assert list(conf_mat1[:, 2]) == [0, 3, 47]

    @skip_plt
    def test_contour(self, titanic_vd):
        model_test = NaiveBayes(
            "model_contour",
        )
        model_test.drop()
        model_test.fit(
            titanic_vd,
            ["age", "fare"],
            "survived",
        )
        result = model_test.contour()
        assert len(result.get_default_bbox_extra_artists()) == 36
        model_test.drop()

    def test_deploySQL(self, model):
        expected_sql = 'PREDICT_NAIVE_BAYES("SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm" USING PARAMETERS model_name = \'nb_model_test\', match_by_pos = \'true\')'
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self):
        current_cursor().execute("DROP MODEL IF EXISTS nb_model_test_drop")
        model_test = NaiveBayes(
            "nb_model_test_drop",
        )
        model_test.fit(
            "public.iris",
            ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
            "Species",
        )

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'nb_model_test_drop'"
        )
        assert current_cursor().fetchone()[0] == "nb_model_test_drop"

        model_test.drop()
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'nb_model_test_drop'"
        )
        assert current_cursor().fetchone() is None

    def test_lift_chart(self, model):
        lift_ch = model.lift_chart(pos_label="Iris-versicolor", nbins=1000, show=False)

        assert lift_ch["decision_boundary"][300] == pytest.approx(0.3)
        assert lift_ch["positive_prediction_ratio"][300] == pytest.approx(0.9)
        assert lift_ch["lift"][300] == pytest.approx(2.8125)
        assert lift_ch["decision_boundary"][900] == pytest.approx(0.9)
        assert lift_ch["positive_prediction_ratio"][900] == pytest.approx(0.98)
        assert lift_ch["lift"][900] == pytest.approx(2.57894736842105)
        plt.close()

    def test_to_python(self, titanic_vd):
        titanic = titanic_vd.copy()
        titanic["has_children"] = "parch > 0"
        model_class = NaiveBayes(
            "nb_model_test_to_python",
        )
        model_class.drop()
        model_class.fit(
            titanic,
            ["age", "fare", "survived", "pclass", "sex", "has_children"],
            "embarked",
        )
        predict_function = model_class.to_python()
        current_cursor().execute(
            "SELECT PREDICT_NAIVE_BAYES(30.0, 200.0, 1, 2, 'female', True USING PARAMETERS model_name = 'nb_model_test_to_python', match_by_pos=True)"
        )
        prediction = current_cursor().fetchone()[0]
        assert prediction == predict_function([[30, 200, 1, 2, "female", True]])[0]
        predict_function = model_class.to_python(return_proba=True)
        current_cursor().execute(
            "SELECT PREDICT_NAIVE_BAYES(30.0, 200.0, 1, 2, 'female', True USING PARAMETERS model_name = 'nb_model_test_to_python', match_by_pos=True, type='probability', class='{}')".format(
                model_class.classes_[0]
            )
        )
        prediction = current_cursor().fetchone()[0]
        assert float(prediction) == pytest.approx(
            float(predict_function([[30, 200, 1, 2, "female", True]])[0][0])
        )
        model_class.drop()

    def test_to_sql(self, model, titanic_vd):
        model_test = NaiveBayes("rfc_sql_test")
        model_test.drop()
        model_test.fit(titanic_vd, ["age", "fare", "sex", "pclass"], "survived")
        current_cursor().execute(
            "SELECT PREDICT_NAIVE_BAYES(* USING PARAMETERS model_name = 'rfc_sql_test', match_by_pos=True)::int, {}::int FROM (SELECT 30.0 AS age, 45.0 AS fare, 'male' AS sex, 1 AS pclass) x".format(
                model_test.to_sql()
            )
        )
        prediction = current_cursor().fetchone()
        assert prediction[0] == pytest.approx(prediction[1], 1e-3)
        model_test.drop()

    def test_to_memmodel(self, titanic_vd):
        titanic = titanic_vd.copy()
        titanic["has_children"] = "parch > 0"
        model_class = NaiveBayes(
            "nb_model_test_to_memmodel",
        )
        model_class.drop()
        model_class.fit(
            titanic,
            ["age", "fare", "survived", "pclass", "sex", "has_children"],
            "embarked",
        )
        mmodel = model_class.to_memmodel()
        res = mmodel.predict(
            [
                [11.0, 1993.0, 1, 3, "male", False],
                [1.0, 1999.0, 1, 1, "female", True],
            ]
        )
        res_py = model_class.to_python()(
            [
                [11.0, 1993.0, 1, 3, "male", False],
                [1.0, 1999.0, 1, 1, "female", True],
            ]
        )
        assert res[0] == res_py[0]
        assert res[1] == res_py[1]
        res = mmodel.predict_proba(
            [
                [11.0, 1993.0, 1, 3, "male", False],
                [1.0, 1999.0, 1, 1, "female", True],
            ]
        )
        res_py = model_class.to_python(return_proba=True)(
            [
                [11.0, 1993.0, 1, 3, "male", False],
                [1.0, 1999.0, 1, 1, "female", True],
            ]
        )
        assert res[0][0] == res_py[0][0]
        assert res[0][1] == res_py[0][1]
        assert res[0][2] == res_py[0][2]
        assert res[1][0] == res_py[1][0]
        assert res[1][1] == res_py[1][1]
        assert res[1][2] == res_py[1][2]
        titanic["prediction_sql"] = mmodel.predict_sql(
            ["age", "fare", "survived", "pclass", "sex", "has_children"]
        )
        titanic["prediction_proba_sql_0"] = mmodel.predict_proba_sql(
            ["age", "fare", "survived", "pclass", "sex", "has_children"]
        )[0]
        titanic["prediction_proba_sql_1"] = mmodel.predict_proba_sql(
            ["age", "fare", "survived", "pclass", "sex", "has_children"]
        )[1]
        titanic["prediction_proba_sql_2"] = mmodel.predict_proba_sql(
            ["age", "fare", "survived", "pclass", "sex", "has_children"]
        )[2]
        model_class.predict(titanic, name="prediction_vertica_sql")
        model_class.predict_proba(
            titanic,
            name="prediction_proba_vertica_sql_0",
            pos_label=model_class.classes_[0],
        )
        model_class.predict_proba(
            titanic,
            name="prediction_proba_vertica_sql_1",
            pos_label=model_class.classes_[1],
        )
        model_class.predict_proba(
            titanic,
            name="prediction_proba_vertica_sql_2",
            pos_label=model_class.classes_[2],
        )
        score = titanic.score(
            "prediction_sql", "prediction_vertica_sql", metric="accuracy"
        )
        assert score == pytest.approx(1.0)
        score = titanic.score(
            "prediction_proba_sql_0", "prediction_proba_vertica_sql_0", metric="r2"
        )
        assert score == pytest.approx(1.0)
        score = titanic.score(
            "prediction_proba_sql_1", "prediction_proba_vertica_sql_1", metric="r2"
        )
        assert score == pytest.approx(1.0)
        score = titanic.score(
            "prediction_proba_sql_2", "prediction_proba_vertica_sql_2", metric="r2"
        )
        assert score == pytest.approx(1.0)

    def test_get_vertica_attributes(self, model):
        attr = model.get_vertica_attributes()
        assert attr["attr_name"] == [
            "details",
            "alpha",
            "prior",
            "accepted_row_count",
            "rejected_row_count",
            "call_string",
            "gaussian.Iris-setosa",
            "gaussian.Iris-versicolor",
            "gaussian.Iris-virginica",
        ]
        assert attr["attr_fields"] == [
            "index, predictor, type",
            "alpha",
            "class, probability",
            "accepted_row_count",
            "rejected_row_count",
            "call_string",
            "index, mu, sigma_sq",
            "index, mu, sigma_sq",
            "index, mu, sigma_sq",
        ]
        assert attr["#_of_rows"] == [5, 1, 3, 1, 1, 1, 4, 4, 4]

        details = model.get_vertica_attributes("details")
        assert details["predictor"] == [
            "Species",
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm",
        ]
        assert details["type"] == [
            "ResponseC",
            "Gaussian",
            "Gaussian",
            "Gaussian",
            "Gaussian",
        ]

        assert model.get_vertica_attributes("alpha")["alpha"][0] == 1.0

        assert model.get_vertica_attributes("prior")["class"] == [
            "Iris-setosa",
            "Iris-versicolor",
            "Iris-virginica",
        ]
        assert model.get_vertica_attributes("prior")["probability"] == [
            pytest.approx(0.333333333333333),
            pytest.approx(0.333333333333333),
            pytest.approx(0.333333333333333),
        ]

        assert (
            model.get_vertica_attributes("accepted_row_count")["accepted_row_count"][0]
            == 150
        )
        assert (
            model.get_vertica_attributes("rejected_row_count")["rejected_row_count"][0]
            == 0
        )

        assert model.get_vertica_attributes("gaussian.Iris-setosa")["mu"] == [
            pytest.approx(5.006),
            pytest.approx(3.418),
            pytest.approx(1.464),
            pytest.approx(0.244),
        ]
        assert model.get_vertica_attributes("gaussian.Iris-setosa")["sigma_sq"] == [
            pytest.approx(0.12424897959183),
            pytest.approx(0.145179591836736),
            pytest.approx(0.0301061224489805),
            pytest.approx(0.0114938775510204),
        ]

        assert (
            model.get_vertica_attributes("call_string")["call_string"][0]
            == "naive_bayes('public.nb_model_test', 'public.iris', '\"species\"', '\"SepalLengthCm\", \"SepalWidthCm\", \"PetalLengthCm\", \"PetalWidthCm\"' USING PARAMETERS exclude_columns='', alpha=1)"
        )

    def test_get_params(self, model):
        params = model.get_params()

        assert params == {"alpha": 1.0, "nbtype": "auto"}

    def test_prc_curve(self, model):
        prc = model.prc_curve(pos_label="Iris-virginica", nbins=1000, show=False)

        assert prc["threshold"][300] == pytest.approx(0.299)
        assert prc["recall"][300] == pytest.approx(0.94)
        assert prc["precision"][300] == pytest.approx(0.903846153846154)
        assert prc["threshold"][800] == pytest.approx(0.799)
        assert prc["recall"][800] == pytest.approx(0.9)
        assert prc["precision"][800] == pytest.approx(0.957446808510638)
        plt.close()

    def test_predict(self, iris_vd, model):
        iris_copy = iris_vd.copy()

        model.predict(iris_copy, name="pred_probability")
        assert iris_copy["pred_probability"][0] == "Iris-setosa"

        model.predict(iris_copy, name="pred_class1", cutoff=0.7)
        assert iris_copy["pred_class1"][0] == "Iris-setosa"

        model.predict(iris_copy, name="pred_class2", cutoff=0.3)
        assert iris_copy["pred_class2"][0] == "Iris-setosa"

    def test_roc_curve(self, model):
        roc = model.roc_curve(pos_label="Iris-virginica", nbins=1000, show=False)

        assert roc["threshold"][100] == pytest.approx(0.1)
        assert roc["false_positive"][100] == pytest.approx(0.08)
        assert roc["true_positive"][100] == pytest.approx(0.96)
        assert roc["threshold"][700] == pytest.approx(0.7)
        assert roc["false_positive"][700] == pytest.approx(0.02)
        assert roc["true_positive"][700] == pytest.approx(0.92)
        plt.close()

    def test_cutoff_curve(self, model):
        cutoff_curve = model.cutoff_curve(
            pos_label="Iris-virginica", nbins=1000, show=False
        )

        assert cutoff_curve["threshold"][100] == pytest.approx(0.1)
        assert cutoff_curve["false_positive"][100] == pytest.approx(0.08)
        assert cutoff_curve["true_positive"][100] == pytest.approx(0.96)
        assert cutoff_curve["threshold"][700] == pytest.approx(0.7)
        assert cutoff_curve["false_positive"][700] == pytest.approx(0.02)
        assert cutoff_curve["true_positive"][700] == pytest.approx(0.92)
        plt.close()

    def test_score(self, model):
        # the value of cutoff has no impact on the result
        assert model.score(metric="accuracy") == pytest.approx(0.96)
        assert model.score(
            cutoff=0.9, metric="auc", pos_label="Iris-virginica"
        ) == pytest.approx(0.9923999999999998)
        assert model.score(
            cutoff=0.1, metric="auc", pos_label="Iris-virginica"
        ) == pytest.approx(0.9923999999999998)
        assert model.score(
            cutoff=0.9, metric="best_cutoff", pos_label="Iris-virginica"
        ) == pytest.approx(0.5099, 1e-2)
        assert model.score(
            cutoff=0.9, metric="bm", pos_label="Iris-virginica"
        ) == pytest.approx(0.8300000000000001)
        assert model.score(
            cutoff=0.9, metric="csi", pos_label="Iris-virginica"
        ) == pytest.approx(0.8235294117647058)
        assert model.score(
            cutoff=0.9, metric="f1", pos_label="Iris-virginica"
        ) == pytest.approx(0.9032258064516129)
        assert model.score(
            cutoff=0.9, metric="logloss", pos_label="Iris-virginica"
        ) == pytest.approx(0.0479202007517544)
        assert model.score(
            cutoff=0.9, metric="mcc", pos_label="Iris-virginica"
        ) == pytest.approx(0.8652407755372198)
        assert model.score(
            cutoff=0.9, metric="mk", pos_label="Iris-virginica"
        ) == pytest.approx(0.9019778309063247)
        assert model.score(
            cutoff=0.9, metric="npv", pos_label="Iris-virginica"
        ) == pytest.approx(0.9252336448598131)
        assert model.score(
            cutoff=0.9, metric="prc_auc", pos_label="Iris-virginica"
        ) == pytest.approx(0.9864010713921592)
        assert model.score(
            cutoff=0.9, metric="precision", pos_label="Iris-virginica"
        ) == pytest.approx(0.9767441860465116)
        assert model.score(
            cutoff=0.9, metric="specificity", pos_label="Iris-virginica"
        ) == pytest.approx(0.99)

    def test_set_params(self, model):
        model.set_params({"alpha": 0.5})

        assert model.get_params()["alpha"] == 0.5

    def test_model_from_vDF(self, iris_vd):
        current_cursor().execute("DROP MODEL IF EXISTS nb_from_vDF")
        model_test = NaiveBayes(
            "nb_from_vDF",
        )
        model_test.fit(
            iris_vd,
            ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
            "Species",
        )

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'nb_from_vDF'"
        )
        assert current_cursor().fetchone()[0] == "nb_from_vDF"

        model_test.drop()

    def test_optional_name(self):
        model = NaiveBayes()
        assert model.model_name is not None
