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
from verticapy.learn.naive_bayes import NaiveBayes, BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB
from verticapy import drop, set_option, vertica_conn
import matplotlib.pyplot as plt

set_option("print_info", False)


@pytest.fixture(scope="module")
def iris_vd(base):
    from verticapy.datasets import load_iris

    iris = load_iris(cursor=base.cursor)
    yield iris
    with warnings.catch_warnings(record=True) as w:
        drop(name="public.iris", cursor=base.cursor)

@pytest.fixture(scope="module")
def winequality_vd(base):
    from verticapy.datasets import load_winequality

    winequality = load_winequality(cursor=base.cursor)
    yield winequality
    with warnings.catch_warnings(record=True) as w:
        drop(name="public.winequality", cursor=base.cursor)

@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    with warnings.catch_warnings(record=True) as w:
        drop(name="public.titanic", cursor=base.cursor)


@pytest.fixture(scope="module")
def model(base, iris_vd):
    base.cursor.execute("DROP MODEL IF EXISTS nb_model_test")
    model_class = NaiveBayes("nb_model_test", cursor=base.cursor)
    model_class.fit(
        "public.iris",
        ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
        "Species",
    )
    yield model_class
    model_class.drop()


class TestNB:
    def test_repr(self, model):
        assert "predictor  |  type" in model.__repr__()
        model_repr = NaiveBayes("model_repr")
        model_repr.drop()
        assert model_repr.__repr__() == "<NaiveBayes>"

    def test_NB_subclasses(self, winequality_vd):
        model_test = BernoulliNB("model_test")
        assert model_test.parameters["nbtype"] == "bernoulli"
        model_test.drop()
        model_test.fit(winequality_vd, ["good"], "quality")
        md = model_test.to_sklearn()
        model_test.cursor.execute(
            "SELECT PREDICT_NAIVE_BAYES(True USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model_test.name
            )
        )
        prediction = model_test.cursor.fetchone()[0]
        assert prediction == pytest.approx(md.predict([[True]])[0][0])
        model_test.drop()
        model_test = CategoricalNB("model_test")
        assert model_test.parameters["nbtype"] == "categorical"
        model_test.drop()
        model_test.fit(winequality_vd, ["color"], "quality")
        md = model_test.to_sklearn()
        model_test.cursor.execute(
            "SELECT PREDICT_NAIVE_BAYES('red' USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model_test.name
            )
        )
        prediction = model_test.cursor.fetchone()[0]
        assert prediction == pytest.approx(md.predict([[0]])[0][0])
        model_test.drop()
        model_test = GaussianNB("model_test")
        assert model_test.parameters["nbtype"] == "gaussian"
        model_test.drop()
        model_test.fit(winequality_vd, ["residual_sugar", "alcohol",], "quality")
        md = model_test.to_sklearn()
        model_test.cursor.execute(
            "SELECT PREDICT_NAIVE_BAYES(0.0, 14.0 USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model_test.name
            )
        )
        prediction = model_test.cursor.fetchone()[0]
        assert prediction == pytest.approx(md.predict([[0.0, 14.0]])[0][0])
        model_test.drop()
        model_test = MultinomialNB("model_test")
        assert model_test.parameters["nbtype"] == "multinomial"
        model_test.drop()
        model_test.fit(winequality_vd, ["good"], "quality")
        md = model_test.to_sklearn()
        model_test.cursor.execute(
            "SELECT PREDICT_NAIVE_BAYES(0 USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model_test.name
            )
        )
        prediction = model_test.cursor.fetchone()[0]
        assert prediction == pytest.approx(md.predict([[0]])[0][0])
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
        assert cls_rep1["cutoff"][0] == pytest.approx(0.999)

        cls_rep2 = model.classification_report(cutoff=0.999).transpose()

        assert cls_rep2["cutoff"][0] == pytest.approx(0.999)

    def test_confusion_matrix(self, model):
        conf_mat1 = model.confusion_matrix()

        assert conf_mat1["Iris-setosa"] == [50, 0, 0]
        assert conf_mat1["Iris-versicolor"] == [0, 47, 3]
        assert conf_mat1["Iris-virginica"] == [0, 3, 47]

        conf_mat2 = model.confusion_matrix(cutoff=0.2)

        assert conf_mat2["Iris-setosa"] == [50, 0, 0]
        assert conf_mat2["Iris-versicolor"] == [0, 47, 3]
        assert conf_mat2["Iris-virginica"] == [0, 3, 47]

    def test_contour(self, base, titanic_vd):
        model_test = NaiveBayes("model_contour", cursor=base.cursor)
        model_test.drop()
        model_test.fit(
            titanic_vd,
            ["age", "fare",],
            "survived",
        )
        result = model_test.contour()
        assert len(result.get_default_bbox_extra_artists()) == 36
        model_test.drop()

    def test_deploySQL(self, model):
        expected_sql = 'PREDICT_NAIVE_BAYES("SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm" USING PARAMETERS model_name = \'nb_model_test\', match_by_pos = \'true\')'
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self, base):
        base.cursor.execute("DROP MODEL IF EXISTS nb_model_test_drop")
        model_test = NaiveBayes("nb_model_test_drop", cursor=base.cursor)
        model_test.fit(
            "public.iris",
            ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
            "Species",
        )

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'nb_model_test_drop'"
        )
        assert base.cursor.fetchone()[0] == "nb_model_test_drop"

        model_test.drop()
        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'nb_model_test_drop'"
        )
        assert base.cursor.fetchone() is None

    def test_lift_chart(self, model):
        lift_ch = model.lift_chart(pos_label="Iris-versicolor", nbins=1000)

        assert lift_ch["decision_boundary"][300] == pytest.approx(0.3)
        assert lift_ch["positive_prediction_ratio"][300] == pytest.approx(0.9)
        assert lift_ch["lift"][300] == pytest.approx(2.8125)
        assert lift_ch["decision_boundary"][900] == pytest.approx(0.9)
        assert lift_ch["positive_prediction_ratio"][900] == pytest.approx(0.98)
        assert lift_ch["lift"][900] == pytest.approx(2.57894736842105)
        plt.close()

    def test_to_sklearn(self, model):
        md = model.to_sklearn()
        model.cursor.execute(
            "SELECT PREDICT_NAIVE_BAYES(1.1, 2.2, 3.3, 4.4 USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model.name
            )
        )
        prediction = model.cursor.fetchone()[0]
        assert prediction == md.predict([[1.1, 2.2, 3.3, 4.4]])[0]

    def test_to_sql(self, model, titanic_vd):
        model_test = NaiveBayes("rfc_sql_test", cursor=model.cursor)
        model_test.drop()
        model_test.fit(titanic_vd, ["age", "fare", "sex", "pclass"], "survived")
        model.cursor.execute(
            "SELECT PREDICT_NAIVE_BAYES(* USING PARAMETERS model_name = 'rfc_sql_test', match_by_pos=True, class=1, type='probability')::float, {}::float FROM (SELECT 30.0 AS age, 45.0 AS fare, 'male' AS sex, 1 AS pclass) x".format(
                model_test.to_sql()
            )
        )
        prediction = model.cursor.fetchone()
        assert prediction[0] == pytest.approx(prediction[1], 1e-3)
        model_test.drop()

    def test_get_attr(self, model):
        attr = model.get_attr()
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

        details = model.get_attr("details")
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

        assert model.get_attr("alpha")["alpha"][0] == 1.0

        assert model.get_attr("prior")["class"] == [
            "Iris-setosa",
            "Iris-versicolor",
            "Iris-virginica",
        ]
        assert model.get_attr("prior")["probability"] == [
            pytest.approx(0.333333333333333),
            pytest.approx(0.333333333333333),
            pytest.approx(0.333333333333333),
        ]

        assert model.get_attr("accepted_row_count")["accepted_row_count"][0] == 150
        assert model.get_attr("rejected_row_count")["rejected_row_count"][0] == 0

        assert model.get_attr("gaussian.Iris-setosa")["mu"] == [
            pytest.approx(5.006),
            pytest.approx(3.418),
            pytest.approx(1.464),
            pytest.approx(0.244),
        ]
        assert model.get_attr("gaussian.Iris-setosa")["sigma_sq"] == [
            pytest.approx(0.12424897959183),
            pytest.approx(0.145179591836736),
            pytest.approx(0.0301061224489805),
            pytest.approx(0.0114938775510204),
        ]

        assert (
            model.get_attr("call_string")["call_string"][0]
            == "naive_bayes('public.nb_model_test', 'public.iris', '\"species\"', '\"SepalLengthCm\", \"SepalWidthCm\", \"PetalLengthCm\", \"PetalWidthCm\"' USING PARAMETERS exclude_columns='', alpha=1)"
        )

    def test_get_params(self, model):
        params = model.get_params()

        assert params == {"alpha": 1.0, "nbtype": "auto"}

    def test_prc_curve(self, model):
        prc = model.prc_curve(pos_label="Iris-virginica", nbins=1000)

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
        assert iris_copy["pred_probability"].mode() == "Iris-setosa"

        model.predict(iris_copy, name="pred_class1", cutoff=0.7)
        assert iris_copy["pred_class1"].mode() == "Iris-setosa"

        model.predict(iris_copy, name="pred_class2", cutoff=0.3)
        assert iris_copy["pred_class2"].mode() == "Iris-setosa"

    def test_roc_curve(self, model):
        roc = model.roc_curve(pos_label="Iris-virginica", nbins=1000)

        assert roc["threshold"][100] == pytest.approx(0.1)
        assert roc["false_positive"][100] == pytest.approx(0.08)
        assert roc["true_positive"][100] == pytest.approx(0.96)
        assert roc["threshold"][700] == pytest.approx(0.7)
        assert roc["false_positive"][700] == pytest.approx(0.02)
        assert roc["true_positive"][700] == pytest.approx(0.92)
        plt.close()

    def test_cutoff_curve(self, model):
        cutoff_curve = model.cutoff_curve(pos_label="Iris-virginica", nbins=1000)

        assert cutoff_curve["threshold"][100] == pytest.approx(0.1)
        assert cutoff_curve["false_positive"][100] == pytest.approx(0.08)
        assert cutoff_curve["true_positive"][100] == pytest.approx(0.96)
        assert cutoff_curve["threshold"][700] == pytest.approx(0.7)
        assert cutoff_curve["false_positive"][700] == pytest.approx(0.02)
        assert cutoff_curve["true_positive"][700] == pytest.approx(0.92)
        plt.close()

    def test_score(self, model):
        # the value of cutoff has no impact on the result
        assert model.score(cutoff=0.9, method="accuracy") == pytest.approx(0.96)
        assert model.score(cutoff=0.1, method="accuracy") == pytest.approx(0.96)
        assert model.score(
            cutoff=0.9, method="auc", pos_label="Iris-virginica"
        ) == pytest.approx(0.9923999999999998)
        assert model.score(
            cutoff=0.1, method="auc", pos_label="Iris-virginica"
        ) == pytest.approx(0.9923999999999998)
        assert model.score(
            cutoff=0.9, method="best_cutoff", pos_label="Iris-virginica"
        ) == pytest.approx(0.509)
        assert model.score(
            cutoff=0.9, method="bm", pos_label="Iris-virginica"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.9, method="csi", pos_label="Iris-virginica"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.9, method="f1", pos_label="Iris-virginica"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.9, method="logloss", pos_label="Iris-virginica"
        ) == pytest.approx(0.0479202007517544)
        assert model.score(
            cutoff=0.9, method="mcc", pos_label="Iris-virginica"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.9, method="mk", pos_label="Iris-virginica"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.9, method="npv", pos_label="Iris-virginica"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.9, method="prc_auc", pos_label="Iris-virginica"
        ) == pytest.approx(0.9864010713921592)
        assert model.score(
            cutoff=0.9, method="precision", pos_label="Iris-virginica"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.9, method="specificity", pos_label="Iris-virginica"
        ) == pytest.approx(1.0)

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
        model.set_params({"alpha": 0.5})

        assert model.get_params()["alpha"] == 0.5

    def test_model_from_vDF(self, base, iris_vd):
        base.cursor.execute("DROP MODEL IF EXISTS nb_from_vDF")
        model_test = NaiveBayes("nb_from_vDF", cursor=base.cursor)
        model_test.fit(
            iris_vd,
            ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
            "Species",
        )

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'nb_from_vDF'"
        )
        assert base.cursor.fetchone()[0] == "nb_from_vDF"

        model_test.drop()
