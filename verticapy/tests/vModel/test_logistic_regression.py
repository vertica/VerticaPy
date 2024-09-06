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
from verticapy.datasets import load_winequality, load_titanic
from verticapy.learn.linear_model import LogisticRegression

# Matplotlib skip
import matplotlib

matplotlib_version = matplotlib.__version__
skip_plt = pytest.mark.skipif(
    matplotlib_version > "3.5.2",
    reason="Test skipped on matplotlib version greater than 3.5.2",
)

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(
        name="public.titanic",
    )


@pytest.fixture(scope="module")
def winequality_vd():
    winequality = load_winequality()
    yield winequality
    drop(
        name="public.winequality",
    )


@pytest.fixture(scope="module")
def model(titanic_vd):
    model_class = LogisticRegression(
        "logreg_model_test",
    )
    model_class.drop()
    model_class.fit("public.titanic", ["age", "fare"], "survived")
    yield model_class
    model_class.drop()


class TestLogisticRegression:
    def test_repr(self, model):
        assert model.__repr__() == "<LogisticRegression>"

    def test_classification_report(self, model):
        cls_rep1 = model.classification_report().transpose()

        assert cls_rep1["auc"][0] == pytest.approx(0.6941239880788826)
        assert cls_rep1["prc_auc"][0] == pytest.approx(0.5979751713359676)
        assert cls_rep1["accuracy"][0] == pytest.approx(0.6586345381526104)
        assert cls_rep1["log_loss"][0] == pytest.approx(0.271495668573431)
        assert cls_rep1["precision"][0] == pytest.approx(0.6758620689655173)
        assert cls_rep1["recall"][0] == pytest.approx(0.2506393861892583)
        assert cls_rep1["f1_score"][0] == pytest.approx(0.3656716417910448)
        assert cls_rep1["mcc"][0] == pytest.approx(0.2394674439996513)
        assert cls_rep1["informedness"][0] == pytest.approx(0.17295343577603517)
        assert cls_rep1["markedness"][0] == pytest.approx(0.3315612464038251)
        assert cls_rep1["csi"][0] == pytest.approx(0.2237442922374429)

    def test_confusion_matrix(self, model):
        conf_mat1 = model.confusion_matrix()

        assert conf_mat1[0][0] == 558
        assert conf_mat1[1][0] == 293
        assert conf_mat1[0][1] == 47
        assert conf_mat1[1][1] == 98

        conf_mat2 = model.confusion_matrix(cutoff=0.2)

        assert conf_mat2[0][0] == 3
        assert conf_mat2[1][0] == 0
        assert conf_mat2[0][1] == 602
        assert conf_mat2[1][1] == 391

    @skip_plt
    def test_contour(self, titanic_vd):
        model_test = LogisticRegression(
            "model_contour",
        )
        model_test.drop()
        model_test.fit(
            titanic_vd,
            ["age", "fare"],
            "survived",
        )
        result = model_test.contour()
        assert len(result.get_default_bbox_extra_artists()) == 38
        model_test.drop()

    def test_deploySQL(self, model):
        expected_sql = "PREDICT_LOGISTIC_REG(\"age\", \"fare\" USING PARAMETERS model_name = 'logreg_model_test', type = 'probability', match_by_pos = 'true')"
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self):
        current_cursor().execute("DROP MODEL IF EXISTS logreg_model_test_drop")
        model_test = LogisticRegression(
            "logreg_model_test_drop",
        )
        model_test.fit("public.titanic", ["age", "fare"], "survived")

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'logreg_model_test_drop'"
        )
        assert current_cursor().fetchone()[0] == "logreg_model_test_drop"

        model_test.drop()
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'logreg_model_test_drop'"
        )
        assert current_cursor().fetchone() is None

    def test_features_importance(self, model):
        f_imp = model.features_importance(show=False)

        assert f_imp["index"] == ["fare", "age"]
        assert f_imp["importance"] == [85.51, 14.49]
        assert f_imp["sign"] == [1, -1]
        plt.close("all")

    def test_lift_chart(self, model):
        lift_ch = model.lift_chart(nbins=1000, show=False)

        assert lift_ch["decision_boundary"][10] == pytest.approx(0.01)
        assert lift_ch["positive_prediction_ratio"][10] == pytest.approx(
            0.010230179028133
        )
        assert lift_ch["lift"][10] == pytest.approx(2.54731457800512)
        assert lift_ch["decision_boundary"][900] == pytest.approx(0.9)
        assert lift_ch["positive_prediction_ratio"][900] == pytest.approx(1.0)
        assert lift_ch["lift"][900] == pytest.approx(1.0)
        plt.close("all")

    @skip_plt
    def test_get_plot(self, winequality_vd):
        # 1D
        current_cursor().execute("DROP MODEL IF EXISTS model_test_plot")
        model_test = LogisticRegression(
            "model_test_plot",
        )
        model_test.fit(winequality_vd, ["alcohol"], "good")
        result = model_test.plot(color="r")
        assert len(result.get_default_bbox_extra_artists()) == 11
        plt.close("all")
        model_test.drop()
        # 2D
        model_test.fit(winequality_vd, ["alcohol", "residual_sugar"], "good")
        result = model_test.plot(color="r")
        assert len(result.get_default_bbox_extra_artists()) == 5
        plt.close("all")
        model_test.drop()

    def test_to_python(self, model):
        current_cursor().execute(
            "SELECT PREDICT_LOGISTIC_REG(3.0, 11.0 USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model.model_name
            )
        )
        prediction = current_cursor().fetchone()[0]
        assert prediction == pytest.approx(model.to_python()([[3.0, 11.0]])[0])
        current_cursor().execute(
            "SELECT PREDICT_LOGISTIC_REG(3.0, 11.0 USING PARAMETERS model_name = '{}', type='probability', match_by_pos=True)".format(
                model.model_name
            )
        )
        prediction = current_cursor().fetchone()[0]
        assert prediction == pytest.approx(
            model.to_python(
                return_proba=True,
            )([[3.0, 11.0]])[
                0
            ][1]
        )

    def test_to_sql(self, model):
        current_cursor().execute(
            "SELECT PREDICT_LOGISTIC_REG(3.0, 11.0 USING PARAMETERS model_name = '{}', match_by_pos=True)::float, {}::float".format(
                model.model_name, model.to_sql([3.0, 11.0])
            )
        )
        prediction = current_cursor().fetchone()
        assert prediction[0] == pytest.approx(prediction[1])

    def test_to_memmodel(self, model, titanic_vd):
        mmodel = model.to_memmodel()
        res = mmodel.predict([[3.0, 11.0], [11.0, 1.0]])
        res_py = model.to_python()([[3.0, 11.0], [11.0, 1.0]])
        assert res[0] == res_py[0]
        assert res[1] == res_py[1]
        res = mmodel.predict_proba([[3.0, 11.0], [11.0, 1.0]])
        res_py = model.to_python(return_proba=True)([[3.0, 11.0], [11.0, 1.0]])
        assert res[0][0] == res_py[0][0]
        assert res[0][1] == res_py[0][1]
        assert res[1][0] == res_py[1][0]
        assert res[1][1] == res_py[1][1]
        vdf = titanic_vd.copy()
        vdf["prediction_sql"] = mmodel.predict_sql(["age", "fare"])
        vdf["prediction_proba_sql_0"] = mmodel.predict_proba_sql(["age", "fare"])[0]
        vdf["prediction_proba_sql_1"] = mmodel.predict_proba_sql(["age", "fare"])[1]
        model.predict(vdf, name="prediction_vertica_sql", cutoff=0.5)
        model.predict_proba(vdf, pos_label=0, name="prediction_proba_vertica_sql_0")
        model.predict_proba(vdf, pos_label=1, name="prediction_proba_vertica_sql_1")
        score = vdf.score("prediction_sql", "prediction_vertica_sql", metric="accuracy")
        assert score == pytest.approx(1.0)
        score = vdf.score(
            "prediction_proba_sql_0", "prediction_proba_vertica_sql_0", metric="r2"
        )
        assert score == pytest.approx(1.0)
        score = vdf.score(
            "prediction_proba_sql_1", "prediction_proba_vertica_sql_1", metric="r2"
        )
        assert score == pytest.approx(1.0)

    def test_get_vertica_attributes(self, model):
        attr = model.get_vertica_attributes()
        assert attr["attr_name"] == [
            "details",
            "regularization",
            "iteration_count",
            "rejected_row_count",
            "accepted_row_count",
            "call_string",
        ]
        assert attr["attr_fields"] == [
            "predictor, coefficient, std_err, z_value, p_value",
            "type, lambda",
            "iteration_count",
            "rejected_row_count",
            "accepted_row_count",
            "call_string",
        ]
        assert attr["#_of_rows"] == [3, 1, 1, 1, 1, 1]

        details = model.get_vertica_attributes("details")
        assert details["predictor"] == ["Intercept", "age", "fare"]
        assert details["coefficient"][0] == pytest.approx(-0.477190254617772)
        assert details["coefficient"][1] == pytest.approx(-0.0152670631243078)
        assert details["coefficient"][2] == pytest.approx(0.0140086238717347)
        assert details["std_err"][0] == pytest.approx(0.157607831241612)
        assert details["std_err"][1] == pytest.approx(0.00487661936756958)
        assert details["std_err"][2] == pytest.approx(0.00183286928098778)
        assert details["z_value"][0] == pytest.approx(-3.02770649693316)
        assert details["z_value"][1] == pytest.approx(-3.13066531824005)
        assert details["z_value"][2] == pytest.approx(7.64300215898927)
        assert details["p_value"][0] == pytest.approx(0.00246417291934784)
        assert details["p_value"][1] == pytest.approx(0.00174410802172094)
        assert details["p_value"][2] == pytest.approx(2.99885239324552e-13)

        reg = model.get_vertica_attributes("regularization")
        assert reg["type"][0] == "none"
        assert reg["lambda"][0] == 1.0

        assert (
            model.get_vertica_attributes("iteration_count")["iteration_count"][0] == 4
        )
        assert (
            model.get_vertica_attributes("rejected_row_count")["rejected_row_count"][0]
            == 238
        )
        assert (
            model.get_vertica_attributes("accepted_row_count")["accepted_row_count"][0]
            == 996
        )

        if get_version()[0] < 12:
            assert (
                model.get_vertica_attributes("call_string")["call_string"][0]
                == "logistic_reg('public.logreg_model_test', 'public.titanic', '\"survived\"', '\"age\", \"fare\"'\nUSING PARAMETERS optimizer='newton', epsilon=1e-06, max_iterations=100, regularization='none', lambda=1, alpha=0.5)"
            )
        else:
            assert (
                model.get_vertica_attributes("call_string")["call_string"][0]
                == "logistic_reg('public.logreg_model_test', 'public.titanic', '\"survived\"', '\"age\", \"fare\"'\nUSING PARAMETERS optimizer='newton', epsilon=1e-06, max_iterations=100, regularization='none', lambda=1, alpha=0.5, fit_intercept=true)"
            )

    def test_get_params(self, model):
        params = model.get_params()

        assert params == {
            "solver": "newton",
            "penalty": "none",
            "max_iter": 100,
            "tol": 1e-06,
            "fit_intercept": True,
        }

    def test_prc_curve(self, model):
        prc = model.prc_curve(nbins=1000, show=False)

        assert prc["threshold"][10] == pytest.approx(0.009)
        assert prc["recall"][10] == pytest.approx(1.0)
        assert prc["precision"][10] == pytest.approx(0.392570281124498)
        assert prc["threshold"][900] == pytest.approx(0.899)
        assert prc["recall"][900] == pytest.approx(0.0460358056265985)
        assert prc["precision"][900] == pytest.approx(0.818181818181818)
        plt.close("all")

    def test_predict(self, titanic_vd, model):
        titanic_copy = titanic_vd.copy()

        model.predict(titanic_copy, name="pred_class1", cutoff=0.7)
        assert titanic_copy["pred_class1"].sum() == 56.0

        model.predict(titanic_copy, name="pred_class2", cutoff=0.3)
        assert titanic_copy["pred_class2"].sum() == 828.0

    def test_predict_proba(self, titanic_vd, model):
        titanic_copy = titanic_vd.copy()

        model.predict_proba(titanic_copy, name="probability", pos_label=1)
        assert titanic_copy["probability"].min() == pytest.approx(0.182718648793846)

    def test_roc_curve(self, model):
        roc = model.roc_curve(nbins=1000, show=False)

        assert roc["threshold"][100] == pytest.approx(0.1)
        assert roc["false_positive"][100] == pytest.approx(1.0)
        assert roc["true_positive"][100] == pytest.approx(1.0)
        assert roc["threshold"][900] == pytest.approx(0.9)
        assert roc["false_positive"][900] == pytest.approx(0.00661157024793388)
        assert roc["true_positive"][900] == pytest.approx(0.0434782608695652)
        plt.close("all")

    def test_cutoff_curve(self, model):
        cutoff_curve = model.cutoff_curve(nbins=1000, show=False)

        assert cutoff_curve["threshold"][100] == pytest.approx(0.1)
        assert cutoff_curve["false_positive"][100] == pytest.approx(1.0)
        assert cutoff_curve["true_positive"][100] == pytest.approx(1.0)
        assert cutoff_curve["threshold"][900] == pytest.approx(0.9)
        assert cutoff_curve["false_positive"][900] == pytest.approx(0.00661157024793388)
        assert cutoff_curve["true_positive"][900] == pytest.approx(0.0434782608695652)
        plt.close("all")

    def test_score(self, model):
        assert model.score(cutoff=0.7, metric="accuracy") == pytest.approx(
            0.6295180722891566
        )
        assert model.score(cutoff=0.3, metric="accuracy") == pytest.approx(
            0.4929718875502008
        )
        assert model.score(cutoff=0.7, metric="auc") == pytest.approx(
            0.6941239880788826
        )
        assert model.score(cutoff=0.3, metric="auc") == pytest.approx(
            0.6941239880788826
        )
        assert model.score(cutoff=0.7, metric="best_cutoff") == pytest.approx(0.3602)
        assert model.score(cutoff=0.3, metric="best_cutoff") == pytest.approx(0.3602)
        assert model.score(cutoff=0.7, metric="bm") == pytest.approx(
            0.07164507197057768
        )
        assert model.score(cutoff=0.3, metric="bm") == pytest.approx(
            0.13453108156665472
        )
        assert model.score(cutoff=0.7, metric="csi") == pytest.approx(
            0.09558823529411764
        )
        assert model.score(cutoff=0.3, metric="csi") == pytest.approx(
            0.41415313225058004
        )
        assert model.score(cutoff=0.7, metric="f1") == pytest.approx(
            0.17449664429530198
        )
        assert model.score(cutoff=0.3, metric="f1") == pytest.approx(0.5857260049220673)
        assert model.score(cutoff=0.7, metric="logloss") == pytest.approx(
            0.271495668573431
        )
        assert model.score(cutoff=0.3, metric="logloss") == pytest.approx(
            0.271495668573431
        )
        assert model.score(cutoff=0.7, metric="mcc") == pytest.approx(
            0.15187785294188016
        )
        assert model.score(cutoff=0.3, metric="mcc") == pytest.approx(
            0.17543607019922353
        )
        assert model.score(cutoff=0.7, metric="mk") == pytest.approx(
            0.32196048632218854
        )
        assert model.score(cutoff=0.3, metric="mk") == pytest.approx(
            0.22877846790890288
        )
        assert model.score(cutoff=0.7, metric="npv") == pytest.approx(0.625531914893617)
        assert model.score(cutoff=0.3, metric="npv") == pytest.approx(
            0.7976190476190477
        )
        assert model.score(cutoff=0.7, metric="prc_auc") == pytest.approx(
            0.5979751713359676
        )
        assert model.score(cutoff=0.3, metric="prc_auc") == pytest.approx(
            0.5979751713359676
        )
        assert model.score(cutoff=0.7, metric="precision") == pytest.approx(
            0.6964285714285714
        )
        assert model.score(cutoff=0.3, metric="precision") == pytest.approx(
            0.4311594202898551
        )
        assert model.score(cutoff=0.7, metric="specificity") == pytest.approx(
            0.971900826446281
        )
        assert model.score(cutoff=0.3, metric="specificity") == pytest.approx(
            0.22148760330578512
        )

    def test_set_params(self, model):
        model.set_params({"max_iter": 1000})

        assert model.get_params()["max_iter"] == 1000

    def test_model_from_vDF(self, titanic_vd):
        current_cursor().execute("DROP MODEL IF EXISTS logreg_from_vDF")
        model_test = LogisticRegression(
            "logreg_from_vDF",
        )
        model_test.fit(titanic_vd, ["age", "fare"], "survived")

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'logreg_from_vDF'"
        )
        assert current_cursor().fetchone()[0] == "logreg_from_vDF"

        model_test.drop()

    def test_optional_name(self):
        model = LogisticRegression()
        assert model.model_name is not None
