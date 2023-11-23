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
from verticapy.datasets import load_titanic, load_winequality
from verticapy.learn.svm import LinearSVC

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
    current_cursor().execute("DROP MODEL IF EXISTS lsvc_model_test")
    model_class = LinearSVC(
        "lsvc_model_test",
    )
    model_class.fit("public.titanic", ["age", "fare"], "survived")
    yield model_class
    model_class.drop()


class TestLinearSVC:
    def test_repr(self, model):
        assert model.__repr__() == "<LinearSVC>"

    def test_classification_report(self, model):
        cls_rep1 = model.classification_report().transpose()

        assert cls_rep1["auc"][0] == pytest.approx(0.6933968844454788, 1e-2)
        assert cls_rep1["prc_auc"][0] == pytest.approx(0.5976470350144453, 1e-2)
        assert cls_rep1["accuracy"][0] == pytest.approx(0.6536144578313253, 1e-2)
        assert cls_rep1["log_loss"][0] == pytest.approx(0.279724470067258, 1e-2)
        assert cls_rep1["precision"][0] == pytest.approx(0.6916666666666667, 1e-2)
        assert cls_rep1["recall"][0] == pytest.approx(0.21227621483375958, 1e-2)
        assert cls_rep1["f1_score"][0] == pytest.approx(0.324853228962818, 1e-2)
        assert cls_rep1["mcc"][0] == pytest.approx(0.22669555629341528, 1e-2)
        assert cls_rep1["informedness"][0] == pytest.approx(0.151119190040371, 1e-2)
        assert cls_rep1["markedness"][0] == pytest.approx(0.34006849315068477, 1e-2)
        assert cls_rep1["csi"][0] == pytest.approx(0.1939252336448598, 1e-2)

    def test_confusion_matrix(self, model):
        conf_mat1 = model.confusion_matrix()

        assert conf_mat1[0][0] == 568
        assert conf_mat1[1][0] == 308
        assert conf_mat1[0][1] == 37
        assert conf_mat1[1][1] == 83

        conf_mat2 = model.confusion_matrix(cutoff=0.2)

        assert conf_mat2[0][0] == 0
        assert conf_mat2[1][0] == 0
        assert conf_mat2[0][1] == 605
        assert conf_mat2[1][1] == 391

    @skip_plt
    def test_contour(self, titanic_vd):
        model_test = LinearSVC(
            "model_contour",
        )
        model_test.drop()
        model_test.fit(
            titanic_vd,
            ["age", "fare"],
            "survived",
        )
        result = model_test.contour()
        assert len(result.get_default_bbox_extra_artists()) == 34
        model_test.drop()

    def test_deploySQL(self, model):
        expected_sql = "PREDICT_SVM_CLASSIFIER(\"age\", \"fare\" USING PARAMETERS model_name = 'lsvc_model_test', type = 'probability', match_by_pos = 'true')"
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self):
        current_cursor().execute("DROP MODEL IF EXISTS lsvc_model_test_drop")
        model_test = LinearSVC(
            "lsvc_model_test_drop",
        )
        model_test.fit("public.titanic", ["age", "fare"], "survived")

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'lsvc_model_test_drop'"
        )
        assert current_cursor().fetchone()[0] == "lsvc_model_test_drop"

        model_test.drop()
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'lsvc_model_test_drop'"
        )
        assert current_cursor().fetchone() is None

    def test_features_importance(self, model):
        f_imp = model.features_importance(show=False)

        assert f_imp["index"] == ["fare", "age"]
        assert f_imp["importance"] == [85.09, 14.91]
        assert f_imp["sign"] == [1, -1]
        plt.close("all")

    def test_lift_chart(self, model):
        lift_ch = model.lift_chart(nbins=1000, show=False)

        assert lift_ch["decision_boundary"][10] == pytest.approx(0.01)
        assert lift_ch["positive_prediction_ratio"][10] == pytest.approx(0.0)
        assert lift_ch["decision_boundary"][900] == pytest.approx(0.9)
        assert lift_ch["positive_prediction_ratio"][900] == pytest.approx(1.0)
        assert lift_ch["lift"][900] == pytest.approx(1.0)
        plt.close("all")

    @skip_plt
    def test_get_plot(self, winequality_vd):
        current_cursor().execute("DROP MODEL IF EXISTS model_test_plot")
        model_test = LinearSVC(
            "model_test_plot",
        )
        model_test.fit(winequality_vd, ["alcohol"], "good")
        result = model_test.plot(color="r")
        assert len(result.get_default_bbox_extra_artists()) == 11
        plt.close("all")
        model_test.drop()
        model_test.fit(winequality_vd, ["alcohol", "residual_sugar"], "good")
        result = model_test.plot(color="r")
        assert len(result.get_default_bbox_extra_artists()) == 11
        plt.close("all")
        model_test.drop()
        model_test.fit(
            winequality_vd, ["alcohol", "residual_sugar", "fixed_acidity"], "good"
        )
        result = model_test.plot(color="r")
        assert len(result.get_default_bbox_extra_artists()) == 5
        plt.close("all")
        model_test.drop()

    def test_to_python(self, model):
        current_cursor().execute(
            "SELECT PREDICT_SVM_CLASSIFIER(3.0, 11.0 USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model.model_name
            )
        )
        prediction = current_cursor().fetchone()[0]
        assert prediction == pytest.approx(model.to_python()([[3.0, 11.0]])[0])
        current_cursor().execute(
            "SELECT PREDICT_SVM_CLASSIFIER(3.0, 11.0 USING PARAMETERS model_name = '{}', type='probability', match_by_pos=True)".format(
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

    def test_to_sql(self, model):
        current_cursor().execute(
            "SELECT PREDICT_SVM_CLASSIFIER(3.0, 11.0 USING PARAMETERS model_name = '{}', match_by_pos=True), {}".format(
                model.model_name, model.to_sql([3.0, 11.0])
            )
        )
        prediction = current_cursor().fetchone()
        assert prediction[0] == pytest.approx(prediction[1])

    def test_get_vertica_attributes(self, model):
        attr = model.get_vertica_attributes()
        assert attr["attr_name"] == [
            "details",
            "accepted_row_count",
            "rejected_row_count",
            "iteration_count",
            "call_string",
        ]
        assert attr["attr_fields"] == [
            "predictor, coefficient",
            "accepted_row_count",
            "rejected_row_count",
            "iteration_count",
            "call_string",
        ]
        assert attr["#_of_rows"] == [3, 1, 1, 1, 1]

        details = model.get_vertica_attributes("details")
        assert details["predictor"] == ["Intercept", "age", "fare"]
        assert details["coefficient"][0] == pytest.approx(-0.226679636751873)
        assert details["coefficient"][1] == pytest.approx(-0.00661256493751514)
        assert details["coefficient"][2] == pytest.approx(0.00587052591948468)

        assert (
            model.get_vertica_attributes("accepted_row_count")["accepted_row_count"][0]
            == 996
        )
        assert (
            model.get_vertica_attributes("rejected_row_count")["rejected_row_count"][0]
            == 238
        )
        assert (
            model.get_vertica_attributes("iteration_count")["iteration_count"][0] == 6
        )
        assert (
            model.get_vertica_attributes("call_string")["call_string"][0]
            == "SELECT svm_classifier('public.lsvc_model_test', 'public.titanic', '\"survived\"', '\"age\", \"fare\"'\nUSING PARAMETERS class_weights='1,1', C=1, max_iterations=100, intercept_mode='regularized', intercept_scaling=1, epsilon=0.0001);"
        )

    def test_get_params(self, model):
        params = model.get_params()

        assert params == {
            "tol": 0.0001,
            "C": 1.0,
            "max_iter": 100,
            "intercept_scaling": 1.0,
            "intercept_mode": "regularized",
            "class_weight": [1, 1],
        }

    def test_prc_curve(self, model):
        prc = model.prc_curve(nbins=1000, show=False)

        assert prc["threshold"][10] == pytest.approx(0.009)
        assert prc["recall"][10] == pytest.approx(1.0)
        assert prc["precision"][10] == pytest.approx(0.392570281124498)
        assert prc["threshold"][900] == pytest.approx(0.899)
        assert prc["recall"][900] == pytest.approx(0.010230179028133)
        assert prc["precision"][900] == pytest.approx(1.0)
        plt.close("all")

    def test_predict(self, titanic_vd, model):
        titanic_copy = titanic_vd.copy()

        model.predict(titanic_copy, name="pred_class1", cutoff=0.7)
        assert titanic_copy["pred_class1"].sum() == 23.0

        model.predict(titanic_copy, name="pred_class2", cutoff=0.3)
        assert titanic_copy["pred_class2"].sum() == 996.0

    def test_predict_proba(self, titanic_vd, model):
        titanic_copy = titanic_vd.copy()

        model.predict_proba(titanic_copy, name="probability", pos_label=1)
        assert titanic_copy["probability"].min() == pytest.approx(0.33841486903496)

    def test_roc_curve(self, model):
        roc = model.roc_curve(nbins=1000, show=False)

        assert roc["threshold"][100] == pytest.approx(0.1)
        assert roc["false_positive"][100] == pytest.approx(1.0)
        assert roc["true_positive"][100] == pytest.approx(1.0)
        assert roc["threshold"][700] == pytest.approx(0.7)
        assert roc["false_positive"][700] == pytest.approx(0.00661157024793388)
        assert roc["true_positive"][700] == pytest.approx(0.0485933503836317)
        plt.close("all")

    def test_cutoff_curve(self, model):
        cutoff_curve = model.cutoff_curve(nbins=1000, show=False)

        assert cutoff_curve["threshold"][100] == pytest.approx(0.1)
        assert cutoff_curve["false_positive"][100] == pytest.approx(1.0)
        assert cutoff_curve["true_positive"][100] == pytest.approx(1.0)
        assert cutoff_curve["threshold"][700] == pytest.approx(0.7)
        assert cutoff_curve["false_positive"][700] == pytest.approx(0.00661157024793388)
        assert cutoff_curve["true_positive"][700] == pytest.approx(0.0485933503836317)
        plt.close("all")

    def test_score(self, model):
        assert model.score(cutoff=0.7, metric="accuracy") == pytest.approx(
            0.6224899598393574
        )
        assert model.score(cutoff=0.3, metric="accuracy") == pytest.approx(
            0.392570281124498
        )
        assert model.score(cutoff=0.7, metric="auc") == pytest.approx(
            0.6933968844454788
        )
        assert model.score(cutoff=0.3, metric="auc") == pytest.approx(
            0.6933968844454788
        )
        assert model.score(cutoff=0.7, metric="best_cutoff") == pytest.approx(0.431)
        assert model.score(cutoff=0.3, metric="best_cutoff") == pytest.approx(0.431)
        assert model.score(cutoff=0.7, metric="bm") == pytest.approx(
            0.041981780135697866
        )
        assert model.score(cutoff=0.3, metric="bm") == pytest.approx(0.0)
        assert model.score(cutoff=0.7, metric="csi") == pytest.approx(
            0.04810126582278481
        )
        assert model.score(cutoff=0.3, metric="csi") == pytest.approx(0.392570281124498)
        assert model.score(cutoff=0.7, metric="f1") == pytest.approx(
            0.09178743961352658
        )
        assert model.score(cutoff=0.3, metric="f1") == pytest.approx(0.5638067772170151)
        assert model.score(cutoff=0.7, metric="logloss") == pytest.approx(
            0.279724470067258
        )
        assert model.score(cutoff=0.3, metric="logloss") == pytest.approx(
            0.279724470067258
        )
        assert model.score(cutoff=0.7, metric="mcc") == pytest.approx(
            0.13649180522208684
        )
        assert model.score(cutoff=0.3, metric="mcc") == pytest.approx(0.0)
        assert model.score(cutoff=0.7, metric="mk") == pytest.approx(
            0.44376424326377406
        )
        assert model.score(cutoff=0.3, metric="mk") == pytest.approx(-0.607429718875502)
        assert model.score(cutoff=0.7, metric="npv") == pytest.approx(
            0.6176772867420349
        )
        assert model.score(cutoff=0.3, metric="npv") == pytest.approx(0.0)
        assert model.score(cutoff=0.7, metric="prc_auc") == pytest.approx(
            0.5976470350144453
        )
        assert model.score(cutoff=0.3, metric="prc_auc") == pytest.approx(
            0.5976470350144453
        )
        assert model.score(cutoff=0.7, metric="precision") == pytest.approx(
            0.8260869565217391
        )
        assert model.score(cutoff=0.3, metric="precision") == pytest.approx(
            0.392570281124498
        )
        assert model.score(cutoff=0.7, metric="specificity") == pytest.approx(
            0.9933884297520661
        )
        assert model.score(cutoff=0.3, metric="specificity") == pytest.approx(0.0)

    def test_set_params(self, model):
        model.set_params({"max_iter": 1000})

        assert model.get_params()["max_iter"] == 1000

    def test_model_from_vDF(self, titanic_vd):
        current_cursor().execute("DROP MODEL IF EXISTS lsvc_from_vDF")
        model_test = LinearSVC(
            "lsvc_from_vDF",
        )
        model_test.fit(titanic_vd, ["age", "fare"], "survived")

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'lsvc_from_vDF'"
        )
        assert current_cursor().fetchone()[0] == "lsvc_from_vDF"

        model_test.drop()

    def test_optional_name(self):
        model = LinearSVC()
        assert model.model_name is not None
