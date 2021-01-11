# (c) Copyright [2018-2020] Micro Focus or one of its affiliates.
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

import pytest, warnings, sys
from verticapy.learn.linear_model import LogisticRegression
from verticapy import drop_table
import matplotlib.pyplot as plt

from verticapy import set_option

set_option("print_info", False)


@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.learn.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    with warnings.catch_warnings(record=True) as w:
        drop_table(name="public.titanic", cursor=base.cursor)


@pytest.fixture(scope="module")
def model(base, titanic_vd):
    base.cursor.execute("DROP MODEL IF EXISTS logreg_model_test")
    model_class = LogisticRegression("logreg_model_test", cursor=base.cursor)
    model_class.fit("public.titanic", ["age", "fare"], "survived")
    yield model_class
    model_class.drop()


class TestLogisticRegression:
    def test_classification_report(self, model):
        cls_rep1 = model.classification_report().transpose()

        assert cls_rep1["auc"][0] == pytest.approx(0.6941239880788826)
        assert cls_rep1["prc_auc"][0] == pytest.approx(0.5979751713359676)
        assert cls_rep1["accuracy"][0] == pytest.approx(0.6766612641815235)
        assert cls_rep1["log_loss"][0] == pytest.approx(0.271495668573431)
        assert cls_rep1["precision"][0] == pytest.approx(0.6758620689655173)
        assert cls_rep1["recall"][0] == pytest.approx(0.21777777777777776)
        assert cls_rep1["f1_score"][0] == pytest.approx(0.3536312493573768)
        assert cls_rep1["mcc"][0] == pytest.approx(0.2359133929510658)
        assert cls_rep1["informedness"][0] == pytest.approx(0.15782879818594098)
        assert cls_rep1["markedness"][0] == pytest.approx(0.35262974573319417)
        assert cls_rep1["csi"][0] == pytest.approx(0.19718309859154928)
        assert cls_rep1["cutoff"][0] == pytest.approx(0.5)

        cls_rep2 = model.classification_report(cutoff=0.2).transpose()

        assert cls_rep2["cutoff"][0] == pytest.approx(0.2)

    def test_confusion_matrix(self, model):
        conf_mat1 = model.confusion_matrix()

        assert conf_mat1[0][0] == 737
        assert conf_mat1[0][1] == 352
        assert conf_mat1[1][0] == 47
        assert conf_mat1[1][1] == 98

        conf_mat2 = model.confusion_matrix(cutoff=0.2)

        assert conf_mat2[0][0] == 182
        assert conf_mat2[0][1] == 59
        assert conf_mat2[1][0] == 602
        assert conf_mat2[1][1] == 391

    def test_deploySQL(self, model):
        expected_sql = "PREDICT_LOGISTIC_REG(\"age\", \"fare\" USING PARAMETERS model_name = 'logreg_model_test', type = 'probability', match_by_pos = 'true')"
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self, base):
        base.cursor.execute("DROP MODEL IF EXISTS logreg_model_test_drop")
        model_test = LogisticRegression("logreg_model_test_drop", cursor=base.cursor)
        model_test.fit("public.titanic", ["age", "fare"], "survived")

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'logreg_model_test_drop'"
        )
        assert base.cursor.fetchone()[0] == "logreg_model_test_drop"

        model_test.drop()
        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'logreg_model_test_drop'"
        )
        assert base.cursor.fetchone() is None

    def test_features_importance(self, model):
        f_imp = model.features_importance()

        assert f_imp["index"] == ["fare", "age"]
        assert f_imp["importance"] == [85.51, 14.49]
        assert f_imp["sign"] == [1, -1]
        plt.close()

    def test_lift_chart(self, model):
        lift_ch = model.lift_chart()

        assert lift_ch["decision_boundary"][10] == pytest.approx(0.01)
        assert lift_ch["positive_prediction_ratio"][10] == pytest.approx(
            0.010230179028133
        )
        assert lift_ch["lift"][10] == pytest.approx(2.54731457800512)
        assert lift_ch["decision_boundary"][900] == pytest.approx(0.9)
        assert lift_ch["positive_prediction_ratio"][900] == pytest.approx(1.0)
        assert lift_ch["lift"][900] == pytest.approx(1.0)
        plt.close()

    @pytest.mark.skip(reason="test not implemented")
    def test_plot(self):
        pass

    def test_to_sklearn(self, model):
        md = model.to_sklearn()
        model.cursor.execute(
            "SELECT PREDICT_LOGISTIC_REG(11.0, 1993. USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model.name
            )
        )
        prediction = model.cursor.fetchone()[0]
        assert prediction == pytest.approx(md.predict([[11.0, 1993.0]])[0])
        model.cursor.execute(
            "SELECT PREDICT_LOGISTIC_REG(11.0, 1993. USING PARAMETERS model_name = '{}', match_by_pos=True, type='probability')".format(
                model.name
            )
        )
        prediction = model.cursor.fetchone()[0]
        assert prediction == pytest.approx(md.predict_proba([[11.0, 1993.0]])[0][1])

    @pytest.mark.skip(reason="shap doesn't want to work on python3.6")
    def test_shapExplainer(self, model):
        explainer = model.shapExplainer()
        assert explainer.expected_value[0] == pytest.approx(-0.4617437138350809)

    def test_get_attr(self, model):
        attr = model.get_attr()
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

        details = model.get_attr("details")
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

        reg = model.get_attr("regularization")
        assert reg["type"][0] == "none"
        assert reg["lambda"][0] == 1.0

        assert model.get_attr("iteration_count")["iteration_count"][0] == 4
        assert model.get_attr("rejected_row_count")["rejected_row_count"][0] == 238
        assert model.get_attr("accepted_row_count")["accepted_row_count"][0] == 996
        assert (
            model.get_attr("call_string")["call_string"][0]
            == "logistic_reg('public.logreg_model_test', 'public.titanic', '\"survived\"', '\"age\", \"fare\"'\nUSING PARAMETERS optimizer='newton', epsilon=1e-06, max_iterations=100, regularization='none', lambda=1, alpha=0.5)"
        )

    def test_get_params(self, model):
        params = model.get_params()

        assert params == {
            "solver": "newton",
            "penalty": "none",
            "max_iter": 100,
            "tol": 1e-06,
            "C": 1.0,
            "l1_ratio": None,
        }

    def test_prc_curve(self, model):
        prc = model.prc_curve()

        assert prc["threshold"][10] == pytest.approx(0.009)
        assert prc["recall"][10] == pytest.approx(1.0)
        assert prc["precision"][10] == pytest.approx(0.392570281124498)
        assert prc["threshold"][900] == pytest.approx(0.899)
        assert prc["recall"][900] == pytest.approx(0.0460358056265985)
        assert prc["precision"][900] == pytest.approx(0.818181818181818)
        plt.close()

    def test_predict(self, titanic_vd, model):
        titanic_copy = titanic_vd.copy()

        model.predict(titanic_copy, name="pred_probability")
        assert titanic_copy["pred_probability"].min() == pytest.approx(
            0.182718648793846
        )

        model.predict(titanic_copy, name="pred_class1", cutoff=0.7)
        assert titanic_copy["pred_class1"].sum() == 56.0

        model.predict(titanic_copy, name="pred_class2", cutoff=0.3)
        assert titanic_copy["pred_class2"].sum() == 828.0

    def test_roc_curve(self, model):
        roc = model.roc_curve()

        assert roc["threshold"][100] == pytest.approx(0.1)
        assert roc["false_positive"][100] == pytest.approx(1.0)
        assert roc["true_positive"][100] == pytest.approx(1.0)
        assert roc["threshold"][900] == pytest.approx(0.9)
        assert roc["false_positive"][900] == pytest.approx(0.00661157024793388)
        assert roc["true_positive"][900] == pytest.approx(0.0434782608695652)
        plt.close()

    def test_score(self, model):
        assert model.score(cutoff=0.7, method="accuracy") == pytest.approx(
            0.653160453808752
        )
        assert model.score(cutoff=0.3, method="accuracy") == pytest.approx(
            0.5429497568881686
        )
        assert model.score(cutoff=0.7, method="auc") == pytest.approx(
            0.6941239880788826
        )
        assert model.score(cutoff=0.3, method="auc") == pytest.approx(
            0.6941239880788826
        )
        assert model.score(cutoff=0.7, method="best_cutoff") == pytest.approx(0.36)
        assert model.score(cutoff=0.3, method="best_cutoff") == pytest.approx(0.36)
        assert model.score(cutoff=0.7, method="bm") == pytest.approx(
            0.06498299319727896
        )
        assert model.score(cutoff=0.3, method="bm") == pytest.approx(
            0.19256802721088428
        )
        assert model.score(cutoff=0.7, method="csi") == pytest.approx(
            0.0835117773019272
        )
        assert model.score(cutoff=0.3, method="csi") == pytest.approx(
            0.38762214983713356
        )
        assert model.score(cutoff=0.7, method="f1") == pytest.approx(0.1541501976284585)
        assert model.score(cutoff=0.3, method="f1") == pytest.approx(0.5586854460093896)
        assert model.score(cutoff=0.7, method="logloss") == pytest.approx(
            0.271495668573431
        )
        assert model.score(cutoff=0.3, method="logloss") == pytest.approx(
            0.271495668573431
        )
        assert model.score(cutoff=0.7, method="mcc") == pytest.approx(
            0.15027866941483783
        )
        assert model.score(cutoff=0.3, method="mcc") == pytest.approx(
            0.19727419700681625
        )
        assert model.score(cutoff=0.7, method="mk") == pytest.approx(
            0.34753213679359685
        )
        assert model.score(cutoff=0.3, method="mk") == pytest.approx(0.202095380880988)
        assert model.score(cutoff=0.7, method="npv") == pytest.approx(
            0.6964285714285714
        )
        assert model.score(cutoff=0.3, method="npv") == pytest.approx(
            0.4311594202898551
        )
        assert model.score(cutoff=0.7, method="prc_auc") == pytest.approx(
            0.5979751713359676
        )
        assert model.score(cutoff=0.3, method="prc_auc") == pytest.approx(
            0.5979751713359676
        )
        assert model.score(cutoff=0.7, method="precision") == pytest.approx(
            0.6964285714285714
        )
        assert model.score(cutoff=0.3, method="precision") == pytest.approx(
            0.4311594202898551
        )
        assert model.score(cutoff=0.7, method="specificity") == pytest.approx(
            0.9783163265306123
        )
        assert model.score(cutoff=0.3, method="specificity") == pytest.approx(
            0.399234693877551
        )

    @pytest.mark.skip(reason="test not implemented")
    def test_set_cursor(self):
        pass

    def test_set_params(self, model):
        model.set_params({"max_iter": 1000})

        assert model.get_params()["max_iter"] == 1000

    def test_model_from_vDF(self, base, titanic_vd):
        base.cursor.execute("DROP MODEL IF EXISTS logreg_from_vDF")
        model_test = LogisticRegression("logreg_from_vDF", cursor=base.cursor)
        model_test.fit(titanic_vd, ["age", "fare"], "survived")

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'logreg_from_vDF'"
        )
        assert base.cursor.fetchone()[0] == "logreg_from_vDF"

        model_test.drop()
