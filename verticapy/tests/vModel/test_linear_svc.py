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

import pytest, warnings, math, sys
from verticapy.learn.svm import LinearSVC
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
    base.cursor.execute("DROP MODEL IF EXISTS lsvc_model_test")
    model_class = LinearSVC("lsvc_model_test", cursor=base.cursor)
    model_class.fit("public.titanic", ["age", "fare"], "survived")
    yield model_class
    model_class.drop()


class TestLinearSVC:
    def test_classification_report(self, model):
        cls_rep1 = model.classification_report().transpose()

        assert cls_rep1["auc"][0] == pytest.approx(0.6933968844454788)
        assert cls_rep1["prc_auc"][0] == pytest.approx(0.5976470350144453)
        assert cls_rep1["accuracy"][0] == pytest.approx(0.6726094003241491)
        assert cls_rep1["log_loss"][0] == pytest.approx(0.279724470067258)
        assert cls_rep1["precision"][0] == pytest.approx(0.6916666666666667)
        assert cls_rep1["recall"][0] == pytest.approx(0.18444444444444444)
        assert cls_rep1["f1_score"][0] == pytest.approx(0.30906081919735207)
        assert cls_rep1["mcc"][0] == pytest.approx(0.22296937510796555)
        assert cls_rep1["informedness"][0] == pytest.approx(0.13725056689342408)
        assert cls_rep1["markedness"][0] == pytest.approx(0.36222321962896453)
        assert cls_rep1["csi"][0] == pytest.approx(0.1704312114989733)
        assert cls_rep1["cutoff"][0] == pytest.approx(0.5)

        cls_rep2 = model.classification_report(cutoff=0.2).transpose()

        assert cls_rep2["cutoff"][0] == pytest.approx(0.2)

    def test_confusion_matrix(self, model):
        conf_mat1 = model.confusion_matrix()

        assert conf_mat1[0][0] == 747
        assert conf_mat1[0][1] == 367
        assert conf_mat1[1][0] == 37
        assert conf_mat1[1][1] == 83

        conf_mat2 = model.confusion_matrix(cutoff=0.2)

        assert conf_mat2[0][0] == 179
        assert conf_mat2[0][1] == 59
        assert conf_mat2[1][0] == 605
        assert conf_mat2[1][1] == 391

    def test_deploySQL(self, model):
        expected_sql = "PREDICT_SVM_CLASSIFIER(\"age\", \"fare\" USING PARAMETERS model_name = 'lsvc_model_test', type = 'probability', match_by_pos = 'true')"
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self, base):
        base.cursor.execute("DROP MODEL IF EXISTS lsvc_model_test_drop")
        model_test = LinearSVC("lsvc_model_test_drop", cursor=base.cursor)
        model_test.fit("public.titanic", ["age", "fare"], "survived")

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'lsvc_model_test_drop'"
        )
        assert base.cursor.fetchone()[0] == "lsvc_model_test_drop"

        model_test.drop()
        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'lsvc_model_test_drop'"
        )
        assert base.cursor.fetchone() is None

    def test_features_importance(self, model):
        f_imp = model.features_importance()

        assert f_imp["index"] == ["fare", "age"]
        assert f_imp["importance"] == [85.09, 14.91]
        assert f_imp["sign"] == [1, -1]
        plt.close()

    def test_lift_chart(self, model):
        lift_ch = model.lift_chart()

        assert lift_ch["decision_boundary"][10] == pytest.approx(0.01)
        assert lift_ch["positive_prediction_ratio"][10] == pytest.approx(0.0)
        assert math.isnan(lift_ch["lift"][10]) == True
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
            "SELECT PREDICT_SVM_CLASSIFIER(11.0, 1993. USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model.name
            )
        )
        prediction = model.cursor.fetchone()[0]
        assert prediction == pytest.approx(md.predict([[11.0, 1993.0]])[0])

        # 'LinearSVC' object (md) has no attribute 'predict_proba'

    @pytest.mark.skip(reason="shap doesn't want to work on python3.6")
    def test_shapExplainer(self, model):
        explainer = model.shapExplainer()
        assert explainer.expected_value[0] == pytest.approx(-0.22667938806360247)

    def test_get_attr(self, model):
        attr = model.get_attr()
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

        details = model.get_attr("details")
        assert details["predictor"] == ["Intercept", "age", "fare"]
        assert details["coefficient"][0] == pytest.approx(-0.226679636751873)
        assert details["coefficient"][1] == pytest.approx(-0.00661256493751514)
        assert details["coefficient"][2] == pytest.approx(0.00587052591948468)

        assert model.get_attr("accepted_row_count")["accepted_row_count"][0] == 996
        assert model.get_attr("rejected_row_count")["rejected_row_count"][0] == 238
        assert model.get_attr("iteration_count")["iteration_count"][0] == 6
        assert (
            model.get_attr("call_string")["call_string"][0]
            == "SELECT svm_classifier('public.lsvc_model_test', 'public.titanic', '\"survived\"', '\"age\", \"fare\"'\nUSING PARAMETERS class_weights='none', C=1, max_iterations=100, intercept_mode='regularized', intercept_scaling=1, epsilon=0.0001);"
        )

    def test_get_params(self, model):
        params = model.get_params()

        assert params == {
            'tol': 0.0001,
            'C': 1.0,
            "max_iter": 100,
            'fit_intercept': True,
            'intercept_scaling': 1.0,
            'intercept_mode': 'regularized',
            'class_weight': [1, 1],
            'penalty': 'l2'
        }

    def test_prc_curve(self, model):
        prc = model.prc_curve()

        assert prc["threshold"][10] == pytest.approx(0.009)
        assert prc["recall"][10] == pytest.approx(1.0)
        assert prc["precision"][10] == pytest.approx(0.392570281124498)
        assert prc["threshold"][900] == pytest.approx(0.899)
        assert prc["recall"][900] == pytest.approx(0.010230179028133)
        assert prc["precision"][900] == pytest.approx(1.0)
        plt.close()

    def test_predict(self, titanic_vd, model):
        titanic_copy = titanic_vd.copy()

        model.predict(titanic_copy, name="pred_probability")
        assert titanic_copy["pred_probability"].min() == pytest.approx(
            0.33841486903496
        )

        model.predict(titanic_copy, name="pred_class1", cutoff=0.7)
        assert titanic_copy["pred_class1"].sum() == 23.0

        model.predict(titanic_copy, name="pred_class2", cutoff=0.3)
        assert titanic_copy["pred_class2"].sum() == 996.0

    def test_roc_curve(self, model):
        roc = model.roc_curve()

        assert roc["threshold"][100] == pytest.approx(0.1)
        assert roc["false_positive"][100] == pytest.approx(1.0)
        assert roc["true_positive"][100] == pytest.approx(1.0)
        assert roc["threshold"][700] == pytest.approx(0.7)
        assert roc["false_positive"][700] == pytest.approx(0.00661157024793388)
        assert roc["true_positive"][700] == pytest.approx(0.0485933503836317)
        plt.close()

    def test_score(self, model):
        assert model.score(cutoff=0.7, method="accuracy") == pytest.approx(
            0.6474878444084279
        )
        assert model.score(cutoff=0.3, method="accuracy") == pytest.approx(
            0.4619124797406807
        )
        assert model.score(cutoff=0.7, method="auc") == pytest.approx(
            0.6933968844454788
        )
        assert model.score(cutoff=0.3, method="auc") == pytest.approx(
            0.6933968844454788
        )
        assert model.score(cutoff=0.7, method="best_cutoff") == pytest.approx(0.431)
        assert model.score(cutoff=0.3, method="best_cutoff") == pytest.approx(0.431)
        assert model.score(cutoff=0.7, method="bm") == pytest.approx(
            0.03712018140589568
        )
        assert model.score(cutoff=0.3, method="bm") == pytest.approx(
            0.09720521541950111
        )
        assert model.score(cutoff=0.7, method="csi") == pytest.approx(
            0.04185022026431718
        )
        assert model.score(cutoff=0.3, method="csi") == pytest.approx(
            0.3706161137440758
        )
        assert model.score(cutoff=0.7, method="f1") == pytest.approx(0.080338266384778)
        assert model.score(cutoff=0.3, method="f1") == pytest.approx(0.5408022130013832)
        assert model.score(cutoff=0.7, method="logloss") == pytest.approx(
            0.279724470067258
        )
        assert model.score(cutoff=0.3, method="logloss") == pytest.approx(
            0.279724470067258
        )
        assert model.score(cutoff=0.7, method="mcc") == pytest.approx(
            0.13211082012086103
        )
        assert model.score(cutoff=0.3, method="mcc") == pytest.approx(
            0.11858662456854734
        )
        assert model.score(cutoff=0.7, method="mk") == pytest.approx(
            0.4701827451261984
        )
        assert model.score(cutoff=0.3, method="mk") == pytest.approx(0.14467112146063243)
        assert model.score(cutoff=0.7, method="npv") == pytest.approx(
            0.8260869565217391
        )
        assert model.score(cutoff=0.3, method="npv") == pytest.approx(
            0.392570281124498
        )
        assert model.score(cutoff=0.7, method="prc_auc") == pytest.approx(
            0.5976470350144453
        )
        assert model.score(cutoff=0.3, method="prc_auc") == pytest.approx(
            0.5976470350144453
        )
        assert model.score(cutoff=0.7, method="precision") == pytest.approx(
            0.8260869565217391
        )
        assert model.score(cutoff=0.3, method="precision") == pytest.approx(
            0.392570281124498
        )
        assert model.score(cutoff=0.7, method="specificity") == pytest.approx(
            0.9948979591836735
        )
        assert model.score(cutoff=0.3, method="specificity") == pytest.approx(
            0.22831632653061223
        )

    @pytest.mark.skip(reason="test not implemented")
    def test_set_cursor(self):
        pass

    def test_set_params(self, model):
        model.set_params({"max_iter": 1000})

        assert model.get_params()["max_iter"] == 1000

    def test_model_from_vDF(self, base, titanic_vd):
        base.cursor.execute("DROP MODEL IF EXISTS lsvc_from_vDF")
        model_test = LinearSVC("lsvc_from_vDF", cursor=base.cursor)
        model_test.fit(titanic_vd, ["age", "fare"], "survived")

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'lsvc_from_vDF'"
        )
        assert base.cursor.fetchone()[0] == "lsvc_from_vDF"

        model_test.drop()
