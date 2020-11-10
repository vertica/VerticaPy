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

import pytest
from verticapy.learn.linear_model import LogisticRegression
from verticapy import drop_table
from decimal import Decimal


@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.learn.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    titanic.set_display_parameters(print_info=False)
    yield titanic
    drop_table(name="public.titanic", cursor=base.cursor)


@pytest.fixture(scope="module")
def model(base, titanic_vd):
    base.cursor.execute("DROP MODEL IF EXISTS logreg_model_test")
    model_class = LogisticRegression("logreg_model_test", cursor=base.cursor)
    model_class.fit("public.titanic", ["age", "fare"], "survived")
    yield model_class
    model_class.drop()


class TestLogisticRegression:
    @pytest.mark.xfail(reason = "The returned cutoff value is wrong")
    def test_classification_report(self, model):
        cls_rep1 = model.classification_report().transpose()

        assert cls_rep1["auc"][0] == pytest.approx(0.6974762740166146)
        assert cls_rep1["prc_auc"][0] == pytest.approx(0.6003540469187277)
        assert cls_rep1["accuracy"][0] == pytest.approx(0.6969205834683955)
        assert cls_rep1["log_loss"][0] == pytest.approx(0.281741003041208)
        assert cls_rep1["precision"][0] == pytest.approx(0.6194968553459119)
        assert cls_rep1["recall"][0] == pytest.approx(0.43777777777777777)
        assert cls_rep1["f1_score"][0] == pytest.approx(0.5769062584198693)
        assert cls_rep1["mcc"][0] == pytest.approx(0.31193616529653234)
        assert cls_rep1["informedness"][0] == pytest.approx(0.2834410430839003)
        assert cls_rep1["markedness"][0] == pytest.approx(0.34329598198346645)
        assert cls_rep1["csi"][0] == pytest.approx(0.3450087565674256)
        assert cls_rep1["cutoff"][0] == pytest.approx(0.5)

        cls_rep2 = model.classification_report(cutoff = 0.2).transpose()

        assert cls_rep2["cutoff"][0] == pytest.approx(0.2)

    def test_confusion_matrix(self, model):
        conf_mat1 = model.confusion_matrix()

        assert conf_mat1[0][0] == 663
        assert conf_mat1[0][1] == 253
        assert conf_mat1[1][0] == 121
        assert conf_mat1[1][1] == 197

        conf_mat2 = model.confusion_matrix(cutoff = 0.2)

        assert conf_mat2[0][0] == 179
        assert conf_mat2[0][1] == 59
        assert conf_mat2[1][0] == 605
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

        assert f_imp["index"] == ['fare', 'age']
        assert f_imp["importance"] == [87.36, 12.64]
        # TODO: it is nicer not to have Decimal for sign
        assert f_imp["sign"] == [Decimal('1'), Decimal('-1')]

    def test_lift_chart(self, model):
        lift_ch = model.lift_chart()

        assert lift_ch["decision_boundary"][10] == pytest.approx(0.01)
        assert lift_ch["positive_prediction_ratio"][10] == pytest.approx(0.010230179028133)
        assert lift_ch["lift"][10] == pytest.approx(2.54731457800512)
        assert lift_ch["decision_boundary"][900] == pytest.approx(0.9)
        assert lift_ch["positive_prediction_ratio"][900] == pytest.approx(1.0)
        assert lift_ch["lift"][900] == pytest.approx(1.0)

    @pytest.mark.skip(reason="test not implemented")
    def test_plot(self):
        pass

    def test_get_model_attribute(self, model):
        attr = model.get_model_attribute()
        assert attr["attr_name"] == ['details', 'regularization', 'iteration_count', 'rejected_row_count',
                                     'accepted_row_count', 'call_string']
        assert attr["attr_fields"] == ['predictor, coefficient, std_err, z_value, p_value', 'type, lambda',
                                       'iteration_count', 'rejected_row_count', 'accepted_row_count', 'call_string']
        assert attr["#_of_rows"] == [3, 1, 1, 1, 1, 1]

        details = model.get_model_attribute('details')
        assert details["predictor"] == ['Intercept', 'age', 'fare']
        assert details["coefficient"][0] == pytest.approx(-0.091348758337523)
        assert details["coefficient"][1] == pytest.approx(-0.0143850235204284)
        assert details["coefficient"][2] == pytest.approx(0.0154603623341147)
        assert details["std_err"][0] == pytest.approx(0.155594583418985)
        assert details["std_err"][1] == pytest.approx(0.00475381848744905)
        assert details["std_err"][2] == pytest.approx(0.00211946971061136)
        assert details["z_value"][0] == pytest.approx(-0.587094719689174)
        assert details["z_value"][1] == pytest.approx(-3.02599343210254)
        assert details["z_value"][2] == pytest.approx(7.29444835031644)
        assert details["p_value"][0] == pytest.approx(0.557140093691285)
        assert details["p_value"][1] == pytest.approx(0.00247817685818198)
        assert details["p_value"][2] == pytest.approx(2.99885239324552e-13)

        reg = model.get_model_attribute('regularization')
        assert reg["type"][0] == 'l2'
        assert reg["lambda"][0] == 1.0

        assert model.get_model_attribute('iteration_count')["iteration_count"][0] == 5
        assert model.get_model_attribute('rejected_row_count')["rejected_row_count"][0] == 238
        assert model.get_model_attribute('accepted_row_count')["accepted_row_count"][0] == 996
        assert model.get_model_attribute('call_string')["call_string"][0] == 'logistic_reg(\'public.logreg_model_test\', \'public.titanic\', \'"survived"\', \'"age", "fare"\'\nUSING PARAMETERS optimizer=\'cgd\', epsilon=0.0001, max_iterations=100, regularization=\'l2\', lambda=1, alpha=0)'

    def test_get_params(self, model):
        params = model.get_params()

        assert params == {'solver': 'cgd', 'penalty': 'l2', 'max_iter': 100, 'l1_ratio': 0.5, 'C': 1, 'tol': 0.0001}

    def test_prc_curve(self, model):
        prc = model.prc_curve()

        assert prc["threshold"][10] == pytest.approx(0.009)
        assert prc["recall"][10] == pytest.approx(1.0)
        assert prc["precision"][10] == pytest.approx(0.392570281124498)
        assert prc["threshold"][900] == pytest.approx(0.899)
        assert prc["recall"][900] == pytest.approx(0.0664961636828645)
        assert prc["precision"][900] == pytest.approx(0.702702702702703)

    def test_predict(self, titanic_vd, model):
        titanic_copy = titanic_vd.copy()

        model.predict(titanic_copy, name = "pred_probability")
        assert titanic_copy["pred_probability"].min() == pytest.approx(0.261992872793673)

        model.predict(titanic_copy, name = "pred_class1", cutoff = 0.7)
        assert titanic_copy["pred_class1"].sum() == 86

        model.predict(titanic_copy, name = "pred_class2", cutoff = 0.3)
        assert titanic_copy["pred_class2"].sum() == 989

    def test_roc_curve(self, model):
        roc = model.roc_curve()

        assert roc["threshold"][100] == pytest.approx(0.1)
        assert roc["false_positive"][100] == pytest.approx(1.0)
        assert roc["true_positive"][100] == pytest.approx(1.0)
        assert roc["threshold"][900] == pytest.approx(0.9)
        assert roc["false_positive"][900] == pytest.approx(0.0181818181818182)
        assert roc["true_positive"][900] == pytest.approx(0.0664961636828645)

    def test_score(self, model):
        assert model.score(cutoff = 0.7, method = "accuracy") == pytest.approx(0.6709886547811994)
        assert model.score(cutoff = 0.3, method = "accuracy") == pytest.approx(0.4659643435980551)
        assert model.score(cutoff = 0.7, method = "auc") == pytest.approx(0.6974762740166146)
        assert model.score(cutoff = 0.3, method = "auc") == pytest.approx(0.6974762740166146)
        assert model.score(cutoff = 0.7, method = "best_cutoff") == pytest.approx(0.458)
        assert model.score(cutoff = 0.3, method = "best_cutoff") == pytest.approx(0.458)
        assert model.score(cutoff = 0.7, method = "bm") == pytest.approx(0.11765873015873018)
        assert model.score(cutoff = 0.3, method = "bm") == pytest.approx(0.10263605442176882)
        assert model.score(cutoff = 0.7, method = "csi") == pytest.approx(0.13800424628450106)
        assert model.score(cutoff = 0.3, method = "csi") == pytest.approx(0.37178265014299333)
        assert model.score(cutoff = 0.7, method = "f1") == pytest.approx(0.24253731343283583)
        assert model.score(cutoff = 0.3, method = "f1") == pytest.approx(0.5420430854760251)
        assert model.score(cutoff = 0.7, method = "logloss") == pytest.approx(0.281741003041208)
        assert model.score(cutoff = 0.3, method = "logloss") == pytest.approx(0.281741003041208)
        assert model.score(cutoff = 0.7, method = "mcc") == pytest.approx(0.22241715204459717)
        assert model.score(cutoff = 0.3, method = "mcc") == pytest.approx(0.12384630352469281)
        assert model.score(cutoff = 0.7, method = "mk") == pytest.approx(0.42044809982983544)
        assert model.score(cutoff = 0.3, method = "mk") == pytest.approx(0.14943975567982504)
        assert model.score(cutoff = 0.7, method = "npv") == pytest.approx(0.7558139534883721)
        assert model.score(cutoff = 0.3, method = "npv") == pytest.approx(0.3943377148634985)
        assert model.score(cutoff = 0.7, method = "prc_auc") == pytest.approx(0.6003540469187277)
        assert model.score(cutoff = 0.3, method = "prc_auc") == pytest.approx(0.6003540469187277)
        assert model.score(cutoff = 0.7, method = "precision") == pytest.approx(0.7558139534883721)
        assert model.score(cutoff = 0.3, method = "precision") == pytest.approx(0.3943377148634985)
        assert model.score(cutoff = 0.7, method = "specificity") == pytest.approx(0.9732142857142857)
        assert model.score(cutoff = 0.3, method = "specificity") == pytest.approx(0.23596938775510204)

    @pytest.mark.skip(reason="test not implemented")
    def test_set_cursor(self):
        pass

    def test_set_params(self, model):
        model.set_params({"max_iter": 1000})

        assert model.get_params()['max_iter'] == 1000

    @pytest.mark.skip(reason="feautre not implemented")
    def test_model_from_vDF(self, base, titanic_vd):
        base.cursor.execute("DROP MODEL IF EXISTS logreg_from_vDF")
        model_test = LinearRegression("logreg_from_vDF", cursor=base.cursor)
        model_test.fit(titanic_vd, ["age", "fare"], "survived")

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'logreg_from_vDF'"
        )
        assert base.cursor.fetchone()[0] == "logreg_from_vDF"

        model_test.drop()
