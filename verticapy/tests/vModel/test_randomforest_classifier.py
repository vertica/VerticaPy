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
from verticapy.learn.ensemble import RandomForestClassifier
from verticapy import vDataFrame, drop_table
import matplotlib.pyplot as plt

from verticapy import set_option

set_option("print_info", False)


@pytest.fixture(scope="module")
def rfc_data_vd(base):
    base.cursor.execute("CREATE TABLE IF NOT EXISTS public.rfc_data(Id INT, transportation VARCHAR, gender VARCHAR, \"owned cars\" INT, cost VARCHAR, income CHAR(4))")
    base.cursor.execute("INSERT INTO rfc_data VALUES (1, 'Bus', 'Male', 0, 'Cheap', 'Low')")
    base.cursor.execute("INSERT INTO rfc_data VALUES (2, 'Bus', 'Male', 1, 'Cheap', 'Med')")
    base.cursor.execute("INSERT INTO rfc_data VALUES (3, 'Train', 'Female', 1, 'Cheap', 'Med')")
    base.cursor.execute("INSERT INTO rfc_data VALUES (4, 'Bus', 'Female', 0, 'Cheap', 'Low')")
    base.cursor.execute("INSERT INTO rfc_data VALUES (5, 'Bus', 'Male', 1, 'Cheap', 'Med')")
    base.cursor.execute("INSERT INTO rfc_data VALUES (6, 'Train', 'Male', 0, 'Standard', 'Med')")
    base.cursor.execute("INSERT INTO rfc_data VALUES (7, 'Train', 'Female', 1, 'Standard', 'Med')")
    base.cursor.execute("INSERT INTO rfc_data VALUES (8, 'Car', 'Female', 1, 'Expensive', 'Hig')")
    base.cursor.execute("INSERT INTO rfc_data VALUES (9, 'Car', 'Male', 2, 'Expensive', 'Med')")
    base.cursor.execute("INSERT INTO rfc_data VALUES (10, 'Car', 'Female', 2, 'Expensive', 'Hig')")
    base.cursor.execute("COMMIT")
    
    rfc_data = vDataFrame(input_relation = 'public.rfc_data', cursor=base.cursor)
    yield rfc_data
    with warnings.catch_warnings(record=True) as w:
        drop_table(name="public.rfc_data", cursor=base.cursor)


@pytest.fixture(scope="module")
def model(base, rfc_data_vd):
    base.cursor.execute("DROP MODEL IF EXISTS rfc_model_test")

    base.cursor.execute("SELECT rf_classifier('rfc_model_test', 'public.rfc_data', 'TransPortation', '*' USING PARAMETERS exclude_columns='id, TransPortation', mtry=2, ntree=5, max_breadth=100, sampling_size=0.8, max_depth=2, min_leaf_size=2, min_info_gain=0.001, nbins=30, seed=1, id_column='id')")

    # I could use load_model but it is buggy
    model_class = RandomForestClassifier("rfc_model_test", cursor=base.cursor, n_estimators = 5,
                                         max_features = 2, max_leaf_nodes = 100, sample = 0.8,
                                         max_depth = 2, min_samples_leaf = 2, min_info_gain = 0.001, nbins = 30)
    model_class.input_relation = 'public.rfc_data'
    model_class.test_relation = model_class.input_relation
    model_class.X = ["Gender", "\"owned cars\"", "cost", "income"]
    model_class.y = "TransPortation"
    base.cursor.execute("SELECT DISTINCT {} FROM {} WHERE {} IS NOT NULL ORDER BY 1".format(
        model_class.y, model_class.input_relation, model_class.y))
    classes = base.cursor.fetchall()
    model_class.classes_ = [item[0] for item in classes]

    yield model_class
    model_class.drop()


class TestRFC:
    @pytest.mark.xfail
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

        assert conf_mat1['Bus'] == [2, 0, 0]
        assert conf_mat1['Car'] == [0, 2, 0]
        assert conf_mat1['Train'] == [2, 1, 3]

        conf_mat2 = model.confusion_matrix(cutoff=0.2)

        assert conf_mat2['Bus'] == [2, 0, 0]
        assert conf_mat2['Car'] == [0, 2, 0]
        assert conf_mat2['Train'] == [2, 1, 3]

    def test_deploySQL(self, model):
        expected_sql = "PREDICT_RF_CLASSIFIER(Gender, \"owned cars\", cost, income USING PARAMETERS model_name = 'rfc_model_test', match_by_pos = 'true')"
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self, base):
        base.cursor.execute("DROP MODEL IF EXISTS rfc_model_test_drop")
        model_test = RandomForestClassifier("rfc_model_test_drop", cursor=base.cursor)
        model_test.fit("public.rfc_data",
                       ["Gender", "\"owned cars\"", "cost", "income"],
                       "TransPortation")

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'rfc_model_test_drop'"
        )
        assert base.cursor.fetchone()[0] == "rfc_model_test_drop"

        model_test.drop()
        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'rfc_model_test_drop'"
        )
        assert base.cursor.fetchone() is None

    def test_features_importance(self, model):
        f_imp = model.features_importance()

        assert f_imp["index"] == ['income', 'owned cars', 'gender', 'cost']
        assert f_imp["importance"] == [44.85, 38.26, 16.89, 0.0]
        assert f_imp["sign"] == [1, 1, 1, 0]
        plt.close()

    def test_lift_chart(self, model):
        lift_ch = model.lift_chart(pos_label = "Bus")

        assert lift_ch["decision_boundary"][300] == pytest.approx(0.3)
        assert lift_ch["positive_prediction_ratio"][300] == pytest.approx(0.25)
        assert lift_ch["lift"][300] == pytest.approx(2.5)
        assert lift_ch["decision_boundary"][900] == pytest.approx(0.9)
        assert lift_ch["positive_prediction_ratio"][900] == pytest.approx(1.0)
        assert lift_ch["lift"][900] == pytest.approx(1.25)
        plt.close()

    @pytest.mark.skip(reason="test not implemented")
    def test_plot(self):
        pass

    @pytest.mark.skip(reason="Model Conversion for RandomForestClassifier is not yet supported.")
    def test_to_sklearn(self, model):
        md = model.to_sklearn()
        model.cursor.execute(
            "SELECT PREDICT_RF_CLASSIFIER('Male', 0, 'Cheap', 'Low' USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model.name
            )
        )
        prediction = model.cursor.fetchone()[0]
        assert prediction == pytest.approx(md.predict([['Bus', 'Male', 0, 'Cheap', 'Low']])[0])

        # 'predict_proba'

    @pytest.mark.skip(reason="to_shapExplainer is not available for RandomForestClassifier")
    def test_shapExplainer(self, model):
        explainer = model.shapExplainer()
        assert explainer.expected_value[0] == pytest.approx(-0.22667938806360247)

    def test_get_attr(self, model):
        attr = model.get_attr()
        assert attr["attr_name"] == [
            "tree_count",
            "rejected_row_count",
            "accepted_row_count",
            "call_string",
            "details"
        ]
        assert attr["attr_fields"] == [
            "tree_count",
            "rejected_row_count",
            "accepted_row_count",
            "call_string",
            "predictor, type"
        ]
        assert attr["#_of_rows"] == [1, 1, 1, 1, 4]

        details = model.get_attr("details")
        assert details["predictor"] == ['gender', 'owned cars', 'cost', 'income']
        assert details["type"] == ['char or varchar', 'int', 'char or varchar', 'char or varchar']

        assert model.get_attr("accepted_row_count")["accepted_row_count"][0] == 10
        assert model.get_attr("rejected_row_count")["rejected_row_count"][0] == 0
        assert model.get_attr("tree_count")["tree_count"][0] == 5
        assert (
            model.get_attr("call_string")["call_string"][0]
            == "SELECT rf_classifier('public.rfc_model_test', 'public.rfc_data', '\"transportation\"', '*' USING PARAMETERS exclude_columns='id, TransPortation', ntree=5, mtry=2, sampling_size=0.8, max_depth=2, max_breadth=100, min_leaf_size=2, min_info_gain=0.001, nbins=30);"
        )

    def test_get_params(self, model):
        params = model.get_params()

        assert params == {
            'n_estimators': 5,
            'max_features': 2,
            'max_leaf_nodes': 100,
            'sample': 0.8,
            'max_depth': 2,
            'min_samples_leaf': 2,
            'min_info_gain': 0.001,
            'nbins': 30
        }

    def test_prc_curve(self, model):
        prc = model.prc_curve(pos_label = 'Car')

        assert prc["threshold"][300] == pytest.approx(0.299)
        assert prc["recall"][300] == pytest.approx(1.0)
        assert prc["precision"][300] == pytest.approx(1.0)
        assert prc["threshold"][800] == pytest.approx(0.799)
        assert prc["recall"][800] == pytest.approx(0.333333333333333)
        assert prc["precision"][800] == pytest.approx(1.0)
        plt.close()

    def test_predict(self, rfc_data_vd, model):
        rfc_data_copy = rfc_data_vd.copy()

        model.predict(rfc_data_copy, name="pred_probability")
        assert rfc_data_copy["pred_probability"].mode() == 'Train'

        model.predict(rfc_data_copy, name="pred_class1", cutoff=0.7)
        assert rfc_data_copy["pred_class1"].mode() == 'Train'

        model.predict(rfc_data_copy, name="pred_class2", cutoff=0.3)
        assert rfc_data_copy["pred_class2"].mode() == 'Train'

    def test_roc_curve(self, model):
        roc = model.roc_curve(pos_label = 'Train')

        assert roc["threshold"][100] == pytest.approx(0.1)
        assert roc["false_positive"][100] == pytest.approx(0.857142857142857)
        assert roc["true_positive"][100] == pytest.approx(1.0)
        assert roc["threshold"][700] == pytest.approx(0.7)
        assert roc["false_positive"][700] == pytest.approx(0.0)
        assert roc["true_positive"][700] == pytest.approx(0.0)
        plt.close()

    def test_score(self, model):
        assert model.score(cutoff=0.9, method="accuracy") == pytest.approx(0.7)
        assert model.score(cutoff=0.1, method="accuracy") == pytest.approx(0.7)
        assert model.score(cutoff=0.9, method="auc", pos_label = 'Train') == pytest.approx(
            0.9523809523809523
        )
        assert model.score(cutoff=0.1, method="auc", pos_label = 'Train') == pytest.approx(
            0.9523809523809523
        )
        assert model.score(cutoff=0.9, method="best_cutoff", pos_label = 'Train') == pytest.approx(0.415)
        assert model.score(cutoff=0.1, method="best_cutoff", pos_label = 'Train') == pytest.approx(0.415)
        assert model.score(cutoff=0.9, method="bm", pos_label = 'Train') == pytest.approx(0.0)
        assert model.score(cutoff=0.1, method="bm", pos_label = 'Train') == pytest.approx(0.0)
        assert model.score(cutoff=0.9, method="csi", pos_label = 'Train') == pytest.approx(0.0)
        assert model.score(cutoff=0.1, method="csi", pos_label = 'Train') == pytest.approx(0.0)
        assert model.score(cutoff=0.9, method="f1", pos_label = 'Train') == pytest.approx(0.0)
        assert model.score(cutoff=0.1, method="f1", pos_label = 'Train') == pytest.approx(0.0)
        assert model.score(cutoff=0.9, method="logloss", pos_label = 'Train') == pytest.approx(
            0.199499611573561
        )
        assert model.score(cutoff=0.1, method="logloss", pos_label = 'Train') == pytest.approx(
            0.199499611573561
        )
        assert model.score(cutoff=0.9, method="mcc", pos_label = 'Train') == pytest.approx(0.0)
        assert model.score(cutoff=0.1, method="mcc", pos_label = 'Train') == pytest.approx(0.0)
        assert model.score(cutoff=0.9, method="mk", pos_label = 'Train') == pytest.approx(0.0)
        assert model.score(cutoff=0.1, method="mk", pos_label = 'Train') == pytest.approx(0.0)
        assert model.score(cutoff=0.9, method="npv", pos_label = 'Train') == pytest.approx(0.0)
        assert model.score(cutoff=0.1, method="npv", pos_label = 'Train') == pytest.approx(0.0)
        assert model.score(cutoff=0.9, method="prc_auc", pos_label = 'Train') == pytest.approx(
            0.9333333333333335
        )
        assert model.score(cutoff=0.1, method="prc_auc", pos_label = 'Train') == pytest.approx(
            0.9333333333333335
        )
        assert model.score(cutoff=0.9, method="precision", pos_label = 'Train') == pytest.approx(0.0)
        assert model.score(cutoff=0.1, method="precision", pos_label = 'Train') == pytest.approx(0.0)
        assert model.score(cutoff=0.9, method="specificity", pos_label = 'Train') == pytest.approx(1.0)
        assert model.score(cutoff=0.1, method="specificity", pos_label = 'Train') == pytest.approx(1.0)

    @pytest.mark.skip(reason="test not implemented")
    def test_set_cursor(self):
        pass

    def test_set_params(self, model):
        model.set_params({"nbins": 1000})

        assert model.get_params()["nbins"] == 1000

    def test_model_from_vDF(self, base, rfc_data_vd):
        base.cursor.execute("DROP MODEL IF EXISTS rfc_from_vDF")
        model_test = RandomForestClassifier("rfc_from_vDF", cursor=base.cursor)
        model_test.fit(rfc_data_vd,
                       ["Gender", "\"owned cars\"", "cost", "income"],
                       "TransPortation")

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'rfc_from_vDF'"
        )
        assert base.cursor.fetchone()[0] == "rfc_from_vDF"

        model_test.drop()

    def test_export_graphviz(self, model):
        gvz_tree_0 = model.export_graphviz(tree_id = 0)
        expected_gvz_0 = 'digraph Tree{\n1 [label = "owned cars < 1.066667 ?", color="blue"];\n1 -> 2 [label = "yes", color = "black"];\n1 -> 3 [label = "no", color = "black"];\n2 [label = "gender == Female ?", color="blue"];\n2 -> 4 [label = "yes", color = "black"];\n2 -> 5 [label = "no", color = "black"];\n4 [label = "prediction: Train, probability: 0.666667", color="red"];\n5 [label = "prediction: Bus, probability: 1", color="red"];\n3 [label = "prediction: Car, probability: 1", color="red"];\n}'

        assert gvz_tree_0 == expected_gvz_0

    def test_get_tree(self, model):
        tree_1 = model.get_tree(tree_id = 1)

        assert tree_1['prediction'] == [None, 'Bus', 'Car']

    @pytest.mark.skip(reason="test not implemented")
    def test_plot_tree(self, model):
        pass

