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
from verticapy import drop_table
import matplotlib.pyplot as plt

from verticapy import set_option

set_option("print_info", False)


@pytest.fixture(scope="module")
def iris_vd(base):
    from verticapy.learn.datasets import load_iris

    iris = load_iris(cursor=base.cursor)
    yield iris
    with warnings.catch_warnings(record=True) as w:
        drop_table(name="public.iris", cursor=base.cursor)


@pytest.fixture(scope="module")
def model(base, iris_vd):
    base.cursor.execute("DROP MODEL IF EXISTS rfc_model_test")
    base.cursor.execute("CREATE TABLE IF NOT EXISTS public.iris2 AS SELECT row_number() over() AS id, * from iris order by SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species")

    base.cursor.execute("SELECT rf_classifier('rfc_model_test', 'public.iris2', 'Species', '*' USING PARAMETERS exclude_columns='id, Species', mtry=2, ntree=5, max_leaf_nodes=100, sampling_size=0.7, max_depth=4, min_leaf_size=2, min_info_gain=0.001, nbins=30, seed=1, id_column='id')")

    # I could use load_model but it is buggy
    model_class = RandomForestClassifier("rfc_model_test", cursor=base.cursor, n_estimators = 5,
                                         max_features = 2, max_leaf_nodes = 100, sample = 0.7,
                                         max_depth = 4, min_samples_leaf = 2, min_info_gain = 0.001, nbins = 30)
    model_class.input_relation = 'public.iris2'
    model_class.test_relation = model_class.input_relation
    model_class.X = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    model_class.y = "Species"
    base.cursor.execute("SELECT DISTINCT {} FROM {} WHERE {} IS NOT NULL ORDER BY 1".format(
        model_class.y, model_class.input_relation, model_class.y))
    classes = base.cursor.fetchall()
    model_class.classes_ = [item[0] for item in classes]

    yield model_class
    model_class.drop()
    with warnings.catch_warnings(record=True) as w:
        drop_table(name="public.iris2", cursor=base.cursor)


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

        assert conf_mat1['Iris-setosa'] == [50, 0, 0]
        assert conf_mat1['Iris-versicolor'] == [0, 48, 5]
        assert conf_mat1['Iris-virginica'] == [0, 2, 45]

        conf_mat2 = model.confusion_matrix(cutoff=0.2)

        assert conf_mat2['Iris-setosa'] == [50, 0, 0]
        assert conf_mat2['Iris-versicolor'] == [0, 48, 5]
        assert conf_mat2['Iris-virginica'] == [0, 2, 45]

    def test_deploySQL(self, model):
        expected_sql = "PREDICT_RF_CLASSIFIER(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm USING PARAMETERS model_name = 'rfc_model_test', match_by_pos = 'true')"
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self, base):
        base.cursor.execute("DROP MODEL IF EXISTS rfc_model_test_drop")
        model_test = RandomForestClassifier("rfc_model_test_drop", cursor=base.cursor)
        model_test.fit("public.iris",
                       ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
                       "Species")

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

        assert f_imp["index"] == ["petalwidthcm", "petallengthcm", "sepalwidthcm", "sepallengthcm"]
        assert f_imp["importance"] == [98.18, 1.1, 0.46, 0.26]
        assert f_imp["sign"] == [1, 1, 1, 1]
        plt.close()

    def test_lift_chart(self, model):
        lift_ch = model.lift_chart(pos_label = "Iris-setosa")

        assert lift_ch["decision_boundary"][10] == pytest.approx(0.01)
        assert lift_ch["positive_prediction_ratio"][10] == pytest.approx(1.0)
        assert lift_ch["lift"][10] == pytest.approx(3.0)
        assert lift_ch["decision_boundary"][900] == pytest.approx(0.9)
        assert lift_ch["positive_prediction_ratio"][900] == pytest.approx(1.0)
        assert lift_ch["lift"][900] == pytest.approx(3.0)
        plt.close()

    @pytest.mark.skip(reason="test not implemented")
    def test_plot(self):
        pass

    @pytest.mark.skip(reason="Model Conversion for RandomForestClassifier is not yet supported.")
    def test_to_sklearn(self, model):
        md = model.to_sklearn()
        model.cursor.execute(
            "SELECT PREDICT_RF_CLASSIFIER(11.0, 1993.0, 1.2, 3.4 USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model.name
            )
        )
        prediction = model.cursor.fetchone()[0]
        assert prediction == pytest.approx(md.predict([[11.0, 1993.0, 1.2, 3.4]])[0])

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
        assert details["predictor"] == ['sepallengthcm', 'sepalwidthcm', 'petallengthcm', 'petalwidthcm']
        assert details["type"] == ['float or numeric', 'float or numeric', 'float or numeric', 'float or numeric']

        assert model.get_attr("accepted_row_count")["accepted_row_count"][0] == 150
        assert model.get_attr("rejected_row_count")["rejected_row_count"][0] == 0
        assert model.get_attr("tree_count")["tree_count"][0] == 5
        assert (
            model.get_attr("call_string")["call_string"][0]
            == "SELECT rf_classifier('public.rfc_model_test', 'public.iris2', '\"species\"', '*' USING PARAMETERS exclude_columns='id, Species', ntree=5, mtry=2, sampling_size=0.7, max_depth=4, max_breadth=32, min_leaf_size=2, min_info_gain=0.001, nbins=30);"
        )

    def test_get_params(self, model):
        params = model.get_params()

        assert params == {
            'n_estimators': 5,
            'max_features': 2,
            'max_leaf_nodes': 100,
            'sample': 0.7,
            'max_depth': 4,
            'min_samples_leaf': 2,
            'min_info_gain': 0.001,
            'nbins': 30
        }

    def test_prc_curve(self, model):
        prc = model.prc_curve(pos_label = 'Iris-versicolor')

        assert prc["threshold"][10] == pytest.approx(0.009)
        assert prc["recall"][10] == pytest.approx(1.0)
        assert prc["precision"][10] == pytest.approx(0.819672131147541)
        assert prc["threshold"][900] == pytest.approx(0.899)
        assert prc["recall"][900] == pytest.approx(0.84)
        assert prc["precision"][900] == pytest.approx(1.0)
        plt.close()

    def test_predict(self, iris_vd, model):
        iris_copy = iris_vd.copy()

        model.predict(iris_copy, name="pred_probability")
        assert iris_copy["pred_probability"].mode() == 'Iris-versicolor'

        model.predict(iris_copy, name="pred_class1", cutoff=0.7)
        assert iris_copy["pred_class1"].mode() == 'Iris-versicolor'

        model.predict(iris_copy, name="pred_class2", cutoff=0.3)
        assert iris_copy["pred_class2"].mode() == 'Iris-versicolor'

    def test_roc_curve(self, model):
        roc = model.roc_curve(pos_label = 'Iris-virginica')

        assert roc["threshold"][100] == pytest.approx(0.1)
        assert roc["false_positive"][100] == pytest.approx(0.03)
        assert roc["true_positive"][100] == pytest.approx(1.0)
        assert roc["threshold"][700] == pytest.approx(0.7)
        assert roc["false_positive"][700] == pytest.approx(0.01)
        assert roc["true_positive"][700] == pytest.approx(0.9)
        plt.close()

    def test_score(self, model):
        assert model.score(cutoff=0.9, method="accuracy") == pytest.approx(
            0.953333333333333
        )
        assert model.score(cutoff=0.1, method="accuracy") == pytest.approx(
            0.953333333333333
        )
        assert model.score(cutoff=0.9, method="auc", pos_label = 'Iris-virginica') == pytest.approx(
            0.9969000000000001
        )
        assert model.score(cutoff=0.1, method="auc", pos_label = 'Iris-virginica') == pytest.approx(
            0.9969000000000001
        )
        assert model.score(cutoff=0.9, method="best_cutoff", pos_label = 'Iris-virginica') == pytest.approx(0.181)
        assert model.score(cutoff=0.1, method="best_cutoff", pos_label = 'Iris-virginica') == pytest.approx(0.181)
        assert model.score(cutoff=0.9, method="bm", pos_label = 'Iris-virginica') == pytest.approx(0.0)
        assert model.score(cutoff=0.1, method="bm", pos_label = 'Iris-virginica') == pytest.approx(0.0)
        assert model.score(cutoff=0.9, method="csi", pos_label = 'Iris-virginica') == pytest.approx(0.0)
        assert model.score(cutoff=0.1, method="csi", pos_label = 'Iris-virginica') == pytest.approx(0.0)
        assert model.score(cutoff=0.9, method="f1", pos_label = 'Iris-virginica') == pytest.approx(0.0)
        assert model.score(cutoff=0.1, method="f1", pos_label = 'Iris-virginica') == pytest.approx(0.0)
        assert model.score(cutoff=0.9, method="logloss", pos_label = 'Iris-virginica') == pytest.approx(
            0.0360850163893819
        )
        assert model.score(cutoff=0.1, method="logloss", pos_label = 'Iris-virginica') == pytest.approx(
            0.0360850163893819
        )
        assert model.score(cutoff=0.9, method="mcc", pos_label = 'Iris-virginica') == pytest.approx(0.0)
        assert model.score(cutoff=0.1, method="mcc", pos_label = 'Iris-virginica') == pytest.approx(0.0)
        assert model.score(cutoff=0.9, method="mk", pos_label = 'Iris-virginica') == pytest.approx(0.0)
        assert model.score(cutoff=0.1, method="mk", pos_label = 'Iris-virginica') == pytest.approx(0.0)
        assert model.score(cutoff=0.9, method="npv", pos_label = 'Iris-virginica') == pytest.approx(0.0)
        assert model.score(cutoff=0.1, method="npv", pos_label = 'Iris-virginica') == pytest.approx(0.0)
        assert model.score(cutoff=0.9, method="prc_auc", pos_label = 'Iris-virginica') == pytest.approx(
            0.9937297235488114
        )
        assert model.score(cutoff=0.1, method="prc_auc", pos_label = 'Iris-virginica') == pytest.approx(
            0.9937297235488114
        )
        assert model.score(cutoff=0.9, method="precision", pos_label = 'Iris-virginica') == pytest.approx(0.0)
        assert model.score(cutoff=0.1, method="precision", pos_label = 'Iris-virginica') == pytest.approx(0.0)
        assert model.score(cutoff=0.9, method="specificity", pos_label = 'Iris-virginica') == pytest.approx(1.0)
        assert model.score(cutoff=0.1, method="specificity", pos_label = 'Iris-virginica') == pytest.approx(1.0)

    @pytest.mark.skip(reason="test not implemented")
    def test_set_cursor(self):
        pass

    def test_set_params(self, model):
        model.set_params({"nbins": 1000})

        assert model.get_params()["nbins"] == 1000

    def test_model_from_vDF(self, base, iris_vd):
        base.cursor.execute("DROP MODEL IF EXISTS rfc_from_vDF")
        model_test = RandomForestClassifier("rfc_from_vDF", cursor=base.cursor)
        model_test.fit(iris_vd,
                       ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
                       "Species")

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'rfc_from_vDF'"
        )
        assert base.cursor.fetchone()[0] == "rfc_from_vDF"

        model_test.drop()
