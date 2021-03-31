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
from verticapy.learn.ensemble import XGBoostClassifier
from verticapy import vDataFrame, drop, version, set_option, vertica_conn
from verticapy.tests.conftest import get_version
import matplotlib.pyplot as plt

set_option("print_info", False)

@pytest.fixture(scope="module")
def xgbc_data_vd(base):
    base.cursor.execute("DROP TABLE IF EXISTS public.xgbc_data")
    base.cursor.execute(
        'CREATE TABLE IF NOT EXISTS public.xgbc_data(Id INT, transportation VARCHAR, gender VARCHAR, "owned cars" INT, cost VARCHAR, income CHAR(4))'
    )
    base.cursor.execute(
        "INSERT INTO xgbc_data VALUES (1, 'Bus', 'Male', 0, 'Cheap', 'Low')"
    )
    base.cursor.execute(
        "INSERT INTO xgbc_data VALUES (2, 'Bus', 'Male', 1, 'Cheap', 'Med')"
    )
    base.cursor.execute(
        "INSERT INTO xgbc_data VALUES (3, 'Train', 'Female', 1, 'Cheap', 'Med')"
    )
    base.cursor.execute(
        "INSERT INTO xgbc_data VALUES (4, 'Bus', 'Female', 0, 'Cheap', 'Low')"
    )
    base.cursor.execute(
        "INSERT INTO xgbc_data VALUES (5, 'Bus', 'Male', 1, 'Cheap', 'Med')"
    )
    base.cursor.execute(
        "INSERT INTO xgbc_data VALUES (6, 'Train', 'Male', 0, 'Standard', 'Med')"
    )
    base.cursor.execute(
        "INSERT INTO xgbc_data VALUES (7, 'Train', 'Female', 1, 'Standard', 'Med')"
    )
    base.cursor.execute(
        "INSERT INTO xgbc_data VALUES (8, 'Car', 'Female', 1, 'Expensive', 'Hig')"
    )
    base.cursor.execute(
        "INSERT INTO xgbc_data VALUES (9, 'Car', 'Male', 2, 'Expensive', 'Med')"
    )
    base.cursor.execute(
        "INSERT INTO xgbc_data VALUES (10, 'Car', 'Female', 2, 'Expensive', 'Hig')"
    )
    base.cursor.execute("COMMIT")

    xgbc_data = vDataFrame(input_relation="public.xgbc_data", cursor=base.cursor)
    yield xgbc_data
    with warnings.catch_warnings(record=True) as w:
        drop(name="public.xgbc_data", cursor=base.cursor)

@pytest.fixture(scope="module")
def titanic_vd(base):
    from verticapy.datasets import load_titanic

    titanic = load_titanic(cursor=base.cursor)
    yield titanic
    with warnings.catch_warnings(record=True) as w:
        drop(name="public.titanic", cursor=base.cursor)

@pytest.fixture(scope="module")
def model(base, xgbc_data_vd):
    base.cursor.execute("DROP MODEL IF EXISTS xgbc_model_test")

    base.cursor.execute(
        "SELECT xgb_classifier('xgbc_model_test', 'public.xgbc_data', 'TransPortation', '*' USING PARAMETERS exclude_columns='id, TransPortation', min_split_loss=0.1, max_ntree=3, learning_rate=0.2, sampling_size=1, max_depth=6, nbins=40, seed=1, id_column='id')"
    )

    # I could use load_model but it is buggy
    model_class = XGBoostClassifier(
        "xgbc_model_test",
        cursor=base.cursor,
        max_ntree=3,
        min_split_loss=0.1,
        learning_rate=0.2,
        sample=1.0,
        max_depth=6,
        nbins=40,
    )
    model_class.input_relation = "public.xgbc_data"
    model_class.test_relation = model_class.input_relation
    model_class.X = ["Gender", '"owned cars"', "cost", "income"]
    model_class.y = "TransPortation"
    base.cursor.execute(
        "SELECT DISTINCT {} FROM {} WHERE {} IS NOT NULL ORDER BY 1".format(
            model_class.y, model_class.input_relation, model_class.y
        )
    )
    classes = base.cursor.fetchall()
    model_class.classes_ = [item[0] for item in classes]

    yield model_class
    model_class.drop()

@pytest.mark.skipif(get_version()[0] < 10 or (get_version()[0] == 10 and get_version()[1] == 0), reason="requires vertica 10.1 or higher")
class TestXGBC:
    def test_classification_report(self, model):
        cls_rep1 = model.classification_report().transpose()

        assert cls_rep1["auc"][0] == pytest.approx(1.0)
        assert cls_rep1["prc_auc"][0] == pytest.approx(1.0)
        assert cls_rep1["accuracy"][0] == pytest.approx(1.0)
        assert cls_rep1["log_loss"][0] == pytest.approx(0.127588478759147)
        assert cls_rep1["precision"][0] == pytest.approx(1.0)
        assert cls_rep1["recall"][0] == pytest.approx(1.0)
        assert cls_rep1["f1_score"][0] == pytest.approx(1.0)
        assert cls_rep1["mcc"][0] == pytest.approx(1.0)
        assert cls_rep1["informedness"][0] == pytest.approx(1.0)
        assert cls_rep1["markedness"][0] == pytest.approx(1.0)
        assert cls_rep1["csi"][0] == pytest.approx(1.0)
        assert cls_rep1["cutoff"][0] == pytest.approx(0.681)

        cls_rep2 = model.classification_report(cutoff=0.681).transpose()

        assert cls_rep2["cutoff"][0] == pytest.approx(0.681)

    def test_confusion_matrix(self, model):
        conf_mat1 = model.confusion_matrix()

        assert conf_mat1["Bus"] == [4, 0, 0]
        assert conf_mat1["Car"] == [0, 3, 0]
        assert conf_mat1["Train"] == [0, 0, 3]

        conf_mat2 = model.confusion_matrix(cutoff=0.2)

        assert conf_mat2["Bus"] == [4, 0, 0]
        assert conf_mat2["Car"] == [0, 3, 0]
        assert conf_mat2["Train"] == [0, 0, 3]

    def test_contour(self, base, titanic_vd):
        model_test = XGBoostClassifier("model_contour", cursor=base.cursor)
        model_test.drop()
        model_test.fit(
            titanic_vd,
            ["age", "fare",],
            "survived",
        )
        result = model_test.contour()
        assert len(result.get_default_bbox_extra_artists()) == 38
        model_test.drop()

    def test_deploySQL(self, model):
        expected_sql = "PREDICT_XGB_CLASSIFIER(Gender, \"owned cars\", cost, income USING PARAMETERS model_name = 'xgbc_model_test', match_by_pos = 'true')"
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self, base):
        base.cursor.execute("DROP MODEL IF EXISTS xgbc_model_test_drop")
        model_test = XGBoostClassifier("xgbc_model_test_drop", cursor=base.cursor)
        model_test.fit(
            "public.xgbc_data",
            ["Gender", '"owned cars"', "cost", "income"],
            "TransPortation",
        )

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'xgbc_model_test_drop'"
        )
        assert base.cursor.fetchone()[0] == "xgbc_model_test_drop"

        model_test.drop()
        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'xgbc_model_test_drop'"
        )
        assert base.cursor.fetchone() is None

    @pytest.mark.skip(reason="not yet available.")
    def test_features_importance(self, model):
        f_imp = model.features_importance()

        assert f_imp["index"] == ["cost", "owned cars", "gender", "income"]
        assert f_imp["importance"] == [75.76, 15.15, 9.09, 0.0]
        assert f_imp["sign"] == [1, 1, 1, 0]
        plt.close("all")

    def test_lift_chart(self, model):
        lift_ch = model.lift_chart(pos_label="Bus", nbins=1000)

        assert lift_ch["decision_boundary"][300] == pytest.approx(0.3)
        assert lift_ch["positive_prediction_ratio"][300] == pytest.approx(0.0)
        assert lift_ch["lift"][300] == pytest.approx(2.5)
        plt.close("all")

    @pytest.mark.skip(reason="not yet available.")
    def test_to_sklearn(self, model):
        md = model.to_sklearn()
        model.cursor.execute(
            "SELECT PREDICT_XGB_CLASSIFIER('Male', 0, 'Cheap', 'Low' USING PARAMETERS model_name = '{}', match_by_pos=True)".format(
                model.name
            )
        )
        prediction = model.cursor.fetchone()[0]
        assert prediction == pytest.approx(
            md.predict([["Bus", "Male", 0, "Cheap", "Low"]])[0]
        )

    @pytest.mark.skip(reason="the SQL prediction doesn't always match the Vertica prediction.")
    def test_to_sql(self, model, titanic_vd):
        model_test = XGBoostClassifier("xgb_sql_test", cursor=model.cursor)
        model_test.drop()
        model_test.fit(titanic_vd, ["age", "fare", "sex"], "survived")
        model.cursor.execute(
            "SELECT PREDICT_XGB_CLASSIFIER(* USING PARAMETERS model_name = 'xgb_sql_test', match_by_pos=True, class=1, type='probability')::float, {}::float FROM (SELECT 30.0 AS age, 45.0 AS fare, 'male' AS sex) x".format(
                model_test.to_sql()
            )
        )
        prediction = model.cursor.fetchone()
        assert prediction[0] == pytest.approx(prediction[1])
        model_test.drop()

    @pytest.mark.skip(reason="not yet available.")
    def test_shapExplainer(self, model):
        explainer = model.shapExplainer()
        assert explainer.expected_value[0] == pytest.approx(
            -0.22667938806360247
        )

    def test_get_attr(self, model):
        attr = model.get_attr()
        assert attr["attr_name"] == [
            "tree_count",
            "rejected_row_count",
            "accepted_row_count",
            "call_string",
            "details",
        ]
        assert attr["attr_fields"] == [
            "tree_count",
            "rejected_row_count",
            "accepted_row_count",
            "call_string",
            "predictor, type",
        ]
        assert attr["#_of_rows"] == [1, 1, 1, 1, 4]

        details = model.get_attr("details")
        assert details["predictor"] == ["gender", "owned cars", "cost", "income"]
        assert details["type"] == [
            "char or varchar",
            "int",
            "char or varchar",
            "char or varchar",
        ]

        assert model.get_attr("accepted_row_count")["accepted_row_count"][0] == 10
        assert model.get_attr("rejected_row_count")["rejected_row_count"][0] == 0
        assert model.get_attr("tree_count")["tree_count"][0] == 3
        assert (
            model.get_attr("call_string")["call_string"][0]
            == "xgb_classifier('public.xgbc_model_test', 'public.xgbc_data', '\"transportation\"', '*' USING PARAMETERS exclude_columns='id, TransPortation', max_ntree=3, max_depth=6, nbins=40, objective=squarederror, split_proposal_method=global, epsilon=0.001, learning_rate=0.2, min_split_loss=0.1, weight_reg=0, sampling_size=1, seed=1, id_column='id')"
        )

    def test_get_params(self, model):
        assert model.get_params() == {
            "max_ntree": 3,
            "min_split_loss": 0.1,
            "learning_rate": 0.2,
            "sample": 1.0,
            "max_depth": 6,
            "nbins": 40,
            "objective": "squarederror",
            "split_proposal_method": "global",
            "tol": 0.001,
            "weight_reg": 0.0,
        }

    def test_prc_curve(self, model):
        prc = model.prc_curve(pos_label="Car", nbins=1000)

        assert prc["threshold"][300] == pytest.approx(0.299)
        assert prc["recall"][300] == pytest.approx(1.0)
        assert prc["precision"][300] == pytest.approx(1.0)
        plt.close("all")

    def test_predict(self, xgbc_data_vd, model):
        xgbc_data_copy = xgbc_data_vd.copy()

        model.predict(xgbc_data_copy, name="pred_probability")
        assert xgbc_data_copy["pred_probability"].mode() == "Bus"

        model.predict(xgbc_data_copy, name="pred_class1", cutoff=0.7)
        assert xgbc_data_copy["pred_class1"].mode() == "Bus"

        model.predict(xgbc_data_copy, name="pred_class2", cutoff=0.3)
        assert xgbc_data_copy["pred_class2"].mode() == "Bus"

    def test_roc_curve(self, model):
        roc = model.roc_curve(pos_label="Train", nbins=1000)

        assert roc["threshold"][100] == pytest.approx(0.1)
        assert roc["false_positive"][100] == pytest.approx(1.0)
        assert roc["true_positive"][100] == pytest.approx(1.0)
        assert roc["threshold"][700] == pytest.approx(0.7)
        assert roc["false_positive"][700] == pytest.approx(0.0)
        assert roc["true_positive"][700] == pytest.approx(0.0)
        plt.close("all")

    def test_cutoff_curve(self, model):
        cutoff_curve = model.cutoff_curve(pos_label="Train", nbins=1000)

        assert cutoff_curve["threshold"][100] == pytest.approx(0.1)
        assert cutoff_curve["false_positive"][100] == pytest.approx(1.0)
        assert cutoff_curve["true_positive"][100] == pytest.approx(1.0)
        assert cutoff_curve["threshold"][700] == pytest.approx(0.7)
        assert cutoff_curve["false_positive"][700] == pytest.approx(0.0)
        assert cutoff_curve["true_positive"][700] == pytest.approx(0.0)
        plt.close("all")

    def test_score(self, model):
        assert model.score(cutoff=0.9, method="accuracy") == pytest.approx(1.0)
        assert model.score(cutoff=0.1, method="accuracy") == pytest.approx(1.0)
        assert model.score(
            cutoff=0.9, method="auc", pos_label="Train"
        ) == pytest.approx(1.0)
        assert model.score(
            cutoff=0.1, method="auc", pos_label="Train"
        ) == pytest.approx(1.0)
        assert model.score(
            cutoff=0.9, method="best_cutoff", pos_label="Train"
        ) == pytest.approx(0.633)
        assert model.score(
            cutoff=0.1, method="best_cutoff", pos_label="Train"
        ) == pytest.approx(0.633)
        assert model.score(
            cutoff=0.633, method="bm", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.1, method="bm", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.9, method="csi", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.1, method="csi", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.9, method="f1", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.1, method="f1", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.9, method="logloss", pos_label="Train"
        ) == pytest.approx(0.111961142833969)
        assert model.score(
            cutoff=0.1, method="logloss", pos_label="Train"
        ) == pytest.approx(0.111961142833969)
        assert model.score(
            cutoff=0.9, method="mcc", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.1, method="mcc", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.9, method="mk", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.1, method="mk", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.9, method="npv", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.1, method="npv", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.9, method="prc_auc", pos_label="Train"
        ) == pytest.approx(1.0)
        assert model.score(
            cutoff=0.1, method="prc_auc", pos_label="Train"
        ) == pytest.approx(1.0)
        assert model.score(
            cutoff=0.9, method="precision", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.1, method="precision", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.9, method="specificity", pos_label="Train"
        ) == pytest.approx(1.0)
        assert model.score(
            cutoff=0.1, method="specificity", pos_label="Train"
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
        model.set_params({"nbins": 1000})

        assert model.get_params()["nbins"] == 1000

    def test_model_from_vDF(self, base, xgbc_data_vd):
        base.cursor.execute("DROP MODEL IF EXISTS xgbc_from_vDF")
        model_test = XGBoostClassifier("xgbc_from_vDF", cursor=base.cursor)
        model_test.fit(
            xgbc_data_vd,
            ["Gender", '"owned cars"', "cost", "income"],
            "TransPortation",
        )

        base.cursor.execute(
            "SELECT model_name FROM models WHERE model_name = 'xgbc_from_vDF'"
        )
        assert base.cursor.fetchone()[0] == "xgbc_from_vDF"

        model_test.drop()

    def test_export_graphviz(self, model):
        gvz_tree_0 = model.export_graphviz(tree_id=0)
        expected_gvz_0 = 'digraph Tree{\n1 [label = "cost == Expensive ?", color="blue"];\n1 -> 2 [label = "yes", color = "black"];\n1 -> 3 [label = "no", color = "black"];\n2 [label = "log_odds: Bus:-1.66667,Car:3.33333,Train:-1.42857", color="red"];\n3 [label = "cost == Cheap ?", color="blue"];\n3 -> 6 [label = "yes", color = "black"];\n3 -> 7 [label = "no", color = "black"];\n6 [label = "gender == Female ?", color="blue"];\n6 -> 10 [label = "yes", color = "black"];\n6 -> 11 [label = "no", color = "black"];\n10 [label = "income == Low ?", color="blue"];\n10 -> 14 [label = "yes", color = "black"];\n10 -> 15 [label = "no", color = "black"];\n14 [label = "log_odds: Bus:2.5,Car:-1.42857,Train:-1.42857", color="red"];\n15 [label = "log_odds: Bus:-1.66667,Car:-1.42857,Train:3.33333", color="red"];\n11 [label = "log_odds: Bus:2.5,Car:-1.42857,Train:-1.42857", color="red"];\n7 [label = "log_odds: Bus:-1.66667,Car:-1.42857,Train:3.33333", color="red"];\n}'

        assert gvz_tree_0 == expected_gvz_0

    def test_get_tree(self, model):
        tree_1 = model.get_tree(tree_id=1)

        assert tree_1["prediction"] == [None, "", None, None, "", None, "", "", ""]

    def test_plot_tree(self, model):
        result = model.plot_tree()
        assert result.by_attr()[0:3] == "[1]"
