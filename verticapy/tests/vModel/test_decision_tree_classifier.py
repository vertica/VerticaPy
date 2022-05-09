# (c) Copyright [2018-2022] Micro Focus or one of its affiliates.
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

# Pytest
import pytest

# Other Modules
import matplotlib.pyplot as plt

# VerticaPy
from verticapy import (
    vDataFrame,
    drop,
    set_option,
)
from verticapy.connect import current_cursor
from verticapy.datasets import load_titanic, load_dataset_cl
from verticapy.learn.tree import DecisionTreeClassifier

set_option("print_info", False)


@pytest.fixture(scope="module")
def dtc_data_vd():
    dtc_data = load_dataset_cl(table_name="dtc_data", schema="public")
    yield dtc_data
    drop(name="public.dtc_data", method="table")


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(name="public.titanic",)


@pytest.fixture(scope="module")
def model(dtc_data_vd):
    current_cursor().execute("DROP MODEL IF EXISTS decision_tc_model_test")

    current_cursor().execute(
        "SELECT rf_classifier('decision_tc_model_test', 'public.dtc_data', 'TransPortation', '*' USING PARAMETERS exclude_columns='id, TransPortation', mtry=4, ntree=1, max_breadth=100, sampling_size=1, max_depth=6, nbins=40, seed=1, id_column='id')"
    )

    # I could use load_model but it is buggy
    model_class = DecisionTreeClassifier(
        "decision_tc_model_test",
        max_features=4,
        max_leaf_nodes=100,
        max_depth=6,
        min_samples_leaf=1,
        min_info_gain=0,
        nbins=40,
    )
    model_class.input_relation = "public.dtc_data"
    model_class.test_relation = model_class.input_relation
    model_class.X = ['"Gender"', '"owned cars"', '"cost"', '"income"']
    model_class.y = '"TransPortation"'
    current_cursor().execute(
        "SELECT DISTINCT {} FROM {} WHERE {} IS NOT NULL ORDER BY 1".format(
            model_class.y, model_class.input_relation, model_class.y
        )
    )
    classes = current_cursor().fetchall()
    model_class.classes_ = [item[0] for item in classes]

    yield model_class
    model_class.drop()


class TestDecisionTreeClassifier:
    def test_repr(self, model):
        assert (
            "SELECT rf_classifier('public.decision_tc_model_test'," in model.__repr__()
        )
        model_repr = DecisionTreeClassifier("RF_repr")
        model_repr.drop()
        assert model_repr.__repr__() == "<RandomForestClassifier>"

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

        cls_rep2 = model.classification_report(cutoff=0.2).transpose()

        assert cls_rep2["cutoff"][0] == pytest.approx(0.2)

    def test_confusion_matrix(self, model):
        conf_mat1 = model.confusion_matrix()

        assert conf_mat1["Bus"] == [4, 0, 0]
        assert conf_mat1["Car"] == [0, 3, 0]
        assert conf_mat1["Train"] == [0, 0, 3]

        conf_mat2 = model.confusion_matrix(cutoff=0.2)

        assert conf_mat2["Bus"] == [4, 0, 0]
        assert conf_mat2["Car"] == [0, 3, 0]
        assert conf_mat2["Train"] == [0, 0, 3]

    def test_contour(self, titanic_vd):
        model_test = DecisionTreeClassifier("model_contour",)
        model_test.drop()
        model_test.fit(
            titanic_vd, ["age", "fare"], "survived",
        )
        result = model_test.contour()
        assert len(result.get_default_bbox_extra_artists()) == 34
        model_test.drop()

    def test_deploySQL(self, model):
        expected_sql = 'PREDICT_RF_CLASSIFIER("Gender", "owned cars", "cost", "income" USING PARAMETERS model_name = \'decision_tc_model_test\', match_by_pos = \'true\')'
        result_sql = model.deploySQL()

        assert result_sql == expected_sql

    def test_drop(self):
        current_cursor().execute("DROP MODEL IF EXISTS decision_tc_model_test_drop")
        model_test = DecisionTreeClassifier("decision_tc_model_test_drop",)
        model_test.fit(
            "public.dtc_data",
            ["Gender", '"owned cars"', "cost", "income"],
            "TransPortation",
        )

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'decision_tc_model_test_drop'"
        )
        assert current_cursor().fetchone()[0] == "decision_tc_model_test_drop"

        model_test.drop()
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'decision_tc_model_test_drop'"
        )
        assert current_cursor().fetchone() is None

    def test_features_importance(self, model):
        f_imp = model.features_importance()

        assert f_imp["index"] == ["cost", "owned cars", "gender", "income"]
        assert f_imp["importance"] == [75.76, 15.15, 9.09, 0.0]
        assert f_imp["sign"] == [1, 1, 1, 0]
        plt.close("all")

    def test_lift_chart(self, model):
        lift_ch = model.lift_chart(pos_label="Bus", nbins=1000)

        assert lift_ch["decision_boundary"][300] == pytest.approx(0.3)
        assert lift_ch["positive_prediction_ratio"][300] == pytest.approx(1.0)
        assert lift_ch["lift"][300] == pytest.approx(2.5)
        assert lift_ch["decision_boundary"][900] == pytest.approx(0.9)
        assert lift_ch["positive_prediction_ratio"][900] == pytest.approx(1.0)
        assert lift_ch["lift"][900] == pytest.approx(2.5)
        plt.close("all")

    def test_to_python(self, model, titanic_vd):
        model_test = DecisionTreeClassifier("rfc_python_test")
        model_test.drop()
        model_test.fit(titanic_vd, ["age", "fare", "sex"], "embarked")
        current_cursor().execute(
            "SELECT PREDICT_RF_CLASSIFIER(30.0, 45.0, 'male' USING PARAMETERS model_name = 'rfc_python_test', match_by_pos=True)"
        )
        prediction = current_cursor().fetchone()[0]
        assert (
            prediction
            == model_test.to_python(return_str=False)([[30.0, 45.0, "male"]])[0]
        )
        current_cursor().execute(
            "SELECT PREDICT_RF_CLASSIFIER(30.0, 145.0, 'female' USING PARAMETERS model_name = 'rfc_python_test', match_by_pos=True)"
        )
        prediction = current_cursor().fetchone()[0]
        assert (
            prediction
            == model_test.to_python(return_str=False)([[30.0, 145.0, "female"]])[0]
        )

    def test_to_sql(self, model, titanic_vd):
        model_test = DecisionTreeClassifier("rfc_sql_test")
        model_test.drop()
        model_test.fit(titanic_vd, ["age", "fare", "sex"], "survived")
        current_cursor().execute(
            "SELECT PREDICT_RF_CLASSIFIER(* USING PARAMETERS model_name = 'rfc_sql_test', match_by_pos=True)::int, {}::int FROM (SELECT 30.0 AS age, 45.0 AS fare, 'male' AS sex) x".format(
                model_test.to_sql()
            )
        )
        prediction = current_cursor().fetchone()
        assert prediction[0] == prediction[1]
        model_test.drop()

    def test_to_memmodel(self, model):
        mmodel = model.to_memmodel()
        res = mmodel.predict(
            [["Male", 0, "Cheap", "Low"], ["Female", 3, "Expensive", "Hig"]]
        )
        res_py = model.to_python()(
            [["Male", 0, "Cheap", "Low"], ["Female", 3, "Expensive", "Hig"]]
        )
        assert res[0] == res_py[0]
        assert res[1] == res_py[1]
        res = mmodel.predict_proba(
            [["Male", 0, "Cheap", "Low"], ["Female", 3, "Expensive", "Hig"]]
        )
        res_py = model.to_python(return_proba=True)(
            [["Male", 0, "Cheap", "Low"], ["Female", 3, "Expensive", "Hig"]]
        )
        assert res[0][0] == res_py[0][0]
        assert res[0][1] == res_py[0][1]
        assert res[0][2] == res_py[0][2]
        assert res[1][0] == res_py[1][0]
        assert res[1][1] == res_py[1][1]
        assert res[1][2] == res_py[1][2]
        vdf = vDataFrame("public.dtc_data")
        vdf["prediction_sql"] = mmodel.predict_sql(
            ['"Gender"', '"owned cars"', '"cost"', '"income"']
        )
        vdf["prediction_proba_sql_0"] = mmodel.predict_proba_sql(
            ['"Gender"', '"owned cars"', '"cost"', '"income"']
        )[0]
        vdf["prediction_proba_sql_1"] = mmodel.predict_proba_sql(
            ['"Gender"', '"owned cars"', '"cost"', '"income"']
        )[1]
        vdf["prediction_proba_sql_2"] = mmodel.predict_proba_sql(
            ['"Gender"', '"owned cars"', '"cost"', '"income"']
        )[2]
        model.predict(vdf, name="prediction_vertica_sql")
        model.predict_proba(
            vdf, name="prediction_proba_vertica_sql_0", pos_label=model.classes_[0]
        )
        model.predict_proba(
            vdf, name="prediction_proba_vertica_sql_1", pos_label=model.classes_[1]
        )
        model.predict_proba(
            vdf, name="prediction_proba_vertica_sql_2", pos_label=model.classes_[2]
        )
        score = vdf.score("prediction_sql", "prediction_vertica_sql", "accuracy")
        assert score == pytest.approx(1.0)
        score = vdf.score(
            "prediction_proba_sql_0", "prediction_proba_vertica_sql_0", "r2"
        )
        assert score == pytest.approx(1.0)
        score = vdf.score(
            "prediction_proba_sql_1", "prediction_proba_vertica_sql_1", "r2"
        )
        assert score == pytest.approx(1.0)
        score = vdf.score(
            "prediction_proba_sql_2", "prediction_proba_vertica_sql_2", "r2"
        )
        assert score == pytest.approx(1.0)

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
        assert model.get_attr("tree_count")["tree_count"][0] == 1
        assert (
            model.get_attr("call_string")["call_string"][0]
            == "SELECT rf_classifier('public.decision_tc_model_test', 'public.dtc_data', 'transportation', '*' USING PARAMETERS exclude_columns='id, TransPortation', ntree=1, mtry=4, sampling_size=1, max_depth=6, max_breadth=100, min_leaf_size=1, min_info_gain=0, nbins=40);"
        )

    def test_get_params(self, model):
        params = model.get_params()

        assert params == {
            "n_estimators": 1,
            "max_features": 4,
            "max_leaf_nodes": 100,
            "sample": 1.0,
            "max_depth": 6,
            "min_samples_leaf": 1,
            "min_info_gain": 0,
            "nbins": 40,
        }

    def test_prc_curve(self, model):
        prc = model.prc_curve(pos_label="Car", nbins=1000)

        assert prc["threshold"][300] == pytest.approx(0.299)
        assert prc["recall"][300] == pytest.approx(1.0)
        assert prc["precision"][300] == pytest.approx(1.0)
        assert prc["threshold"][800] == pytest.approx(0.799)
        assert prc["recall"][800] == pytest.approx(1.0)
        assert prc["precision"][800] == pytest.approx(1.0)
        plt.close("all")

    def test_predict(self, dtc_data_vd, model):
        dtc_data_copy = dtc_data_vd.copy()

        model.predict(dtc_data_copy, name="pred")
        assert dtc_data_copy["pred"].mode() == "Bus"

        model.predict(dtc_data_copy, name="pred1", cutoff=0.7)
        assert dtc_data_copy["pred1"].mode() == "Bus"

        model.predict(dtc_data_copy, name="pred2", cutoff=0.3)
        assert dtc_data_copy["pred2"].mode() == "Bus"

    def test_predict_proba(self, dtc_data_vd, model):
        dtc_data_copy = dtc_data_vd.copy()

        model.predict_proba(dtc_data_copy, name="prob")
        assert dtc_data_copy["prob_bus"].avg() == 0.4
        assert dtc_data_copy["prob_train"].avg() == 0.3
        assert dtc_data_copy["prob_car"].avg() == 0.3

        model.predict_proba(dtc_data_copy, name="prob_bus_2", pos_label="Bus")
        assert dtc_data_copy["prob_bus_2"].avg() == 0.4

    def test_roc_curve(self, model):
        roc = model.roc_curve(pos_label="Train", nbins=1000)

        assert roc["threshold"][100] == pytest.approx(0.1)
        assert roc["false_positive"][100] == pytest.approx(0.0)
        assert roc["true_positive"][100] == pytest.approx(1.0)
        assert roc["threshold"][700] == pytest.approx(0.7)
        assert roc["false_positive"][700] == pytest.approx(0.0)
        assert roc["true_positive"][700] == pytest.approx(1.0)
        plt.close("all")

    def test_cutoff_curve(self, model):
        cutoff_curve = model.cutoff_curve(pos_label="Train", nbins=1000)
        assert cutoff_curve["threshold"][100] == pytest.approx(0.1)
        assert cutoff_curve["false_positive"][100] == pytest.approx(0.0)
        assert cutoff_curve["true_positive"][100] == pytest.approx(1.0)
        assert cutoff_curve["threshold"][700] == pytest.approx(0.7)
        assert cutoff_curve["false_positive"][700] == pytest.approx(0.0)
        assert cutoff_curve["true_positive"][700] == pytest.approx(1.0)
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
        ) == pytest.approx(0.999)
        assert model.score(
            cutoff=0.1, method="best_cutoff", pos_label="Train"
        ) == pytest.approx(0.999)
        assert model.score(cutoff=0.9, method="bm", pos_label="Train") == pytest.approx(
            0.0
        )
        assert model.score(cutoff=0.1, method="bm", pos_label="Train") == pytest.approx(
            0.0
        )
        assert model.score(
            cutoff=0.9, method="csi", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.1, method="csi", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(cutoff=0.9, method="f1", pos_label="Train") == pytest.approx(
            0.0
        )
        assert model.score(cutoff=0.1, method="f1", pos_label="Train") == pytest.approx(
            0.0
        )
        assert model.score(
            cutoff=0.9, method="logloss", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.1, method="logloss", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.9, method="mcc", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.1, method="mcc", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(cutoff=0.9, method="mk", pos_label="Train") == pytest.approx(
            0.0
        )
        assert model.score(cutoff=0.1, method="mk", pos_label="Train") == pytest.approx(
            0.0
        )
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

    def test_set_params(self, model):
        model.set_params({"nbins": 1000})

        assert model.get_params()["nbins"] == 1000

    def test_model_from_vDF(self, dtc_data_vd):
        current_cursor().execute("DROP MODEL IF EXISTS tc_from_vDF")
        model_test = DecisionTreeClassifier("tc_from_vDF",)
        model_test.fit(
            dtc_data_vd, ["Gender", '"owned cars"', "cost", "income"], "TransPortation"
        )

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'tc_from_vDF'"
        )
        assert current_cursor().fetchone()[0] == "tc_from_vDF"

        model_test.drop()

    def test_to_graphviz(self, model):
        gvz_tree_0 = model.to_graphviz(
            tree_id=0,
            classes_color=["red", "blue", "green"],
            round_pred=4,
            percent=True,
            vertical=False,
            node_style={"shape": "box", "style": "filled"},
            arrow_style={"color": "blue"},
            leaf_style={"shape": "circle", "style": "filled"},
        )
        assert 'digraph Tree{\ngraph [rankdir = "LR"];\n0' in gvz_tree_0
        assert "0 -> 1" in gvz_tree_0

    def test_get_tree(self, model):
        tree_1 = model.get_tree()

        assert tree_1["prediction"] == [
            None,
            "Car",
            None,
            None,
            "Train",
            None,
            "Bus",
            "Bus",
            "Train",
        ]

    def test_plot_tree(self, model):
        result = model.plot_tree()
        assert model.to_graphviz() == result.source.strip()
