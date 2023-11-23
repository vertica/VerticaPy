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

# Standard Python Modules
import os

# Other Modules
import matplotlib.pyplot as plt
import xgboost as xgb

# VerticaPy
import verticapy
from verticapy.tests.conftest import get_version
from verticapy.core.vdataframe.base import vDataFrame
from verticapy.utilities import drop
from verticapy._config.config import set_option
from verticapy.connection import current_cursor
from verticapy.datasets import load_titanic, load_dataset_cl
from verticapy.learn.ensemble import XGBClassifier
from verticapy._utils._sql._format import clean_query

# Matplotlib skip
import matplotlib

matplotlib_version = matplotlib.__version__
skip_plt = pytest.mark.skipif(
    matplotlib_version > "3.5.2",
    reason="Test skipped on matplotlib version greater than 3.5.2",
)

set_option("print_info", False)


@pytest.fixture(scope="module")
def xgbc_data_vd():
    xgbc_data = load_dataset_cl(table_name="xgbc_data", schema="public")
    yield xgbc_data
    drop(name="public.xgbc_data", method="table")


@pytest.fixture(scope="module")
def titanic_vd():
    titanic = load_titanic()
    yield titanic
    drop(
        name="public.titanic",
    )


@pytest.fixture(scope="module")
def model(xgbc_data_vd):
    current_cursor().execute("DROP MODEL IF EXISTS xgbc_model_test")

    current_cursor().execute(
        """SELECT xgb_classifier(
                    'xgbc_model_test', 
                    'public.xgbc_data', 
                    'TransPortation', 
                    '*' 
                    USING PARAMETERS 
                    exclude_columns='id, TransPortation', 
                    min_split_loss=0.1, 
                    max_ntree=3, 
                    learning_rate=0.2, 
                    sampling_size=1, 
                    max_depth=6, 
                    nbins=40, 
                    seed=1, 
                    id_column='id')"""
    )

    # I could use load_model but it is buggy
    model_class = XGBClassifier(
        "xgbc_model_test",
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
    model_class._compute_attributes()

    yield model_class
    model_class.drop()


@pytest.mark.skipif(
    get_version()[0] < 10 or (get_version()[0] == 10 and get_version()[1] == 0),
    reason="requires vertica 10.1 or higher",
)
class TestXGBC:
    def test_classification_report(self, model):
        cls_rep1 = model.classification_report().transpose()

        assert cls_rep1["auc"][0] == pytest.approx(1.0)
        assert cls_rep1["prc_auc"][0] == pytest.approx(1.0)
        assert cls_rep1["accuracy"][0] == pytest.approx(1.0)
        assert cls_rep1["log_loss"][0] in (
            pytest.approx(0.127588478759147),
            pytest.approx(0.23458261830345),
        )
        assert cls_rep1["precision"][0] == pytest.approx(1.0)
        assert cls_rep1["recall"][0] == pytest.approx(1.0)
        assert cls_rep1["f1_score"][0] == pytest.approx(1.0)
        assert cls_rep1["mcc"][0] == pytest.approx(1.0)
        assert cls_rep1["informedness"][0] == pytest.approx(1.0)
        assert cls_rep1["markedness"][0] == pytest.approx(1.0)
        assert cls_rep1["csi"][0] == pytest.approx(1.0)

    def test_confusion_matrix(self, model):
        conf_mat1 = model.confusion_matrix()

        assert list(conf_mat1[:, 0]) == [4, 0, 0]
        assert list(conf_mat1[:, 1]) == [0, 3, 0]
        assert list(conf_mat1[:, 2]) == [0, 0, 3]

        conf_mat2 = model.confusion_matrix(cutoff=0.2)

        assert list(conf_mat2[:, 0]) == [4, 0, 0]
        assert list(conf_mat2[:, 1]) == [0, 3, 0]
        assert list(conf_mat2[:, 2]) == [0, 0, 3]

    @skip_plt
    def test_contour(self, titanic_vd):
        model_test = XGBClassifier(
            "model_contour",
        )
        model_test.drop()
        model_test.fit(
            titanic_vd,
            ["age", "fare"],
            "survived",
        )
        result = model_test.contour()
        assert len(result.get_default_bbox_extra_artists()) in (38, 40, 43)
        model_test.drop()

    def test_deploySQL(self, model):
        expected_sql = """PREDICT_XGB_CLASSIFIER("Gender", "owned cars", "cost", "income" 
                              USING PARAMETERS 
                              model_name = 'xgbc_model_test', 
                              match_by_pos = 'true')"""
        result_sql = model.deploySQL()

        assert result_sql == clean_query(expected_sql)

    def test_drop(self):
        current_cursor().execute("DROP MODEL IF EXISTS xgbc_model_test_drop")
        model_test = XGBClassifier(
            "xgbc_model_test_drop",
        )
        model_test.fit(
            "public.xgbc_data",
            ["Gender", '"owned cars"', "cost", "income"],
            "TransPortation",
        )

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'xgbc_model_test_drop'"
        )
        assert current_cursor().fetchone()[0] == "xgbc_model_test_drop"

        model_test.drop()
        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'xgbc_model_test_drop'"
        )
        assert current_cursor().fetchone() is None

    @pytest.mark.skipif(
        get_version()[0] < 12
        or (get_version()[0] == 12 and get_version()[1] == 0 and get_version()[2] < 3),
        reason="requires vertica 12.0.3 or higher",
    )
    def test_features_importance(self, model):
        fimp = model.features_importance(show=False)

        assert fimp["index"] == ["gender"]
        assert fimp["importance"] == [9.61]
        assert fimp["sign"] == [1]
        plt.close("all")

    @pytest.mark.skipif(
        get_version()[0] < 12
        or (get_version()[0] == 12 and get_version()[1] == 0 and get_version()[2] < 3),
        reason="requires vertica 12.0.3 or higher",
    )
    def test_get_score(self, model):
        fim = model.get_score()

        assert fim["predictor_name"] == ["gender", "owned cars", "cost", "income"]
        assert fim["frequency"] == [0.25, 0.25, 0.5, 0.0]
        assert fim["total_gain"] == [
            pytest.approx(-0.0276367140130583),
            pytest.approx(0.0546732664706745),
            pytest.approx(0.972963447542384),
            pytest.approx(0.0),
        ]

    def test_lift_chart(self, model):
        lift_ch = model.lift_chart(pos_label="Bus", nbins=1000, show=False)

        assert lift_ch["decision_boundary"][300] == pytest.approx(0.3)
        assert lift_ch["positive_prediction_ratio"][300] == pytest.approx(0.0)
        assert lift_ch["lift"][300] == pytest.approx(2.5)
        plt.close("all")

    def test_to_python(self, model, titanic_vd):
        model_test = XGBClassifier("rfc_python_test")
        model_test.drop()
        model_test.fit(titanic_vd, ["age", "fare", "sex"], "embarked")
        current_cursor().execute(
            "SELECT PREDICT_XGB_CLASSIFIER(30.0, 45.0, 'male' USING PARAMETERS model_name = 'rfc_python_test', match_by_pos=True)"
        )
        prediction = current_cursor().fetchone()[0]
        assert prediction == model_test.to_python()([[30.0, 45.0, "male"]])[0]
        current_cursor().execute(
            "SELECT PREDICT_XGB_CLASSIFIER(30.0, 145.0, 'female' USING PARAMETERS model_name = 'rfc_python_test', match_by_pos=True)"
        )
        prediction = current_cursor().fetchone()[0]
        assert prediction == model_test.to_python()([[30.0, 145.0, "female"]])[0]

    def test_to_sql(self, model, titanic_vd):
        model_test = XGBClassifier("xgb_sql_test")
        model_test.drop()
        model_test.fit(titanic_vd, ["age", "fare", "sex"], "survived")
        current_cursor().execute(
            "SELECT PREDICT_XGB_CLASSIFIER(* USING PARAMETERS model_name = 'xgb_sql_test', match_by_pos=True)::int, {}::int FROM (SELECT 30.0 AS age, 45.0 AS fare, 'male' AS sex) x".format(
                model_test.to_sql()
            )
        )
        prediction = current_cursor().fetchone()
        assert prediction[0] == pytest.approx(prediction[1])
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
        vdf = vDataFrame("public.xgbc_data")
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
        score = vdf.score(
            "prediction_proba_sql_2", "prediction_proba_vertica_sql_2", metric="r2"
        )
        assert score == pytest.approx(1.0)

    def test_get_vertica_attributes(self, model):
        attr = model.get_vertica_attributes()
        assert attr["attr_name"] == [
            "tree_count",
            "rejected_row_count",
            "accepted_row_count",
            "call_string",
            "details",
            "initial_prediction",
        ]
        assert attr["attr_fields"] == [
            "tree_count",
            "rejected_row_count",
            "accepted_row_count",
            "call_string",
            "predictor, type",
            "response_label, value",
        ]
        assert attr["#_of_rows"] == [1, 1, 1, 1, 4, 3]

        details = model.get_vertica_attributes("details")
        assert details["predictor"] == ["gender", "owned cars", "cost", "income"]
        assert details["type"] == [
            "char or varchar",
            "int",
            "char or varchar",
            "char or varchar",
        ]

        assert (
            model.get_vertica_attributes("accepted_row_count")["accepted_row_count"][0]
            == 10
        )
        assert (
            model.get_vertica_attributes("rejected_row_count")["rejected_row_count"][0]
            == 0
        )
        assert model.get_vertica_attributes("tree_count")["tree_count"][0] == 3
        assert (
            "xgb_classifier('public.xgbc_model_test', 'public.xgbc_data', '\"transportation\"', '*' USING PARAMETERS"
            in model.get_vertica_attributes("call_string")["call_string"][0]
        )

    def test_get_params(self, model):
        assert model.get_params() == {
            "max_ntree": 3,
            "min_split_loss": 0.1,
            "learning_rate": 0.2,
            "sample": 1.0,
            "max_depth": 6,
            "nbins": 40,
            "split_proposal_method": "global",
            "tol": 0.001,
            "weight_reg": 0.0,
        } or model.get_params() == {
            "max_ntree": 3,
            "min_split_loss": 0.1,
            "learning_rate": 0.2,
            "sample": 1.0,
            "max_depth": 6,
            "nbins": 40,
            "split_proposal_method": "global",
            "tol": 0.001,
            "weight_reg": 0.0,
            "col_sample_by_tree": 1.0,
            "col_sample_by_node": 1.0,
        }

    def test_prc_curve(self, model):
        prc = model.prc_curve(pos_label="Car", nbins=1000, show=False)

        assert prc["threshold"][300] == pytest.approx(0.299)
        assert prc["recall"][300] == pytest.approx(1.0)
        assert prc["precision"][300] in (pytest.approx(1.0), pytest.approx(0.6))
        plt.close("all")

    def test_predict(self, xgbc_data_vd, model):
        xgbc_data_copy = xgbc_data_vd.copy()

        model.predict(xgbc_data_copy, name="pred")
        assert xgbc_data_copy["pred"].mode() == "Bus"

        model.predict(xgbc_data_copy, name="pred1", cutoff=0.7)
        assert xgbc_data_copy["pred1"].mode() == "Bus"

        model.predict(xgbc_data_copy, name="pred2", cutoff=0.3)
        assert xgbc_data_copy["pred2"].mode() == "Bus"

    def test_predict_proba(self, xgbc_data_vd, model):
        xgbc_data_copy = xgbc_data_vd.copy()

        model.predict_proba(xgbc_data_copy, name="prob")
        assert xgbc_data_copy["prob_bus"].avg() == 0.3440198
        assert xgbc_data_copy["prob_train"].avg() == 0.3199195
        assert xgbc_data_copy["prob_car"].avg() == 0.3360605

        model.predict_proba(xgbc_data_copy, name="prob_bus_2", pos_label="Bus")
        assert xgbc_data_copy["prob_bus_2"].avg() == 0.3440198

    def test_roc_curve(self, model):
        roc = model.roc_curve(pos_label="Train", nbins=1000, show=False)

        assert roc["threshold"][100] == pytest.approx(0.1)
        assert roc["false_positive"][100] == pytest.approx(1.0)
        assert roc["true_positive"][100] == pytest.approx(1.0)
        assert roc["threshold"][700] == pytest.approx(0.7)
        assert roc["false_positive"][700] == pytest.approx(0.0)
        assert roc["true_positive"][700] == pytest.approx(0.0)
        plt.close("all")

    def test_cutoff_curve(self, model):
        cutoff_curve = model.cutoff_curve(pos_label="Train", nbins=1000, show=False)

        assert cutoff_curve["threshold"][100] == pytest.approx(0.1)
        assert cutoff_curve["false_positive"][100] == pytest.approx(1.0)
        assert cutoff_curve["true_positive"][100] == pytest.approx(1.0)
        assert cutoff_curve["threshold"][700] == pytest.approx(0.7)
        assert cutoff_curve["false_positive"][700] == pytest.approx(0.0)
        assert cutoff_curve["true_positive"][700] == pytest.approx(0.0)
        plt.close("all")

    def test_score(self, model):
        assert model.score(cutoff=0.9, metric="accuracy") == pytest.approx(1.0)
        assert model.score(cutoff=0.1, metric="accuracy") == pytest.approx(1.0)
        assert model.score(
            cutoff=0.9, metric="auc", pos_label="Train"
        ) == pytest.approx(1.0)
        assert model.score(
            cutoff=0.1, metric="auc", pos_label="Train"
        ) == pytest.approx(1.0)
        assert model.score(cutoff=0.9, metric="best_cutoff", pos_label="Train") in (
            pytest.approx(0.6338, 1e-2),
            pytest.approx(0.3863, 1e-2),
        )
        assert model.score(cutoff=0.1, metric="best_cutoff", pos_label="Train") in (
            pytest.approx(0.6338, 1e-2),
            pytest.approx(0.3863, 1e-2),
        )
        assert model.score(
            cutoff=0.633, metric="bm", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(cutoff=0.1, metric="bm", pos_label="Train") == pytest.approx(
            0.0
        )
        assert model.score(
            cutoff=0.9, metric="csi", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.1, metric="csi", pos_label="Train"
        ) == pytest.approx(0.3)
        assert model.score(cutoff=0.9, metric="f1", pos_label="Train") == pytest.approx(
            0.0
        )
        assert model.score(cutoff=0.1, metric="f1", pos_label="Train") == pytest.approx(
            0.4615384615384615
        )
        assert model.score(cutoff=0.9, metric="logloss", pos_label="Train") in (
            pytest.approx(0.111961142833969),
            pytest.approx(0.21696238336042),
        )
        assert model.score(cutoff=0.1, metric="logloss", pos_label="Train") in (
            pytest.approx(0.111961142833969),
            pytest.approx(0.21696238336042),
        )
        assert model.score(
            cutoff=0.9, metric="mcc", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.1, metric="mcc", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(cutoff=0.9, metric="mk", pos_label="Train") == pytest.approx(
            -0.30000000000000004
        )
        assert model.score(cutoff=0.1, metric="mk", pos_label="Train") == pytest.approx(
            -0.7
        )
        assert model.score(
            cutoff=0.9, metric="npv", pos_label="Train"
        ) == pytest.approx(0.7)
        assert model.score(
            cutoff=0.1, metric="npv", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.9, metric="prc_auc", pos_label="Train"
        ) == pytest.approx(1.0)
        assert model.score(
            cutoff=0.1, metric="prc_auc", pos_label="Train"
        ) == pytest.approx(1.0)
        assert model.score(
            cutoff=0.9, metric="precision", pos_label="Train"
        ) == pytest.approx(0.0)
        assert model.score(
            cutoff=0.1, metric="precision", pos_label="Train"
        ) == pytest.approx(0.3)
        assert model.score(
            cutoff=0.9, metric="specificity", pos_label="Train"
        ) == pytest.approx(1.0)
        assert model.score(
            cutoff=0.1, metric="specificity", pos_label="Train"
        ) == pytest.approx(0.0)

    def test_set_params(self, model):
        model.set_params({"nbins": 1000})

        assert model.get_params()["nbins"] == 1000

    def test_model_from_vDF(self, xgbc_data_vd):
        current_cursor().execute("DROP MODEL IF EXISTS xgbc_from_vDF")
        model_test = XGBClassifier(
            "xgbc_from_vDF",
        )
        model_test.fit(
            xgbc_data_vd,
            ["Gender", '"owned cars"', "cost", "income"],
            "TransPortation",
        )

        current_cursor().execute(
            "SELECT model_name FROM models WHERE model_name = 'xgbc_from_vDF'"
        )
        assert current_cursor().fetchone()[0] == "xgbc_from_vDF"

        model_test.drop()

    def test_to_graphviz(self, model):
        gvz_tree_1 = model.to_graphviz(
            tree_id=1,
            classes_color=["red", "blue", "green"],
            round_pred=4,
            percent=True,
            vertical=False,
            node_style={"shape": "box", "style": "filled"},
            arrow_style={"color": "blue"},
            leaf_style={"shape": "circle", "style": "filled"},
        )
        assert 'digraph Tree{\ngraph [rankdir = "LR"];\n0' in gvz_tree_1
        assert "0 -> 1" in gvz_tree_1

    def test_get_tree(self, model):
        tree_1 = model.get_tree(tree_id=1)

        assert tree_1["prediction"] == [None, "", None, None, "", None, "", "", ""]

    def test_plot_tree(self, model):
        result = model.plot_tree()
        assert model.to_graphviz() == result.source.strip()

    def test_to_json_binary(self, titanic_vd):
        import xgboost as xgb

        titanic = titanic_vd.copy()
        titanic.fillna()
        path = "verticapy_test_xgbr.json"
        X = ["pclass", "age", "fare"]
        y = "survived"
        model = XGBClassifier(
            "verticapy_xgb_binaryclassifier_test", max_ntree=10, max_depth=5
        )
        model.drop()
        model.fit(titanic, X, y)
        X_test = titanic[X].to_numpy()
        y_test_vertica = model.to_python(return_proba=True)(X_test)
        if os.path.exists(path):
            os.remove(path)
        model.to_json(path)
        model_python = xgb.XGBClassifier()
        model_python.load_model(path)
        y_test_python = model_python.predict_proba(X_test)
        result = (y_test_vertica - y_test_python) ** 2
        result = result.sum() / len(result)
        assert result == pytest.approx(0.0, abs=1.0e-14)
        y_test_vertica = model.to_python()(X_test)
        y_test_python = model_python.predict(X_test)
        result = (y_test_vertica - y_test_python) ** 2
        result = result.sum() / len(result)
        assert result == 0.0
        model.drop()
        os.remove(path)

    def test_to_json_multiclass(self, titanic_vd):
        titanic = titanic_vd.copy()
        titanic.fillna()
        path = "verticapy_test_xgbr.json"
        X = ["survived", "age", "fare"]
        y = "pclass"
        model = XGBClassifier(
            "verticapy_xgb_multiclass_classifier_test", max_ntree=10, max_depth=5
        )
        model.drop()
        model.fit(titanic, X, y)
        X_test = titanic[X].to_numpy()
        y_test_vertica = model.to_python(return_proba=True)(X_test).argsort()
        if os.path.exists(path):
            os.remove(path)
        model.to_json(path)
        model_python = xgb.XGBClassifier()
        model_python.load_model(path)
        y_test_python = model_python.predict_proba(X_test).argsort()
        result = (y_test_vertica - y_test_python) ** 2
        result = result.sum() / len(result)
        assert result == 0.0
        y_test_vertica = model.to_python()(X_test)
        y_test_python = model_python.predict(X_test)
        # in xgboost 2.0.0 the label classes are 0-indexed, but they were 1-indexed in its 1.7.6 version
        if xgb.__version__ >= "2.0.0":
            y_test_python = y_test_python + 1
        result = (y_test_vertica - y_test_python) ** 2
        result = result.sum() / len(result)
        assert result == 0.0
        model.drop()
        os.remove(path)

    def test_optional_name(self):
        model = XGBClassifier()
        assert model.model_name is not None
