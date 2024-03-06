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
from decimal import Decimal
import os
import pandas as pd
import pytest

import sklearn.metrics as skl_metrics

from verticapy.tests_new.machine_learning.vertica.test_base_model_methods import (
    rel_abs_tol_map,
    REL_TOLERANCE,
    classification_metrics_args,
    model_params,
    model_score,
    regression_metrics_args,
    anova_report_args,
    details_report_args,
    regression_report_none,
    regression_report_details,
    regression_report_anova,
    calculate_tolerance,
)


@pytest.mark.parametrize(
    "model_class",
    [
        "RandomForestRegressor",
        "RandomForestClassifier",
        "DecisionTreeRegressor",
        "DecisionTreeClassifier",
        "XGBRegressor",
        "XGBClassifier",
        # "DummyTreeRegressor",
        # "DummyTreeClassifier",
    ],
)
class TestBaseTreeModel:
    """
    test class for base tree models
    """

    @pytest.mark.parametrize("fit_attr", ["score"])
    def test_fit(
        self,
        get_vpy_model,
        get_py_model,
        model_class,
        fit_attr,
    ):
        """
        test function - fit
        """
        vpy_model_obj, py_model_obj = (
            get_vpy_model(model_class),
            get_py_model(model_class),
        )

        vpy_res = getattr(vpy_model_obj.model, fit_attr)()
        py_res = getattr(py_model_obj.model, fit_attr)(py_model_obj.X, py_model_obj.y)

        _rel_tol, _abs_tol = calculate_tolerance(vpy_res, py_res)
        print(
            f"Model_class: {model_class}, Metric_name: {fit_attr}, rel_tol(e): {'%.e' % Decimal(_rel_tol)}, abs_tol(e): {'%.e' % Decimal(_abs_tol)}"
        )

        assert vpy_res == pytest.approx(
            py_res, rel=rel_abs_tol_map[model_class][fit_attr]["rel"]
        )

    def test_get_tree(self, get_vpy_model, model_class):
        """
        test function - test_get_tree
        """
        model = get_vpy_model(model_class).model
        if model_class in [
            "RandomForestRegressor",
            "RandomForestClassifier",
            "XGBRegressor",
            "XGBClassifier",
        ]:
            vpy_trees = model.get_tree(tree_id=5)
            assert vpy_trees["tree_id"][-1] == 5
        elif model_class in [
            "DecisionTreeRegressor",
            "DecisionTreeClassifier",
            "DummyTreeRegressor",
            "DummyTreeClassifier",
        ]:
            vpy_trees = model.get_tree(tree_id=0)
            assert vpy_trees["tree_id"][-1] == 0

    def test_plot_tree(self, get_vpy_model, model_class):
        """
        test function - test_plot_tree
        """
        model = get_vpy_model(model_class).model
        trees = model.plot_tree()
        graph_trees = model.to_graphviz()

        assert graph_trees == trees.source.strip()

    def test_to_graphviz(self, get_vpy_model, model_class):
        """
        test function - test_to_graphviz
        """
        gvz_tree_0 = get_vpy_model(model_class).model.to_graphviz(
            tree_id=0,
            classes_color=["red", "blue", "green"],
            round_pred=4,
            percent=True,
            vertical=False,
            node_style={"shape": "box", "style": "filled"},
            edge_style={"color": "blue"},
            leaf_style={"shape": "circle", "style": "filled"},
        )
        assert "digraph Tree {\ngraph" in gvz_tree_0 and "0 -> 1" in gvz_tree_0


@pytest.mark.parametrize(
    "model_class",
    [
        "RandomForestRegressor",
        "DecisionTreeRegressor",
        "XGBRegressor",
        # "DummyTreeRegressor"
    ],
)
class TestRegressionTreeModel:
    """
    test class - test class for tree regression model
    """

    abs_error_report_reg_tree = {}
    model_class_set_reg_tree = set()

    @pytest.mark.skip(reason="Getting different value at each run. Need to check")
    @pytest.mark.parametrize(
        "key_name, expected",
        [
            ("predictor_index", [0, 2, 1]),
            ("predictor_name", ["citric_acid", "alcohol", "residual_sugar"]),
            (
                "importance_value",
                [0.298645513828444, 0.36169688305476, 0.339657603116797],
            ),
        ],
    )
    def test_get_score(self, model_class, get_vpy_model, key_name, expected):
        """
        test function - test_get_score
        """
        score = get_vpy_model(model_class).model.get_score()
        if key_name == "importance_value":
            if model_class == "RandomForestRegressor":
                expected = [0.135338765036947, 0.817269880453645, 0.0473913545094082]
            elif model_class == "DecisionTreeRegressor":
                expected = [0.113007597173622, 0.834435990806013, 0.0525564120203654]
            elif model_class == "XGBRegressor":
                key_name = "total_gain"
                # expected = frequency = [0.353333324193954, 0.433333337306976, 0.213333338499069]
                expected = [
                    0.240065249135922,
                    0.566047435405588,
                    0.19388731545849,
                ]
                # expected = avg_gain = [0.149246239140166, 0.745136129976554, 0.10561763088328]

        assert score[key_name] == pytest.approx(expected, rel=1e-0)

    @pytest.mark.parametrize(*regression_metrics_args)
    @pytest.mark.parametrize("fun_name", ["regression", "report"])
    def test_regression_report_none(
        self,
        get_vpy_model,
        get_py_model,
        model_class,
        regression_metrics,
        fun_name,
        vpy_metric_name,
        py_metric_name,
        model_params,
    ):
        """
        test function - test_regression_report_none
        """
        vpy_score, py_score = regression_report_none(
            get_vpy_model,
            get_py_model,
            model_class,
            regression_metrics,
            fun_name,
            vpy_metric_name,
            py_metric_name,
            model_params,
        )

        assert vpy_score == pytest.approx(
            py_score, rel=rel_abs_tol_map[model_class][vpy_metric_name[0]]["rel"]
        )

    @pytest.mark.parametrize(*details_report_args)
    @pytest.mark.parametrize("fun_name", ["regression", "report"])
    def test_regression_report_details(
        self,
        model_class,
        get_vpy_model,
        get_py_model,
        regression_metrics,
        fun_name,
        metric,
        expected,
    ):
        """
        test function - test_regression_report_details
        """
        vpy_reg_rep_details_map, py_res = regression_report_details(
            model_class,
            get_vpy_model,
            get_py_model,
            regression_metrics,
            fun_name,
            metric,
            expected,
        )

        if metric in ["Dep. Variable", "Model", "No. Observations", "No. Predictors"]:
            tol = REL_TOLERANCE
        else:
            tol = rel_abs_tol_map[model_class][metric]["rel"]
            _rel_tol, _abs_tol = calculate_tolerance(
                vpy_reg_rep_details_map[metric], py_res
            )
            print(
                f"Model_class: {model_class}, Metric_name: {metric}, rel_tol(e): {'%.e' % Decimal(_rel_tol)}, abs_tol(e): {'%.e' % Decimal(_abs_tol)}"
            )

        if py_res == 0:
            assert vpy_reg_rep_details_map[metric] == pytest.approx(
                py_res, abs=rel_abs_tol_map[model_class][metric]["abs"]
            )
        else:
            assert vpy_reg_rep_details_map[metric] == pytest.approx(py_res, rel=tol)

    @pytest.mark.parametrize(*anova_report_args)
    @pytest.mark.parametrize("fun_name", ["regression", "report"])
    def test_regression_report_anova(
        self,
        model_class,
        get_vpy_model,
        get_py_model,
        regression_metrics,
        fun_name,
        metric,
        metric_types,
    ):
        """
        test function - test_regression_report_anova
        """
        reg_rep_anova, regression_metrics_map = regression_report_anova(
            model_class,
            get_vpy_model,
            get_py_model,
            regression_metrics,
            fun_name,
        )
        for vpy_res, metric_type in zip(reg_rep_anova[metric], metric_types):
            py_res = regression_metrics_map[metric_type]

            if metric_type != "":
                _rel_tol, _abs_tol = calculate_tolerance(vpy_res, py_res)
                print(
                    f"Model_class: {model_class}, Metric_name: {metric}, Metric_type: {metric_type}, rel_tol(e): {'%.e' % Decimal(_rel_tol)}, abs_tol(e): {'%.e' % Decimal(_abs_tol)}"
                )

                if py_res == 0:
                    assert vpy_res == pytest.approx(py_res, abs=1e-9)
                else:
                    assert vpy_res == pytest.approx(
                        py_res, rel=rel_abs_tol_map[model_class][metric]["rel"]
                    )

    @pytest.mark.parametrize(*regression_metrics_args)
    def test_score(
        self,
        model_class,
        get_vpy_model,
        get_py_model,
        regression_metrics,
        vpy_metric_name,
        py_metric_name,
        model_params,
        request,
    ):
        """
        test function - test_score
        """
        vpy_score, py_score = model_score(
            model_class,
            get_vpy_model,
            get_py_model,
            regression_metrics,
            vpy_metric_name,
            py_metric_name,
            model_params,
        )

        _rel_tol, _abs_tol = calculate_tolerance(vpy_score, py_score)

        self.abs_error_report_reg_tree[(model_class, py_metric_name)] = {
            "Model_class": model_class,
            "Metric_name": py_metric_name.title()
            if "_" in py_metric_name
            else py_metric_name.upper(),
            "Vertica": vpy_score,
            "Sklearn": py_score,
            "rel_tol": _rel_tol,
            "abs_tol": _abs_tol,
            "rel_tol(e)": "%.e" % Decimal(_rel_tol),
            "abs_tol(e)": "%.e" % Decimal(_abs_tol),
            "abs_pct_diff": ((vpy_score - py_score) / (py_score if py_score else 1e-15))
            * 100,
        }
        print(self.abs_error_report_reg_tree[(model_class, py_metric_name)])

        self.model_class_set_reg_tree.add(model_class)
        tc_count = (len(request.node.keywords["pytestmark"][0].args[1])) * len(
            self.model_class_set_reg_tree
        )

        if len(self.abs_error_report_reg_tree.keys()) == tc_count:
            abs_error_report_reg_tree_pdf = (
                pd.DataFrame(self.abs_error_report_reg_tree.values())
                .sort_values(by=["Model_class", "Metric_name"])
                .reset_index(drop=True)
            )
            abs_error_report_reg_tree_pdf.to_csv(
                "abs_error_report_reg_tree.csv", index=False
            )

        assert vpy_score == pytest.approx(
            py_score, rel=rel_abs_tol_map[model_class][vpy_metric_name[0]]["rel"]
        )


@pytest.mark.parametrize(
    "model_class",
    [
        "RandomForestClassifier",
        "DecisionTreeClassifier",
        "XGBClassifier",
        # "DummyTreeClassifier"
    ],
)
class TestClassificationTreeModel:
    """
    test class - test class for classification model
    """

    abs_error_report_cls_tree = {}
    model_class_set_cls_tree = set()

    @pytest.mark.parametrize(*classification_metrics_args)
    def test_score(
        self,
        model_class,
        get_vpy_model,
        get_py_model,
        classification_metrics,
        vpy_metric_name,
        py_metric_name,
        model_params,
        request,
    ):
        """
        test function - test_score
        """
        vpy_score, py_score = model_score(
            model_class,
            get_vpy_model,
            get_py_model,
            classification_metrics,
            vpy_metric_name,
            py_metric_name,
            model_params,
        )

        self.model_class_set_cls_tree.add(model_class)
        tc_count = (len(request.node.keywords["pytestmark"][0].args[1])) * len(
            self.model_class_set_cls_tree
        )

        _rel_tol, _abs_tol = calculate_tolerance(vpy_score, py_score)

        self.abs_error_report_cls_tree[(model_class, py_metric_name)] = {
            "Model_class": model_class,
            "Metric_name": py_metric_name.title()
            if "_" in py_metric_name
            else py_metric_name.upper(),
            "Vertica": vpy_score,
            "Sklearn": py_score,
            "rel_tol": _rel_tol,
            "abs_tol": _abs_tol,
            "rel_tol(e)": "%.e" % Decimal(_rel_tol),
            "abs_tol(e)": "%.e" % Decimal(_abs_tol),
            "abs_pct_diff": ((vpy_score - py_score) / (py_score if py_score else 1e-15))
            * 100,
        }
        print(self.abs_error_report_cls_tree[(model_class, py_metric_name)])

        if len(self.abs_error_report_cls_tree.keys()) == tc_count:
            abs_error_report_cls_tree_pdf = (
                pd.DataFrame(self.abs_error_report_cls_tree.values())
                .sort_values(by=["Model_class", "Metric_name"])
                .reset_index(drop=True)
            )
            abs_error_report_cls_tree_pdf.to_csv(
                "abs_error_report_cls_tree.csv", index=False
            )

        assert vpy_score == pytest.approx(
            py_score, rel=rel_abs_tol_map[model_class][vpy_metric_name[0]]["rel"]
        )

    @pytest.mark.skip(reason="Getting different value at each run. Need to check")
    @pytest.mark.parametrize(
        "key_name, expected",
        [
            ("predictor_index", [0, 1, 2]),
            ("predictor_name", ["age", "fare", "sex"]),
            (
                "importance_value",
                [0.14889501520446, 0.252062721468843, 0.599042263326698],
            ),
        ],
    )
    def test_get_score(self, model_class, get_vpy_model, key_name, expected):
        """
        test function - test_get_score
        """
        score = get_vpy_model(model_class).model.get_score()

        if key_name == "importance_value":
            if model_class == "RandomForestClassifier":
                expected = [0.133497674381064, 0.15895498045659, 0.707547345162347]
            elif model_class == "DecisionTreeClassifier":
                expected = [0.106193606304587, 0.159078836047269, 0.734727557648144]
            elif model_class == "XGBClassifier":
                key_name = "total_gain"
                # expected = frequency = [0.353333324193954, 0.433333337306976, 0.213333338499069]
                expected = [
                    0.270225563897779,
                    0.24395515721239,
                    0.485819278889831,
                ]
                # expected = avg_gain = [0.149246239140166, 0.745136129976554, 0.10561763088328]

        assert score[key_name] == pytest.approx(expected, rel=1e-0)

    @pytest.mark.parametrize(
        "metric, expected",
        [
            ("auc", None),
            ("prc_auc", None),
            ("accuracy", None),
            ("log_loss", None),
            ("precision", None),
            ("recall", None),
            ("f1_score", None),
            ("mcc", None),
            # ("informedness", None), # getting mismatch for xgb
            ("markedness", None),
        ],
    )
    @pytest.mark.parametrize("fun_name", ["classification_report", "report"])
    def test_classification_report(
        self,
        model_class,
        get_vpy_model,
        classification_metrics,
        metric,
        expected,
        fun_name,
    ):
        """
        test function - test_classification_report
        """
        vpy_model_obj = get_vpy_model(model_class)
        report = getattr(vpy_model_obj.model, fun_name)()
        vpy_report_map = dict(zip(report["index"], report["value"]))

        py_report_map = classification_metrics(model_class)

        assert vpy_report_map[metric] == pytest.approx(
            py_report_map[metric], rel=rel_abs_tol_map[model_class][metric]["rel"]
        )

    def test_confusion_matrix(self, model_class, get_vpy_model, get_py_model):
        """
        test function - test_confusion_matrix
        """
        vpy_res = get_vpy_model(model_class).model.confusion_matrix()

        py_model_obj = get_py_model(model_class)
        py_res = skl_metrics.confusion_matrix(py_model_obj.y, py_model_obj.pred)

        print(f"vertica: {vpy_res}, sklearn: {py_res}")
        # _rel_tol, _abs_tol = calculate_tolerance(vpy_res, py_res)

        assert vpy_res == pytest.approx(
            py_res, rel=rel_abs_tol_map[model_class]["confusion_matrix"]["rel"]
        )

    def test_cutoff_curve(self, model_class, get_vpy_model, get_py_model):
        """
        test function - test_cutoff_curve
        """
        cutoff_curve = get_vpy_model(model_class).model.cutoff_curve(show=False)
        _fpr, _tpr = cutoff_curve["false_positive"], cutoff_curve["true_positive"]
        vpy_res = skl_metrics.auc(_fpr, _tpr)

        py_model_obj = get_py_model(model_class)
        y, score = py_model_obj.y.ravel(), py_model_obj.pred_prob[:, 1].ravel()
        py_fpr, py_tpr, _ = skl_metrics.roc_curve(y_true=y, y_score=score)
        py_res = skl_metrics.auc(py_fpr, py_tpr)

        _rel_tol, _abs_tol = calculate_tolerance(vpy_res, py_res)

        assert vpy_res == pytest.approx(
            py_res, rel=rel_abs_tol_map[model_class]["cutoff_curve"]["rel"]
        )

    @pytest.mark.parametrize(
        "lift_chart_type, expected",
        [
            ("decision_boundary", 0.299),
            ("positive_prediction_ratio", 0.30946),
            ("lift", 2.4658),
        ],
    )
    def test_lift_chart(self, model_class, get_vpy_model, lift_chart_type, expected):
        """
        test function - test_lift_chart
        """
        lift_chart = get_vpy_model(model_class).model.lift_chart(show=False)
        actual = lift_chart[lift_chart_type][300]
        _rel_tol, _abs_tol = calculate_tolerance(actual, expected)

        assert actual == pytest.approx(
            expected, rel=rel_abs_tol_map[model_class]["lift_chart"]["rel"]
        )

    def test_prc_curve(self, model_class, get_vpy_model, get_py_model):
        """
        test function - test_prc_curve
        """
        vpy_prc_curve = get_vpy_model(model_class).model.prc_curve(show=False)
        vpy_recall, vpy_precision = vpy_prc_curve["recall"], vpy_prc_curve["precision"]
        vpy_res = skl_metrics.auc(vpy_recall, vpy_precision)

        py_model_obj = get_py_model(model_class)
        precision, recall, _ = skl_metrics.precision_recall_curve(
            y_true=py_model_obj.y.ravel(),
            probas_pred=py_model_obj.pred_prob[:, 1].ravel(),
        )
        py_res = skl_metrics.auc(recall, precision)

        _rel_tol, _abs_tol = calculate_tolerance(vpy_res, py_res)

        assert vpy_res == pytest.approx(
            py_res, rel=rel_abs_tol_map[model_class]["prc_curve"]["rel"]
        )

    def test_predict_proba(self, model_class, get_vpy_model, get_py_model):
        """
        test function - test_predict_proba
        """
        vpy_res = (
            get_vpy_model(model_class)
            .pred_prob_vdf[["survived_pred_1"]]
            .to_numpy()
            .ravel()
        )
        py_res = get_py_model(model_class).pred_prob[:, 1].ravel()

        _rel_tol, _abs_tol = calculate_tolerance(vpy_res, py_res)

        assert vpy_res[:5] == pytest.approx(
            py_res[:5], rel=rel_abs_tol_map[model_class]["predict_proba"]["rel"]
        )

    def test_roc_curve(self, model_class, get_vpy_model, get_py_model):
        """
        test function - test_roc_curve
        """
        vpy_roc_curve = get_vpy_model(model_class).model.roc_curve(show=False)
        vpy_fpr, vpy_tpr = (
            vpy_roc_curve["false_positive"],
            vpy_roc_curve["true_positive"],
        )
        vpy_res = skl_metrics.auc(vpy_fpr, vpy_tpr)

        py_model_obj = get_py_model(model_class)
        py_fpr, py_tpr, _ = skl_metrics.roc_curve(
            y_true=py_model_obj.y.ravel(), y_score=py_model_obj.pred_prob[:, 1].ravel()
        )
        py_res = skl_metrics.auc(py_fpr, py_tpr)

        _rel_tol, _abs_tol = calculate_tolerance(vpy_res, py_res)

        assert vpy_res == pytest.approx(
            py_res, rel=rel_abs_tol_map[model_class]["roc_curve"]["rel"]
        )


@pytest.mark.parametrize(
    "model_class",
    [
        "XGBRegressor",
        "XGBClassifier",
    ],
)
class TestXGBModel:
    """
    test class - test class for xgb model
    """

    def test_to_json(self, model_class, get_vpy_model, get_py_model):
        """
        test function - test_to_json
        """
        file_path = f"vpy_exported_{model_class}_model.json"

        if model_class == "XGBClassifier":
            X = ["survived", "age", "fare"]
            y = ["pclass"]
            return_proba = True
            pred_method = "predict_proba"
            vpy_model_obj = get_vpy_model(model_class, X=X, y=y)
        else:
            return_proba = False
            pred_method = "predict"
            vpy_model_obj = get_vpy_model(model_class)

        vpy_model_obj.model.to_json(path=file_path)

        py_model_obj = get_py_model(model_class)
        py_model_obj.model.load_model(file_path)
        y_test_vertica = vpy_model_obj.model.to_python(return_proba=return_proba)(
            py_model_obj.X
        )
        y_test_python = getattr(py_model_obj.model, pred_method)(py_model_obj.X)
        result = (y_test_vertica - y_test_python) ** 2
        result = result.sum() / len(result)
        os.remove(file_path)

        assert result == pytest.approx(
            0.0, abs=rel_abs_tol_map[model_class]["to_json"]["rel"]
        )
