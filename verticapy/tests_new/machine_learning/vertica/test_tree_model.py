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
import os
import pytest
from verticapy.tests_new.machine_learning.vertica.test_base_model_methods import (
    rel_tolerance_map,
    classification_metrics_args,
    model_params,
    model_score,
    regression_metrics_args,
    anova_report_args,
    details_report_args,
    regression_report_none,
    regression_report_details,
    regression_report_anova,
)
from verticapy.tests_new.machine_learning.vertica import ABS_TOLERANCE
import sklearn.metrics as skl_metrics


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

        assert vpy_res == pytest.approx(py_res, rel=rel_tolerance_map[model_class])

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
            arrow_style={"color": "blue"},
            leaf_style={"shape": "circle", "style": "filled"},
        )
        assert (
            'digraph Tree{\ngraph [rankdir = "LR"];\n0' in gvz_tree_0
            and "0 -> 1" in gvz_tree_0
        )


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
        _rel_tolerance,
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
            _rel_tolerance,
            model_params,
        )

        assert vpy_score == pytest.approx(
            py_score,
            rel=_rel_tolerance[model_class]
            if isinstance(_rel_tolerance, dict)
            else _rel_tolerance,
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
        _rel_tolerance,
        _abs_tolerance,
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
            _rel_tolerance,
            _abs_tolerance,
        )

        if py_res == 0:
            assert vpy_reg_rep_details_map[metric] == pytest.approx(
                py_res, abs=_abs_tolerance
            )
        else:
            assert vpy_reg_rep_details_map[metric] == pytest.approx(
                py_res,
                rel=_rel_tolerance[model_class]
                if isinstance(_rel_tolerance, dict)
                else _rel_tolerance,
            )

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
        _rel_tolerance,
        _abs_tolerance,
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
            _rel_tolerance,
            _abs_tolerance,
        )
        for vpy_res, metric_type in zip(reg_rep_anova[metric], metric_types):
            py_res = regression_metrics_map[metric_type]

            if py_res == 0:
                assert vpy_res == pytest.approx(py_res, abs=1e-9)
            else:
                assert vpy_res == pytest.approx(
                    py_res,
                    rel=_rel_tolerance[model_class]
                    if isinstance(_rel_tolerance, dict)
                    else _rel_tolerance,
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
        _rel_tolerance,
        model_params,
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
            _rel_tolerance,
            model_params,
        )

        print(
            f"Metric Name: {py_metric_name}, vertica: {vpy_score}, sklearn: {py_score}"
        )

        assert vpy_score == pytest.approx(py_score, rel=_rel_tolerance[model_class])


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

    @pytest.mark.parametrize(*classification_metrics_args)
    def test_score(
        self,
        model_class,
        get_vpy_model,
        get_py_model,
        classification_metrics,
        vpy_metric_name,
        py_metric_name,
        _rel_tolerance,
        model_params,
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
            _rel_tolerance,
            model_params,
        )

        print(
            f"Metric Name: {py_metric_name}, vertica: {vpy_score}, sklearn: {py_score}"
        )

        assert vpy_score == pytest.approx(py_score, rel=_rel_tolerance[model_class])

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
        "metric, expected, _rel_tolerance, _abs_tolerance",
        [
            ("auc", None, rel_tolerance_map, ABS_TOLERANCE),
            ("prc_auc", None, rel_tolerance_map, ABS_TOLERANCE),
            ("accuracy", None, rel_tolerance_map, ABS_TOLERANCE),
            ("log_loss", None, rel_tolerance_map, ABS_TOLERANCE),
            ("precision", None, rel_tolerance_map, ABS_TOLERANCE),
            ("recall", None, rel_tolerance_map, ABS_TOLERANCE),
            ("f1_score", None, rel_tolerance_map, ABS_TOLERANCE),
            ("mcc", None, rel_tolerance_map, ABS_TOLERANCE),
            # ("informedness", None, rel_tolerance_map, ABS_TOLERANCE), # getting mismatch for xgb
            ("markedness", None, rel_tolerance_map, ABS_TOLERANCE),
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
        _rel_tolerance,
        _abs_tolerance,
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
            py_report_map[metric],
            rel=_rel_tolerance[model_class]
            if isinstance(_rel_tolerance, dict)
            else _rel_tolerance,
        )

    def test_confusion_matrix(self, model_class, get_vpy_model, get_py_model):
        """
        test function - test_confusion_matrix
        """
        vpy_res = get_vpy_model(model_class).model.confusion_matrix()

        py_model_obj = get_py_model(model_class)
        py_res = skl_metrics.confusion_matrix(py_model_obj.y, py_model_obj.pred)

        assert vpy_res == pytest.approx(py_res, rel=rel_tolerance_map[model_class])

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

        assert vpy_res == pytest.approx(py_res, rel=rel_tolerance_map[model_class])

    def test_lift_chart(self, model_class, get_vpy_model):
        """
        test function - test_lift_chart
        """
        lift_chart = get_vpy_model(model_class).model.lift_chart(show=False)

        assert lift_chart["decision_boundary"][300] == pytest.approx(
            0.299, rel=rel_tolerance_map[model_class]
        )
        assert lift_chart["positive_prediction_ratio"][300] == pytest.approx(
            0.30946, rel=rel_tolerance_map[model_class]
        )
        assert lift_chart["lift"][300] == pytest.approx(
            2.4658, rel=rel_tolerance_map[model_class]
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

        assert vpy_res == pytest.approx(py_res, rel=rel_tolerance_map[model_class])

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

        assert vpy_res[:5] == pytest.approx(
            py_res[:5], rel=rel_tolerance_map[model_class]
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

        assert vpy_res == pytest.approx(py_res, rel=rel_tolerance_map[model_class])


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
            y = "pclass"
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

        assert result == pytest.approx(0.0, abs=rel_tolerance_map[model_class])
