"""
(c)  Copyright  [2018-2023]  OpenText  or one of its
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
import pytest
from verticapy.tests_new.machine_learning.vertica.test_base_model_methods import (
    rel_tolerance_map,
    classification_metrics_args,
    model_score,
    regression_metrics_args,
    model_params,
    anova_report_args,
    details_report_args,
)
from verticapy.tests_new.machine_learning.vertica import REL_TOLERANCE, ABS_TOLERANCE
from verticapy.tests_new.machine_learning.vertica.linear_model.test_linear_model import (
    TestLinearModel,
)
import sklearn.metrics as skl_metrics


@pytest.mark.parametrize(
    "model_class",
    [
        "RandomForestRegressor",
        "RandomForestClassifier",
        "DecisionTreeRegressor",
        "DecisionTreeClassifier",
        "DummyTreeRegressor",
        "DummyTreeClassifier",
    ],
)
class TestBaseTreeModel:
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
        vpy_model_obj, py_model_obj = get_vpy_model(model_class), get_py_model(
            model_class
        )

        vpy_res = getattr(vpy_model_obj.model, fit_attr)()
        py_res = getattr(py_model_obj.model, fit_attr)(py_model_obj.X, py_model_obj.y)

        assert vpy_res == pytest.approx(py_res, rel=rel_tolerance_map[model_class])

    def test_get_tree(self, get_vpy_model, get_py_model, model_class):
        model = get_vpy_model(model_class).model
        if model_class in ["RandomForestRegressor", "RandomForestClassifier"]:
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

    def test_plot_tree(self, get_vpy_model, get_py_model, model_class):
        model = get_vpy_model(model_class).model
        trees = model.plot_tree()
        graph_trees = model.to_graphviz()

        assert graph_trees == trees.source.strip()

    def test_to_graphviz(self, get_vpy_model, model_class):
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
    ["RandomForestRegressor", "DecisionTreeRegressor", "DummyTreeRegressor"],
)
class TestRegressionTreeModel:
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
        score = get_vpy_model(model_class).model.get_score()

        assert score[key_name] == pytest.approx(expected, rel=1e-02)

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
    ):
        TestLinearModel().test_regression_report_none(
            get_vpy_model,
            get_py_model,
            model_class,
            regression_metrics,
            fun_name,
            vpy_metric_name,
            py_metric_name,
            _rel_tolerance,
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
        TestLinearModel().test_regression_report_details(
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
        TestLinearModel().test_regression_report_anova(
            model_class,
            get_vpy_model,
            get_py_model,
            regression_metrics,
            fun_name,
            metric,
            metric_types,
            _rel_tolerance,
            _abs_tolerance,
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
    ["RandomForestClassifier", "DecisionTreeClassifier", "DummyTreeClassifier"],
)
class TestClassificationTreeModel:
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
        score = get_vpy_model(model_class).model.get_score()

        assert score[key_name] == expected

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
            ("informedness", None, rel_tolerance_map, ABS_TOLERANCE),
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

    def test_confusion_matrix(
        self, model_class, get_vpy_model, get_py_model, classification_metrics
    ):
        vpy_res = get_vpy_model(model_class).model.confusion_matrix()

        py_model_obj = get_py_model(model_class)
        py_res = skl_metrics.confusion_matrix(py_model_obj.y, py_model_obj.pred)

        assert vpy_res == pytest.approx(py_res, rel=1e-01)

    def test_cutoff_curve(
        self, model_class, get_vpy_model, get_py_model, classification_metrics
    ):
        cutoff_curve = get_vpy_model(model_class).model.cutoff_curve(show=False)
        _fpr, _tpr = cutoff_curve["false_positive"], cutoff_curve["true_positive"]
        vpy_res = skl_metrics.auc(_fpr, _tpr)

        py_model_obj = get_py_model(model_class)
        y, score = py_model_obj.y.ravel(), py_model_obj.pred_prob[:, 1].ravel()
        py_fpr, py_tpr, _ = skl_metrics.roc_curve(y_true=y, y_score=score)
        py_res = skl_metrics.auc(py_fpr, py_tpr)

        assert vpy_res == pytest.approx(py_res, rel=1e-01)

    def test_lift_chart(
        self, model_class, get_vpy_model, get_py_model, classification_metrics
    ):
        lift_chart = get_vpy_model(model_class).model.lift_chart(show=False)

        assert lift_chart["decision_boundary"][300] == pytest.approx(0.299, rel=1e-02)
        assert lift_chart["positive_prediction_ratio"][300] == pytest.approx(
            0.30946, rel=1e-01
        )
        assert lift_chart["lift"][300] == pytest.approx(2.4658, rel=1e-01)

    def test_prc_curve(
        self, model_class, get_vpy_model, get_py_model, classification_metrics
    ):
        vpy_prc_curve = get_vpy_model(model_class).model.prc_curve(show=False)
        vpy_recall, vpy_precision = vpy_prc_curve["recall"], vpy_prc_curve["precision"]
        vpy_res = skl_metrics.auc(vpy_recall, vpy_precision)

        py_model_obj = get_py_model(model_class)
        precision, recall, thresholds = skl_metrics.precision_recall_curve(
            y_true=py_model_obj.y.ravel(),
            probas_pred=py_model_obj.pred_prob[:, 1].ravel(),
        )
        py_res = skl_metrics.auc(recall, precision)

        assert vpy_res == pytest.approx(py_res, rel=1e-01)

    def test_predict_proba(self, model_class, get_vpy_model, get_py_model):
        vpy_res = (
            get_vpy_model(model_class)
            .pred_prob_vdf[["survived_pred_1"]]
            .to_numpy()
            .ravel()
        )
        py_res = get_py_model(model_class).pred_prob[:, 1].ravel()

        assert vpy_res[:10] == pytest.approx(py_res[:10], rel=1e-0)

    def test_roc_curve(
        self, model_class, get_vpy_model, get_py_model, classification_metrics
    ):
        vpy_roc_curve = get_vpy_model(model_class).model.roc_curve(show=False)
        vpy_fpr, vpy_tpr = (
            vpy_roc_curve["false_positive"],
            vpy_roc_curve["true_positive"],
        )
        vpy_res = skl_metrics.auc(vpy_fpr, vpy_tpr)

        py_model_obj = get_py_model(model_class)
        py_fpr, py_tpr, py_thresholds = skl_metrics.roc_curve(
            y_true=py_model_obj.y.ravel(), y_score=py_model_obj.pred_prob[:, 1].ravel()
        )
        py_res = skl_metrics.auc(py_fpr, py_tpr)

        assert vpy_res == pytest.approx(py_res, rel=1e-01)
