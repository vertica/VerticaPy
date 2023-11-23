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
from collections import defaultdict
import verticapy.machine_learning.metrics.classification as vpy_metrics
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import sklearn.metrics as skl_metrics
from sklearn.preprocessing import label_binarize
import pytest

python_metrics_map = defaultdict(list)


def calculate_python_metrics(
    true_negative, false_positive, false_negative, true_positive, class_idx=0
):
    """
    Computes the classification metrics.
    """
    python_metrics_map["skl_tpr"].append(
        true_positive / (true_positive + false_negative)
    )
    python_metrics_map["skl_sensitivity"].append(
        python_metrics_map["skl_tpr"][class_idx]
    )
    python_metrics_map["skl_recall_score"].append(
        python_metrics_map["skl_tpr"][class_idx]
    )

    python_metrics_map["skl_tnr"].append(
        true_negative / (true_negative + false_positive)
    )
    python_metrics_map["skl_specificity_score"].append(
        python_metrics_map["skl_tnr"][class_idx]
    )

    python_metrics_map["skl_fnr"].append(
        false_negative / (false_negative + true_positive)
    )
    python_metrics_map["skl_false_negative_rate"].append(
        python_metrics_map["skl_fnr"][class_idx]
    )

    python_metrics_map["skl_fpr"].append(
        false_positive / (false_positive + true_negative)
    )
    python_metrics_map["skl_false_positive_rate"].append(
        python_metrics_map["skl_fpr"][class_idx]
    )
    python_metrics_map["skl_precision_score"].append(
        true_positive / (true_positive + false_positive)
    )
    python_metrics_map["skl_accuracy_score"].append(
        (true_positive + true_negative)
        / (true_positive + true_negative + false_negative + false_positive)
    )
    python_metrics_map["skl_balanced_accuracy_score"].append(
        (
            python_metrics_map["skl_recall_score"][class_idx]
            + python_metrics_map["skl_specificity_score"][class_idx]
        )
        / 2
    )
    python_metrics_map["skl_f1_score"].append(
        2
        * (
            python_metrics_map["skl_precision_score"][class_idx]
            * python_metrics_map["skl_recall_score"][class_idx]
        )
        / (
            python_metrics_map["skl_precision_score"][class_idx]
            + python_metrics_map["skl_recall_score"][class_idx]
        )
    )
    python_metrics_map["skl_critical_success_index"].append(
        true_positive / (true_positive + false_positive + false_negative)
    )
    python_metrics_map["skl_diagnostic_odds_ratio"].append(
        (true_positive * true_negative) / (false_positive * false_negative)
    )
    python_metrics_map["skl_false_discovery_rate"].append(
        false_positive / (false_positive + true_positive)
    )
    python_metrics_map["skl_false_omission_rate"].append(
        false_negative / (false_negative + true_negative)
    )
    python_metrics_map["skl_fowlkes_mallows_index"].append(
        true_positive
        / np.sqrt((true_positive + false_positive) * (true_positive + false_negative))
    )
    python_metrics_map["skl_informedness"].append(
        python_metrics_map["skl_tpr"][class_idx]
        + python_metrics_map["skl_tnr"][class_idx]
        - 1
    )
    python_metrics_map["skl_markedness"].append(
        python_metrics_map["skl_precision_score"][class_idx]
        + (true_negative / (true_negative + false_negative))
        - 1
    )
    python_metrics_map["skl_matthews_corrcoef"].append(
        (true_positive * true_negative - false_positive * false_negative)
        / np.sqrt(
            (true_positive + false_positive)
            * (true_positive + false_negative)
            * (true_negative + false_positive)
            * (true_negative + false_negative)
        )
    )
    python_metrics_map["skl_negative_predictive_score"].append(
        true_negative / (true_negative + false_negative)
    )
    python_metrics_map["skl_negative_likelihood_ratio"].append(
        python_metrics_map["skl_fnr"][class_idx]
        / python_metrics_map["skl_tnr"][class_idx]
    )
    python_metrics_map["skl_positive_likelihood_ratio"].append(
        python_metrics_map["skl_tpr"][class_idx]
        / python_metrics_map["skl_fpr"][class_idx]
    )
    python_metrics_map["skl_prevalence_threshold"].append(
        np.sqrt(1 - python_metrics_map["skl_specificity_score"][class_idx])
        / (
            np.sqrt(python_metrics_map["skl_sensitivity"][class_idx])
            + np.sqrt(1 - python_metrics_map["skl_specificity_score"][class_idx])
        )
    )

    return python_metrics_map


def python_metrics(y_true, y_pred, average="binary", metric_name=""):
    """
    master function to call calculate_python_metrics, and it
    calculates confusion metrics for multi class
    """
    support = []
    metrics_map = {}
    metrics_val = 0

    if average == "binary":
        (
            _true_negative,
            _false_positive,
            _false_negative,
            _true_positive,
        ) = skl_metrics.confusion_matrix(y_true, y_pred).ravel()
        metrics_map = calculate_python_metrics(
            _true_negative, _false_positive, _false_negative, _true_positive
        )
        metrics_val = metrics_map[f"skl_{metric_name}"][0]

    else:
        cnf_matrix = skl_metrics.confusion_matrix(y_true, y_pred)
        num_classes = cnf_matrix.shape[0]

        _true_positive = np.diag(cnf_matrix)
        _false_positive = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        _false_negative = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        _true_negative = cnf_matrix.sum() - (
            _false_positive + _false_negative + _true_positive
        )

        for class_row in cnf_matrix:
            support.append(sum(class_row))

        if average == "micro":
            _true_positive, _false_positive, _false_negative, _true_negative = (
                sum(_true_positive),
                sum(_false_positive),
                sum(_false_negative),
                sum(_true_negative),
            )
            metrics_map = calculate_python_metrics(
                _true_negative, _false_positive, _false_negative, _true_positive
            )
            metrics_val = metrics_map[f"skl_{metric_name}"][0]
        elif average in ["macro", "weighted", "scores", None]:
            support_proportion = sum(support)
            for num_class in range(num_classes):
                metrics_map = calculate_python_metrics(
                    _true_negative[num_class],
                    _false_positive[num_class],
                    _false_negative[num_class],
                    _true_positive[num_class],
                    class_idx=num_class,
                )
            if average == "macro":
                metrics_val = sum(metrics_map[f"skl_{metric_name}"]) / num_classes
            elif average == "weighted":
                metrics_val = sum(
                    metrics_map[f"skl_{metric_name}"] * (support / support_proportion)
                )

            elif average == "scores" or average is None:
                if metric_name == "confusion_matrix":
                    metrics_val = cnf_matrix
                elif metric_name == "accuracy_score" and average is None:
                    metrics_val = np.trace(cnf_matrix) / np.sum(cnf_matrix)
                elif metric_name == "balanced_accuracy_score" and average is None:
                    metrics_val = sum(metrics_map["skl_recall_score"]) / num_classes
                else:
                    metrics_val = metrics_map[f"skl_{metric_name}"]

    return metrics_val


@pytest.mark.parametrize(
    "compute_method, dataset",
    [
        ("binary", "pred_cl_dataset_binary"),
        ("micro", "pred_cl_dataset_multi"),
        ("macro", "pred_cl_dataset_multi"),
        ("weighted", "pred_cl_dataset_multi"),
        ("scores", "pred_cl_dataset_multi"),
        (None, "pred_cl_dataset_multi"),
    ],
)
@pytest.mark.parametrize(
    "is_skl_metric, metric_name",
    [
        ("y", "confusion_matrix"),
        ("y", "accuracy_score"),
        ("y", "balanced_accuracy_score"),
        ("n", "critical_success_index"),
        ("n", "diagnostic_odds_ratio"),
        ("y", "f1_score"),
        ("n", "false_negative_rate"),
        ("n", "false_positive_rate"),
        ("n", "false_discovery_rate"),
        ("n", "false_omission_rate"),
        ("n", "fowlkes_mallows_index"),
        ("n", "informedness"),
        ("n", "markedness"),
        ("y", "matthews_corrcoef"),
        ("n", "negative_predictive_score"),
        ("n", "negative_likelihood_ratio"),
        ("n", "positive_likelihood_ratio"),
        ("y", "precision_score"),
        ("y", "average_precision_score"),
        ("n", "prevalence_threshold"),
        ("y", "recall_score"),
        ("n", "specificity_score"),
        ("n", "best_cutoff"),  # need to implement for multiclass
        ("y", "roc_auc_score"),
        ("n", "prc_auc_score"),
        (
            "y",
            "log_loss",
        ),  # failed all - vertica uses log base 10, sklean uses natural log(e)
        ("y", "classification_report"),
    ],
)
class TestClassificationMetrics:
    """
    class for unit test - classification metrics
    """

    def test_master_classification_metrics(
        self, compute_method, dataset, is_skl_metric, metric_name, request
    ):
        """
        test function test_master_classification_metrics
        """
        global python_metrics_map
        python_metrics_map = defaultdict(list)
        func_args = {}
        rel_tolerance = 1e-2

        (
            vdf,
            y_true,
            y_pred,
            y_prob,
            labels,
            y_true_num,
            y_pred_num,
            labels_num,
        ) = request.getfixturevalue(dataset)

        # set metrics argument
        if compute_method in ["binary"]:
            func_args["pos_label"] = "b"
        else:
            func_args["labels"] = labels

        # skipping a test
        if metric_name in ["best_cutoff"] and compute_method != "binary":
            pytest.skip("Need to fix function for multi-class")
        elif metric_name in ["roc_auc_score", "prc_auc_score"] and compute_method in [
            "weighted",
            "scores",
            None,
        ]:
            pytest.skip("Not yet Implemented")
        elif metric_name in ["prc_auc_score"] and compute_method in [
            "macro",
        ]:
            pytest.skip("vertica binning issue needs to be fixed.")
        elif metric_name in ["log_loss"]:
            pytest.skip("vertica has default base 10, sklean uses natural log (e)")

        # verticapy logic
        def get_vertica_metrics():
            """
            getter function for vertica metrics
            """
            if metric_name in ["confusion_matrix"]:
                _vpy_res = getattr(vpy_metrics, metric_name)(
                    "y_true", "y_pred", vdf, **func_args
                )
            elif metric_name in [
                "roc_auc_score",
                "prc_auc_score",
                "best_cutoff",
                "log_loss",
                "average_precision_score",
            ]:
                if compute_method == "binary":
                    if metric_name in [
                        "roc_auc_score",
                        "prc_auc_score",
                        "average_precision_score",
                    ]:
                        _vpy_res = getattr(vpy_metrics, metric_name)(
                            "y_true_num",
                            "y_pred_num",
                            vdf,
                            average=compute_method,
                            pos_label="1",
                        )
                    else:
                        _vpy_res = getattr(vpy_metrics, metric_name)(
                            "y_true", "y_prob", vdf, average=compute_method, **func_args
                        )
                else:
                    _vpy_res = getattr(vpy_metrics, metric_name)(
                        "y_true_num",
                        ["y_prob0", "y_prob1", "y_prob2"],
                        vdf,
                        average=compute_method,
                        labels=[str(label_num) for label_num in labels_num]
                        if metric_name
                        in ["roc_auc_score", "prc_auc_score", "average_precision_score"]
                        else labels_num,
                    )
                # rounding as best_cutoff metrics value precisions are upto 2/3 decimals
                _vpy_res = (
                    round(_vpy_res, 2) if metric_name == "best_cutoff" else _vpy_res
                )
            elif metric_name in ["classification_report"]:
                vpy_report = getattr(vpy_metrics, metric_name)(
                    "y_true_num",
                    "y_pred_num",
                    vdf,
                    metrics=["precision", "recall", "f1_score"],
                    labels=labels_num,
                )
                vpy_report_pdf = vpy_report.to_pandas()
                vpy_report_pdf = vpy_report_pdf.drop("avg_micro", axis=1)
                _vpy_res = vpy_report_pdf.to_dict()
            else:
                _vpy_res = getattr(vpy_metrics, metric_name)(
                    "y_true", "y_pred", vdf, average=compute_method, **func_args
                )
            return _vpy_res

        # python logic
        def get_python_metrics():
            """
            getter function for python metrics
            """
            nonlocal y_true_num

            if metric_name in ["classification_report"]:
                skl_report = getattr(skl_metrics, metric_name)(
                    y_true_num, y_pred_num, labels=labels_num, output_dict=True
                )
                skl_report_pdf = pd.DataFrame(skl_report)
                skl_report_pdf.rename(
                    columns={
                        "0": 0,
                        "1": 1,
                        "2": 2,
                        "macro avg": "avg_macro",
                        "weighted avg": "avg_weighted",
                    },
                    index={"f1-score": "f1_score"},
                    inplace=True,
                )
                skl_report_pdf = skl_report_pdf.drop("accuracy", axis=1)
                skl_report_pdf = skl_report_pdf.drop("support", axis=0)
                _skl_res = skl_report_pdf.to_dict()
            elif is_skl_metric == "y" and compute_method not in ["scores", None]:
                if metric_name in [
                    "accuracy_score",
                    "matthews_corrcoef",
                    "balanced_accuracy_score",
                ]:
                    if compute_method == "binary":
                        _skl_res = getattr(skl_metrics, metric_name)(y_true, y_pred)
                    else:
                        _skl_res = python_metrics(
                            y_true,
                            y_pred,
                            average=compute_method,
                            metric_name=metric_name,
                        )
                elif metric_name in ["precision_score", "recall_score"]:
                    _skl_res = getattr(skl_metrics, metric_name)(
                        y_true, y_pred, average=compute_method, **func_args
                    )
                elif metric_name in ["f1_score"]:
                    func_args["labels"] = (
                        "b" if "labels" in func_args else func_args["pos_label"]
                    )
                    _skl_res = getattr(skl_metrics, metric_name)(
                        y_true, y_pred, average=compute_method, pos_label="b"
                    )
                elif metric_name in ["roc_auc_score", "log_loss"]:
                    if compute_method == "binary":
                        if metric_name == "roc_auc_score":
                            _skl_res = getattr(skl_metrics, metric_name)(
                                y_true_num, y_pred_num
                            )
                        else:
                            _skl_res = getattr(skl_metrics, metric_name)(
                                y_true, y_prob, labels=labels
                            )
                    else:
                        _skl_res = getattr(skl_metrics, metric_name)(
                            y_true_num,
                            y_prob,
                            average=compute_method,
                            multi_class="ovr",
                            labels=labels_num,
                        )
                elif metric_name in ["average_precision_score"]:
                    if compute_method == "binary":
                        _skl_res = getattr(skl_metrics, metric_name)(
                            y_true_num, y_pred_num, pos_label=1
                        )
                    else:
                        _skl_res = getattr(skl_metrics, metric_name)(
                            y_true_num,
                            y_prob,
                            average=compute_method,
                            pos_label=1,
                        )
                else:
                    _skl_res = getattr(skl_metrics, metric_name)(
                        y_true, y_pred, labels=labels
                    )
            else:
                if metric_name in ["prc_auc_score"]:
                    if compute_method == "binary":
                        precision, recall, _ = skl_metrics.precision_recall_curve(
                            y_true_num, y_pred_num
                        )
                        _skl_res = skl_metrics.auc(recall, precision)
                    elif compute_method == "micro":
                        y_true_num_ = LabelBinarizer().fit_transform(y_true_num)
                        (
                            precision,
                            recall,
                            thresholds,
                        ) = skl_metrics.precision_recall_curve(
                            y_true_num_.ravel(), y_prob.ravel()
                        )
                        _skl_res = skl_metrics.auc(recall, precision)
                    elif compute_method == "macro":
                        y_true_num_ = LabelBinarizer().fit_transform(y_true_num)
                        n_classes = len(y_prob[0])
                        precision, recall, prc_auc = dict(), dict(), dict()
                        for i in range(n_classes):
                            (
                                precision[i],
                                recall[i],
                                _,
                            ) = skl_metrics.precision_recall_curve(
                                y_true_num_[:, i], y_prob[:, i]
                            )

                        recall_grid = np.linspace(0.0, 1.0, 10000)

                        # Interpolate all ROC curves at these points
                        mean_precision = np.zeros_like(recall_grid)

                        for i in range(n_classes):
                            sorted_pairs = sorted(zip(recall[i], precision[i]))
                            recall_sorted, precision_sorted = zip(*sorted_pairs)
                            mean_precision += np.interp(
                                recall_grid, recall_sorted, precision_sorted
                            )  # linear interpolation

                        # Average it and compute AUC
                        mean_precision /= n_classes
                        _skl_res = skl_metrics.auc(recall_grid, mean_precision)

                elif metric_name in ["best_cutoff"]:
                    if compute_method == "binary":
                        fpr, tpr, thresholds = skl_metrics.roc_curve(
                            y_true, y_prob, pos_label="b"
                        )
                        optimal_idx = np.argmax(abs(tpr - fpr))
                        _skl_res = thresholds[optimal_idx]
                    else:
                        precision, recall, thresholds = skl_metrics.roc_curve(
                            y_true, y_prob, pos_label="b"
                        )
                        _skl_res = skl_metrics.auc(recall, precision)

                    # rounding as best_cutoff metrics value precisions are upto 2/3 decimals
                    _skl_res = round(_skl_res, 2)
                elif metric_name in ["average_precision_score"]:
                    _skl_res = getattr(skl_metrics, metric_name)(
                        y_true_num,
                        y_prob,
                        average=None,
                        pos_label=1,
                    )
                else:
                    _skl_res = python_metrics(
                        y_true, y_pred, average=compute_method, metric_name=metric_name
                    )

            return _skl_res

        vpy_res = get_vertica_metrics()
        skl_res = get_python_metrics()

        print(f"{metric_name} - vertica: {vpy_res}, sklearn: {skl_res}")
        assert (
            vpy_res == skl_res
            if isinstance(vpy_res, dict)
            else vpy_res == pytest.approx(skl_res, rel=rel_tolerance)
        ), f"vertica: {vpy_res}, sklearn: {skl_res}"
