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
import sklearn.metrics as skl_metrics
import verticapy.machine_learning.metrics.classification as vpy_metrics
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelBinarizer

python_metrics_map = defaultdict(list)


def calculate_python_metrics(tn, fp, fn, tp, class_idx=0):
    python_metrics_map["skl_tpr"].append(tp / (tp + fn))
    python_metrics_map["skl_sensitivity"].append(
        python_metrics_map["skl_tpr"][class_idx]
    )
    python_metrics_map["skl_recall_score"].append(
        python_metrics_map["skl_tpr"][class_idx]
    )

    python_metrics_map["skl_tnr"].append(tn / (tn + fp))
    python_metrics_map["skl_specificity_score"].append(
        python_metrics_map["skl_tnr"][class_idx]
    )

    python_metrics_map["skl_fnr"].append(fn / (fn + tp))
    python_metrics_map["skl_false_negative_rate"].append(
        python_metrics_map["skl_fnr"][class_idx]
    )

    python_metrics_map["skl_fpr"].append(fp / (fp + tn))
    python_metrics_map["skl_false_positive_rate"].append(
        python_metrics_map["skl_fpr"][class_idx]
    )
    python_metrics_map["skl_precision_score"].append(tp / (tp + fp))
    python_metrics_map["skl_accuracy_score"].append((tp + tn) / (tp + tn + fn + fp))
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
    python_metrics_map["skl_critical_success_index"].append(tp / (tp + fp + fn))
    python_metrics_map["skl_diagnostic_odds_ratio"].append((tp * tn) / (fp * fn))
    python_metrics_map["skl_false_discovery_rate"].append(fp / (fp + tp))
    python_metrics_map["skl_false_omission_rate"].append(fn / (fn + tn))
    python_metrics_map["skl_fowlkes_mallows_index"].append(
        tp / np.sqrt((tp + fp) * (tp + fn))
    )
    python_metrics_map["skl_informedness"].append(
        python_metrics_map["skl_tpr"][class_idx]
        + python_metrics_map["skl_tnr"][class_idx]
        - 1
    )
    python_metrics_map["skl_markedness"].append(
        python_metrics_map["skl_precision_score"][class_idx] + (tn / (tn + fn)) - 1
    )
    python_metrics_map["skl_matthews_corrcoef"].append(
        (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    python_metrics_map["skl_negative_predictive_score"].append(tn / (tn + fn))
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
    support = []

    if average == "binary":
        _tn, _fp, _fn, _tp = skl_metrics.confusion_matrix(y_true, y_pred).ravel()
        _python_metrics_map = calculate_python_metrics(_tn, _fp, _fn, _tp)

        return _python_metrics_map[f"skl_{metric_name}"][0]

    else:
        cnf_matrix = skl_metrics.confusion_matrix(y_true, y_pred)
        num_classes = cnf_matrix.shape[0]

        _tp = np.diag(cnf_matrix)
        _fp = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        _fn = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        _tn = cnf_matrix.sum() - (_fp + _fn + _tp)

        for class_row in cnf_matrix:
            support.append(sum(class_row))

        if average == "micro":
            _tp, _fp, _fn, _tn = sum(_tp), sum(_fp), sum(_fn), sum(_tn)
            _python_metrics_map = calculate_python_metrics(_tn, _fp, _fn, _tp)

            return _python_metrics_map[f"skl_{metric_name}"][0]

        elif average in ["macro", "weighted", "scores", None]:
            support_proportion = sum(support)
            for num_class in range(num_classes):
                _python_metrics_map = calculate_python_metrics(
                    _tn[num_class],
                    _fp[num_class],
                    _fn[num_class],
                    _tp[num_class],
                    class_idx=num_class,
                )
            if average == "macro":
                return sum(_python_metrics_map[f"skl_{metric_name}"]) / num_classes
            elif average == "weighted":
                return sum(
                    _python_metrics_map[f"skl_{metric_name}"]
                    * (support / support_proportion)
                )
            elif average == "scores" or average is None:
                if metric_name == "confusion_matrix":
                    return cnf_matrix
                if metric_name == "accuracy_score" and average is None:
                    return np.trace(cnf_matrix) / np.sum(cnf_matrix)
                elif metric_name == "balanced_accuracy_score" and average is None:
                    return sum(_python_metrics_map["skl_recall_score"]) / num_classes
                else:
                    return _python_metrics_map[f"skl_{metric_name}"]


@pytest.mark.parametrize(
    "compute_method, dataset, expected",
    [
        # ("binary", "pred_cl_dataset_binary", ""),
        # ('micro', 'pred_cl_dataset_multi', ''),
        ('macro', 'pred_cl_dataset_multi', ''),
        # ('weighted', 'pred_cl_dataset_multi', ''),
        # ('scores', 'pred_cl_dataset_multi', ''),
        # (None, 'pred_cl_dataset_multi', ''),
        # pytest.param('invalid', 'pred_cl_dataset_binary', {'pos_label': 'b'}, '', marks=pytest.mark.xfail)
    ],
)
@pytest.mark.parametrize(
    "is_skl_metric, metric_name",
    [
        # ('y', 'confusion_matrix'),
        # ('y', 'accuracy_score'),
        # ('y', 'balanced_accuracy_score'),
        # ('n', 'critical_success_index'),
        # ('n', 'diagnostic_odds_ratio'),
        # ('y', 'f1_score'),
        # ('n', 'false_negative_rate'),
        # ('n', 'false_positive_rate'),
        # ('n', 'false_discovery_rate'),
        # ('n', 'false_omission_rate'),
        # ('n', 'fowlkes_mallows_index'),
        # ('n', 'informedness'),
        # ('n', 'markedness'),
        # ('y', 'matthews_corrcoef'),
        # ('n', 'negative_predictive_score'),
        # ('n', 'negative_likelihood_ratio'),
        # ('n', 'positive_likelihood_ratio'),
        # ('y', 'precision_score'),
        # ('n', 'prevalence_threshold'),
        # ('y', 'recall_score'),
        # ('n', 'specificity_score'),
        # ('n', 'best_cutoff'),  # need to implement for multiclass
        ("y", "roc_auc_score"),  # need to implement for multiclass
        # ('n', 'prc_auc_score'),  # need to implement for multiclass
        # ('y', 'log_loss'),  # failed all - vertica has default base 10, sklean uses natural log (e)
        # ('y', 'classification_report')
    ],
)
class TestClassificationMetrics:
    def test_master_classification_metrics(self, compute_method, dataset, expected, is_skl_metric,
                                           metric_name, request):
        global python_metrics_map
        python_metrics_map = defaultdict(list)
        func_args = {}
        rel_tolerance = 1e-4

        vdf, y_true, y_pred, y_prob, labels, y_true_num, y_pred_num, labels_num = request.getfixturevalue(dataset)

        # set metrics argument
        if compute_method in ["binary"]:
            func_args["pos_label"] = "b"
        else:
            func_args["labels"] = labels

        # verticapy logic
        def get_vertica_metrics():
            if metric_name in ["confusion_matrix"]:
                _vpy_res = getattr(vpy_metrics, metric_name)(
                    "y_true", "y_pred", vdf, **func_args
                )
            elif metric_name in [
                "roc_auc_score",
                "prc_auc_score",
                "best_cutoff",
                "log_loss",
            ]:
                if compute_method == "binary":
                    _vpy_res = getattr(vpy_metrics, metric_name)(
                        "y_true", "y_prob", vdf, average=compute_method, **func_args
                    )
                else:
                    _vpy_res = getattr(vpy_metrics, metric_name)(
                        "y_true_num",
                        ["y_prob0", "y_prob1", "y_prob2"],
                        vdf,
                        average=compute_method,
                        labels=labels_num,
                    )
            elif metric_name in ["classification_report"]:
                vpy_report = getattr(vpy_metrics, metric_name)(
                    "y_true_num", "y_pred_num", vdf, metrics=['precision', 'recall', 'f1_score'], labels=labels_num
                )
                vpy_report_pdf = vpy_report.to_pandas()
                vpy_report_pdf = vpy_report_pdf.drop('avg_micro', axis=1)
                _vpy_res = vpy_report_pdf.to_dict()
            else:
                _vpy_res = getattr(vpy_metrics, metric_name)(
                    "y_true", "y_pred", vdf, average=compute_method, **func_args
                )
            return _vpy_res

        # python logic
        def get_python_metrics():
            nonlocal y_true_num

            if metric_name in ["classification_report"]:
                skl_report = getattr(skl_metrics, metric_name)(
                    y_true_num, y_pred_num, labels=labels_num, output_dict=True
                )
                skl_report_pdf = pd.DataFrame(skl_report)
                skl_report_pdf.rename(
                    columns={'0': 0, '1': 1, '2': 2, 'macro avg': 'avg_macro', 'weighted avg': 'avg_weighted'},
                    index={'f1-score': 'f1_score'}, inplace=True)
                skl_report_pdf = skl_report_pdf.drop('accuracy', axis=1)
                skl_report_pdf = skl_report_pdf.drop('support', axis=0)
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
                        _skl_res = getattr(skl_metrics, metric_name)(
                            y_true, y_prob, labels=labels
                        )
                    else:
                        _y_true_num = LabelBinarizer().fit_transform(y_true)
                        fpr, tpr, thresholds = skl_metrics.roc_curve(
                            _y_true_num.ravel(), y_prob.ravel()
                        )
                        _skl_res = skl_metrics.auc(fpr, tpr)
                else:
                    _skl_res = getattr(skl_metrics, metric_name)(
                        y_true, y_pred, labels=labels
                    )
            else:
                if metric_name in ["prc_auc_score"]:
                    if compute_method == "binary":
                        (
                            precision,
                            recall,
                            thresholds,
                        ) = skl_metrics.precision_recall_curve(
                            y_true, y_prob, pos_label="b"
                        )
                        _skl_res = skl_metrics.auc(recall, precision)
                    else:
                        from sklearn.preprocessing import label_binarize

                        y_true_num = label_binarize(y_true, classes=[0, 1, 2])
                        print()
                        print(y_true_num)
                        fpr, tpr, thresholds = skl_metrics.roc_curve(
                            y_true_num, y_prob, pos_label="b"
                        )
                        _skl_res = skl_metrics.auc(fpr, tpr)
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
                else:
                    _skl_res = python_metrics(
                        y_true, y_pred, average=compute_method, metric_name=metric_name
                    )

            return _skl_res

        vpy_res = get_vertica_metrics()
        skl_res = get_python_metrics()

        print(f"vertica: {vpy_res}, sklearn: {skl_res}")
        assert vpy_res == skl_res if isinstance(vpy_res, dict) else vpy_res == pytest.approx(skl_res, rel=rel_tolerance)
