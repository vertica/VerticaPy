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
from verticapy.machine_learning.metrics.classification import (
    accuracy_score,
    balanced_accuracy_score,
    best_cutoff,
    classification_report,
    confusion_matrix,
    critical_success_index,
    diagnostic_odds_ratio,
    f1_score,
    false_discovery_rate,
    false_negative_rate,
    false_omission_rate,
    false_positive_rate,
    fowlkes_mallows_index,
    informedness,
    log_loss,
    markedness,
    matthews_corrcoef,
    negative_predictive_score,
    negative_likelihood_ratio,
    positive_likelihood_ratio,
    prc_auc_score,
    precision_score,
    prevalence_threshold,
    recall_score,
    roc_auc_score,
    specificity_score,
)
from verticapy.machine_learning.metrics.regression import (
    aic_score,
    bic_score,
    anova_table,
    explained_variance,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    quantile_error,
    r2_score,
    regression_report,
)
from verticapy.machine_learning.metrics.plotting import (
    lift_chart,
    prc_curve,
    roc_curve,
)


FUNCTIONS_CLASSIFICATION_DICTIONNARY = {
    "aic": aic_score,
    "bic": bic_score,
    "accuracy": accuracy_score,
    "acc": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "ba": balanced_accuracy_score,
    "auc": roc_auc_score,
    "roc_auc": roc_auc_score,
    "prc_auc": prc_auc_score,
    "best_cutoff": best_cutoff,
    "best_threshold": best_cutoff,
    "false_discovery_rate": false_discovery_rate,
    "fdr": false_discovery_rate,
    "false_omission_rate": false_omission_rate,
    "for": false_omission_rate,
    "false_negative_rate": false_negative_rate,
    "fnr": false_negative_rate,
    "false_positive_rate": false_positive_rate,
    "fpr": false_positive_rate,
    "recall": recall_score,
    "tpr": recall_score,
    "precision": precision_score,
    "ppv": precision_score,
    "specificity": specificity_score,
    "tnr": specificity_score,
    "negative_predictive_value": negative_predictive_score,
    "npv": negative_predictive_score,
    "negative_likelihood_ratio": negative_likelihood_ratio,
    "lr-": negative_likelihood_ratio,
    "positive_likelihood_ratio": positive_likelihood_ratio,
    "lr+": positive_likelihood_ratio,
    "diagnostic_odds_ratio": diagnostic_odds_ratio,
    "dor": diagnostic_odds_ratio,
    "log_loss": log_loss,
    "logloss": log_loss,
    "f1": f1_score,
    "f1_score": f1_score,
    "mcc": matthews_corrcoef,
    "bm": informedness,
    "informedness": informedness,
    "mk": markedness,
    "markedness": markedness,
    "ts": critical_success_index,
    "csi": critical_success_index,
    "critical_success_index": critical_success_index,
    "fowlkes_mallows_index": fowlkes_mallows_index,
    "fm": fowlkes_mallows_index,
    "prevalence_threshold": prevalence_threshold,
    "pm": prevalence_threshold,
    "confusion_matrix": confusion_matrix,
    "classification_report": classification_report,
}

FUNCTIONS_REGRESSION_DICTIONNARY = {
    "aic": aic_score,
    "bic": bic_score,
    "r2": r2_score,
    "rsquared": r2_score,
    "mae": mean_absolute_error,
    "mean_absolute_error": mean_absolute_error,
    "mse": mean_squared_error,
    "mean_squared_error": mean_squared_error,
    "msle": mean_squared_log_error,
    "mean_squared_log_error": mean_squared_log_error,
    "max": max_error,
    "max_error": max_error,
    "median": median_absolute_error,
    "median_absolute_error": median_absolute_error,
    "var": explained_variance,
    "explained_variance": explained_variance,
}

FUNCTIONS_DICTIONNARY = {
    **FUNCTIONS_CLASSIFICATION_DICTIONNARY,
    **FUNCTIONS_REGRESSION_DICTIONNARY,
}
