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
from verticapy.machine_learning.metrics.classification import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    critical_success_index,
    f1_score,
    informedness,
    log_loss,
    markedness,
    matthews_corrcoef,
    multilabel_confusion_matrix,
    negative_predictive_score,
    prc_auc,
    roc_auc,
    precision_score,
    recall_score,
    specificity_score,
)
from verticapy.machine_learning.metrics.regression import (
    aic_bic,
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
from verticapy.machine_learning.model_selection.model_validation import (
    prc_curve,
    roc_curve,
    lift_chart,
)


FUNCTIONS_CLASSIFICATION_DICTIONNARY = {
    "aic": aic_score,
    "bic": bic_score,
    "accuracy": accuracy_score,
    "acc": accuracy_score,
    "auc": roc_auc,
    "prc_auc": prc_auc,
    "best_cutoff": roc_curve,
    "best_threshold": roc_curve,
    "recall": recall_score,
    "tpr": recall_score,
    "precision": precision_score,
    "ppv": precision_score,
    "specificity": specificity_score,
    "tnr": specificity_score,
    "negative_predictive_value": negative_predictive_score,
    "npv": negative_predictive_score,
    "log_loss": log_loss,
    "logloss": log_loss,
    "f1": f1_score,
    "mcc": matthews_corrcoef,
    "bm": informedness,
    "informedness": informedness,
    "mk": markedness,
    "markedness": markedness,
    "csi": critical_success_index,
    "critical_success_index": critical_success_index,
    "roc_curve": roc_curve,
    "roc": roc_curve,
    "prc_curve": prc_curve,
    "prc": prc_curve,
    "lift_chart": lift_chart,
    "lift": lift_chart,
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
