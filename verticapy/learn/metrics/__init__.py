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
import warnings

warning_message = (
    "Importing from 'verticapy.learn.cluster' is deprecated, "
    "and it will no longer be possible in the next minor release. "
    "Please use 'verticapy.machine_learning.metrics' instead "
    "to ensure compatibility with upcoming versions."
)
warnings.warn(warning_message, Warning)

from verticapy.machine_learning.metrics.classification import (
    accuracy_score,
    best_cutoff,
    classification_report,
    confusion_matrix,
    critical_success_index,
    f1_score,
    informedness,
    log_loss,
    markedness,
    matthews_corrcoef,
    negative_predictive_score,
    prc_auc_score,
    roc_auc_score,
    precision_score,
    recall_score,
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


FUNCTIONS_CLASSIFICATION_DICTIONNARY = {
    "aic": aic_score,
    "bic": bic_score,
    "accuracy": accuracy_score,
    "acc": accuracy_score,
    "auc": roc_auc_score,
    "roc_auc": roc_auc_score,
    "prc_auc": prc_auc_score,
    "best_cutoff": best_cutoff,
    "best_threshold": best_cutoff,
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
    "f1_score": f1_score,
    "mcc": matthews_corrcoef,
    "bm": informedness,
    "informedness": informedness,
    "mk": markedness,
    "markedness": markedness,
    "csi": critical_success_index,
    "critical_success_index": critical_success_index,
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
