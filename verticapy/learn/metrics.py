# (c) Copyright [2018-2020] Micro Focus or one of its affiliates.
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
#
# |_     |~) _  _| _  /~\    _ |.
# |_)\/  |_)(_|(_||   \_/|_|(_|||
#    /
#              ____________       ______
#             / __        `\     /     /
#            |  \/         /    /     /
#            |______      /    /     /
#                   |____/    /     /
#          _____________     /     /
#          \           /    /     /
#           \         /    /     /
#            \_______/    /     /
#             ______     /     /
#             \    /    /     /
#              \  /    /     /
#               \/    /     /
#                    /     /
#                   /     /
#                   \    /
#                    \  /
#                     \/
#                    _
# \  / _  __|_. _ _ |_)
#  \/ (/_|  | |(_(_|| \/
#                     /
# VerticaPy is a Python library with scikit-like functionality to use to conduct
# data science projects on data stored in Vertica, taking advantage Vertica’s
# speed and built-in analytics and machine learning features. It supports the
# entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize
# data transformation operations, and offers beautiful graphical options.
#
# VerticaPy aims to solve all of these problems. The idea is simple: instead
# of moving data around for processing, VerticaPy brings the logic to the data.
#
#
# Modules
#
# Standard Python Modules
import math

# VerticaPy Modules
from verticapy import *
from verticapy.learn.plot import *
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.connections.connect import read_auto_connect

#
# Regression
#
# ---#
def explained_variance(y_true: str, y_score: str, input_relation: str, cursor=None):
    """
---------------------------------------------------------------------------
Computes the Explained Variance.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.

Returns
-------
float
	score
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [str], False),
            ("input_relation", input_relation, [str], False),
        ]
    )
    if not (cursor):
        conn = read_auto_connect()
        cursor = conn.cursor()
    else:
        conn = False
        check_cursor(cursor)
    query = "SELECT 1 - VARIANCE({} - {}) / VARIANCE({}) FROM {}".format(
        y_true, y_score, y_true, input_relation
    )
    cursor.execute(query)
    result = cursor.fetchone()[0]
    if conn:
        conn.close()
    return result


# ---#
def max_error(y_true: str, y_score: str, input_relation: str, cursor=None):
    """
---------------------------------------------------------------------------
Computes the Max Error.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.

Returns
-------
float
	score
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [str], False),
            ("input_relation", input_relation, [str], False),
        ]
    )
    if not (cursor):
        conn = read_auto_connect()
        cursor = conn.cursor()
    else:
        conn = False
        check_cursor(cursor)
    query = "SELECT MAX(ABS({} - {})) FROM {}".format(y_true, y_score, input_relation)
    cursor.execute(query)
    result = cursor.fetchone()[0]
    if conn:
        conn.close()
    return result


# ---#
def mean_absolute_error(y_true: str, y_score: str, input_relation: str, cursor=None):
    """
---------------------------------------------------------------------------
Computes the Mean Absolute Error.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.

Returns
-------
float
	score
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [str], False),
            ("input_relation", input_relation, [str], False),
        ]
    )
    if not (cursor):
        conn = read_auto_connect()
        cursor = conn.cursor()
    else:
        conn = False
        check_cursor(cursor)
    query = "SELECT AVG(ABS({} - {})) FROM {}".format(y_true, y_score, input_relation)
    cursor.execute(query)
    result = cursor.fetchone()[0]
    if conn:
        conn.close()
    return result


# ---#
def mean_squared_error(y_true: str, y_score: str, input_relation: str, cursor=None):
    """
---------------------------------------------------------------------------
Computes the Mean Squared Error.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.

Returns
-------
float
	score
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [str], False),
            ("input_relation", input_relation, [str], False),
        ]
    )
    if not (cursor):
        conn = read_auto_connect()
        cursor = conn.cursor()
    else:
        conn = False
        check_cursor(cursor)
    query = "SELECT MSE({}, {}) OVER () FROM {}".format(y_true, y_score, input_relation)
    cursor.execute(query)
    result = cursor.fetchone()[0]
    if conn:
        conn.close()
    return result


# ---#
def mean_squared_log_error(y_true: str, y_score: str, input_relation: str, cursor=None):
    """
---------------------------------------------------------------------------
Computes the Mean Squared Log Error.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.

Returns
-------
float
	score
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [str], False),
            ("input_relation", input_relation, [str], False),
        ]
    )
    if not (cursor):
        conn = read_auto_connect()
        cursor = conn.cursor()
    else:
        conn = False
        check_cursor(cursor)
    query = "SELECT AVG(POW(LOG({} + 1) - LOG({} + 1), 2)) FROM {}".format(
        y_true, y_score, input_relation
    )
    cursor.execute(query)
    result = cursor.fetchone()[0]
    if conn:
        conn.close()
    return result


# ---#
def median_absolute_error(y_true: str, y_score: str, input_relation: str, cursor=None):
    """
---------------------------------------------------------------------------
Computes the Median Absolute Error.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.

Returns
-------
float
	score
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [str], False),
            ("input_relation", input_relation, [str], False),
        ]
    )
    if not (cursor):
        conn = read_auto_connect()
        cursor = conn.cursor()
    else:
        conn = False
        check_cursor(cursor)
    query = "SELECT APPROXIMATE_MEDIAN(ABS({} - {})) FROM {}".format(
        y_true, y_score, input_relation
    )
    cursor.execute(query)
    result = cursor.fetchone()[0]
    if conn:
        conn.close()
    return result


# ---#
def r2_score(y_true: str, y_score: str, input_relation: str, cursor=None):
    """
---------------------------------------------------------------------------
Computes the R2 Score.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.

Returns
-------
float
	score
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [str], False),
            ("input_relation", input_relation, [str], False),
        ]
    )
    if not (cursor):
        conn = read_auto_connect()
        cursor = conn.cursor()
    else:
        conn = False
        check_cursor(cursor)
    query = "SELECT RSQUARED({}, {}) OVER() FROM {}".format(
        y_true, y_score, input_relation
    )
    cursor.execute(query)
    result = cursor.fetchone()[0]
    if conn:
        conn.close()
    return result


# ---#
def regression_report(y_true: str, y_score: str, input_relation: str, cursor=None):
    """
---------------------------------------------------------------------------
Computes a regression report using multiple metrics (r2, mse, max error...). 

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.

Returns
-------
tablesample
 	An object containing the result. For more information, see
 	utilities.tablesample.
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [str], False),
            ("input_relation", input_relation, [str], False),
        ]
    )
    if not (cursor):
        conn = read_auto_connect()
        cursor = conn.cursor()
    else:
        conn = False
        check_cursor(cursor)
    query = "SELECT 1 - VARIANCE({} - {}) / VARIANCE({}), MAX(ABS({} - {})), ".format(
        y_true, y_score, y_true, y_true, y_score
    )
    query += "APPROXIMATE_MEDIAN(ABS({} - {})), AVG(ABS({} - {})), ".format(
        y_true, y_score, y_true, y_score
    )
    query += "AVG(POW({} - {}, 2)) FROM {}".format(y_true, y_score, input_relation)
    r2 = r2_score(y_true, y_score, input_relation, cursor)
    values = {
        "index": [
            "explained_variance",
            "max_error",
            "median_absolute_error",
            "mean_absolute_error",
            "mean_squared_error",
            "r2",
        ]
    }
    cursor.execute(query)
    values["value"] = [item for item in cursor.fetchone()] + [r2]
    if conn:
        conn.close()
    return tablesample(values, table_info=False)


#
# Classification
#
# ---#
def accuracy_score(
    y_true: str, y_score: str, input_relation: str, cursor=None, pos_label=1
):
    """
---------------------------------------------------------------------------
Computes the Accuracy Score.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.
pos_label: int/float/str, optional
	Label to use to identify the positive class. If pos_label is NULL then the
	global accuracy will be computed.

Returns
-------
float
	score
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [str], False),
            ("input_relation", input_relation, [str], False),
        ]
    )
    if not (cursor):
        conn = read_auto_connect()
        cursor = conn.cursor()
    else:
        conn = False
        check_cursor(cursor)
    if pos_label != None:
        matrix = confusion_matrix(y_true, y_score, input_relation, cursor, pos_label)
        non_pos_label = 0 if (pos_label == 1) else "Non-{}".format(pos_label)
        tn, fn, fp, tp = (
            matrix.values[non_pos_label][0],
            matrix.values[non_pos_label][1],
            matrix.values[pos_label][0],
            matrix.values[pos_label][1],
        )
        acc = (tp + tn) / (tp + tn + fn + fp)
    else:
        try:
            query = "SELECT AVG(CASE WHEN {} = {} THEN 1 ELSE 0 END) AS accuracy FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL"
            query = query.format(y_true, y_score, input_relation, y_true, y_score)
            cursor.execute(query)
        except:
            query = "SELECT AVG(CASE WHEN {}::varchar = {}::varchar THEN 1 ELSE 0 END) AS accuracy FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL"
            query = query.format(y_true, y_score, input_relation, y_true, y_score)
            cursor.execute(query)
        acc = cursor.fetchone()[0]
    if conn:
        conn.close()
    return acc


# ---#
def auc(y_true: str, y_score: str, input_relation: str, cursor=None, pos_label=1):
    """
---------------------------------------------------------------------------
Computes the ROC AUC (Area Under Curve).

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction Probability.
input_relation: str
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.
pos_label: int/float/str, optional
	To compute the ROC AUC, one of the response column class has to be the 
	positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [str], False),
            ("input_relation", input_relation, [str], False),
        ]
    )
    return roc_curve(
        y_true, y_score, input_relation, cursor, pos_label, nbins=10000, auc_roc=True
    )


# ---#
def classification_report(
    y_true: str = "",
    y_score: list = [],
    input_relation: str = "",
    cursor=None,
    labels: list = [],
    cutoff=[],
    estimator=None,
):
    """
---------------------------------------------------------------------------
Computes a classification report using multiple metrics (AUC, accuracy, PRC 
AUC, F1...). It will consider each category as positive and switch to the 
next one during the computation.

Parameters
----------
y_true: str, optional
	Response column.
y_score: list, optional
	List containing the probability and the prediction.
input_relation: str, optional
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.
labels: list, optional
	List of the response column categories to use.
cutoff: float / list, optional
	Cutoff for which the tested category will be accepted as prediction. 
	In case of multiclass classification, the list will represent the 
	the classes threshold. If it is empty, the best cutoff will be used.
estimator: object, optional
	Estimator to use to compute the classification report.

Returns
-------
tablesample
 	An object containing the result. For more information, see
 	utilities.tablesample.
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [list], False),
            ("input_relation", input_relation, [str], False),
            ("labels", labels, [list], False),
            ("cutoff", cutoff, [int, float, list], False),
        ]
    )
    if estimator:
        num_classes = len(estimator.classes)
        labels = labels if (num_classes > 2) else [estimator.classes[1]]
    else:
        if not (cursor):
            conn = read_auto_connect()
            cursor = conn.cursor()
        else:
            conn = False
            check_cursor(cursor)
        labels = [1] if not (labels) else labels
        num_classes = len(labels) + 1
    values = {
        "index": [
            "auc",
            "prc_auc",
            "accuracy",
            "log_loss",
            "precision",
            "recall",
            "f1_score",
            "mcc",
            "informedness",
            "markedness",
            "csi",
            "cutoff",
        ]
    }
    for idx, elem in enumerate(labels):
        pos_label = elem
        non_pos_label = 0 if (elem == 1) else "Non-{}".format(elem)
        if estimator:
            if not (cutoff):
                current_cutoff = estimator.score(
                    method="best_cutoff", pos_label=pos_label
                )
            elif type(cutoff) == list:
                if len(cutoff) == 1:
                    current_cutoff = cutoff[0]
                else:
                    current_cutoff = cutoff[idx]
            else:
                current_cutoff = cutoff
            try:
                matrix = estimator.confusion_matrix(pos_label, current_cutoff)
            except:
                matrix = estimator.confusion_matrix(pos_label)
        else:
            y_s, y_p, y_t = (
                y_score[0].format(elem),
                y_score[1],
                "DECODE({}, '{}', 1, 0)".format(y_true, elem),
            )
            matrix = confusion_matrix(y_true, y_p, input_relation, cursor, pos_label)
        try:
            tn, fn, fp, tp = (
                matrix.values[non_pos_label][0],
                matrix.values[non_pos_label][1],
                matrix.values[pos_label][0],
                matrix.values[pos_label][1],
            )
        except:
            try:
                tn, fn, fp, tp = (
                    matrix.values[0][0],
                    matrix.values[0][1],
                    matrix.values[1][0],
                    matrix.values[1][1],
                )
            except:
                tn, fn, fp, tp = (
                    matrix.values["0"][0],
                    matrix.values["0"][1],
                    matrix.values["1"][0],
                    matrix.values["1"][1],
                )
        ppv = tp / (tp + fp) if (tp + fp != 0) else 0  # precision
        tpr = tp / (tp + fn) if (tp + fn != 0) else 0  # recall
        tnr = tn / (tn + fp) if (tn + fp != 0) else 0
        npv = tn / (tn + fn) if (tn + fn != 0) else 0
        f1 = 2 * (tpr * tnr) / (tpr + tnr) if (tpr + tnr != 0) else 0  # f1
        csi = tp / (tp + fn + fp) if (tp + fn + fp != 0) else 0  # csi
        bm = tpr + tnr - 1  # informedness
        mk = ppv + npv - 1  # markedness
        mcc = (
            (tp * tn - fp * fn)
            / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            if (tp + fp != 0) and (tp + fn != 0) and (tn + fp != 0) and (tn + fn != 0)
            else 0
        )
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        if estimator:
            auc_score, logloss, prc_auc_score = (
                estimator.score(pos_label=pos_label, method="auc"),
                estimator.score(pos_label=pos_label, method="log_loss"),
                estimator.score(pos_label=pos_label, method="prc_auc"),
            )
        else:
            auc_score = auc(y_t, y_s, input_relation, cursor, 1)
            prc_auc_score = prc_auc(y_t, y_s, input_relation, cursor, 1)
            y_p = "DECODE({}, '{}', 1, 0)".format(y_p, elem)
            logloss = log_loss(y_t, y_s, input_relation, cursor, 1)
            if not (cutoff):
                current_cutoff = roc_curve(
                    y_t, y_p, input_relation, cursor, best_threshold=True
                )
            elif type(cutoff) == list:
                if len(cutoff) == 1:
                    current_cutoff = cutoff[0]
                else:
                    current_cutoff = cutoff[idx]
            else:
                current_cutoff = cutoff
        elem = "value" if (len(labels) == 1) else elem
        values[elem] = [
            auc_score,
            prc_auc_score,
            accuracy,
            logloss,
            ppv,
            tpr,
            f1,
            mcc,
            bm,
            mk,
            csi,
            current_cutoff,
        ]
    if not (estimator):
        if conn:
            conn.close()
    return tablesample(values, table_info=False)


# ---#
def confusion_matrix(
    y_true: str, y_score: str, input_relation: str, cursor=None, pos_label=1
):
    """
---------------------------------------------------------------------------
Computes the Confusion Matrix.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.
pos_label: int/float/str, optional
	To compute the one dimension Confusion Matrix, one of the response column 
	class has to be the positive one. The parameter 'pos_label' represents 
	this class.

Returns
-------
tablesample
 	An object containing the result. For more information, see
 	utilities.tablesample.
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [str], False),
            ("input_relation", input_relation, [str], False),
        ]
    )
    if not (cursor):
        conn = read_auto_connect()
        cursor = conn.cursor()
    else:
        conn = False
        check_cursor(cursor)
    query = "SELECT CONFUSION_MATRIX(obs, response USING PARAMETERS num_classes = 2) OVER() FROM (SELECT DECODE({}".format(
        y_true
    )
    query += ", '{}', 1, NULL, NULL, 0) AS obs, DECODE({}, '{}', 1, NULL, NULL, 0) AS response FROM {}) x".format(
        pos_label, y_score, pos_label, input_relation
    )
    result = to_tablesample(query, cursor)
    if conn:
        conn.close()
    if pos_label in [1, "1"]:
        labels = [0, 1]
    else:
        labels = ["Non-{}".format(pos_label), pos_label]
    result.table_info = False
    del result.values["comment"]
    result = result.transpose()
    result.values["actual_class"] = labels
    result = result.transpose()
    matrix = {"index": labels}
    for elem in result.values:
        if elem != "actual_class":
            matrix[elem] = result.values[elem]
    result.values = matrix
    return result


# ---#
def critical_success_index(
    y_true: str, y_score: str, input_relation: str, cursor=None, pos_label=1
):
    """
---------------------------------------------------------------------------
Computes the Critical Success Index.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.
pos_label: int/float/str, optional
	To compute the CSI, one of the response column class has to be the 
	positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [str], False),
            ("input_relation", input_relation, [str], False),
        ]
    )
    if not (cursor):
        conn = read_auto_connect()
        cursor = conn.cursor()
    else:
        conn = False
        check_cursor(cursor)
    matrix = confusion_matrix(y_true, y_score, input_relation, cursor, pos_label)
    if conn:
        conn.close()
    non_pos_label = 0 if (pos_label == 1) else "Non-{}".format(pos_label)
    tn, fn, fp, tp = (
        matrix.values[non_pos_label][0],
        matrix.values[non_pos_label][1],
        matrix.values[pos_label][0],
        matrix.values[pos_label][1],
    )
    csi = tp / (tp + fn + fp) if (tp + fn + fp != 0) else 0
    return csi


# ---#
def f1_score(y_true: str, y_score: str, input_relation: str, cursor=None, pos_label=1):
    """
---------------------------------------------------------------------------
Computes the F1 Score.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.
pos_label: int/float/str, optional
	To compute the F1 Score, one of the response column class has to be the 
	positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [str], False),
            ("input_relation", input_relation, [str], False),
        ]
    )
    if not (cursor):
        conn = read_auto_connect()
        cursor = conn.cursor()
    else:
        conn = False
        check_cursor(cursor)
    matrix = confusion_matrix(y_true, y_score, input_relation, cursor, pos_label)
    if conn:
        conn.close()
    non_pos_label = 0 if (pos_label == 1) else "Non-{}".format(pos_label)
    tn, fn, fp, tp = (
        matrix.values[non_pos_label][0],
        matrix.values[non_pos_label][1],
        matrix.values[pos_label][0],
        matrix.values[pos_label][1],
    )
    recall = tp / (tp + fn) if (tp + fn != 0) else 0
    precision = tp / (tp + fp) if (tp + fp != 0) else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall != 0)
        else 0
    )
    return f1


# ---#
def informedness(
    y_true: str, y_score: str, input_relation: str, cursor=None, pos_label=1
):
    """
---------------------------------------------------------------------------
Computes the Informedness.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.
pos_label: int/float/str, optional
	To compute the informedness, one of the response column class has to be the 
	positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [str], False),
            ("input_relation", input_relation, [str], False),
        ]
    )
    if not (cursor):
        conn = read_auto_connect()
        cursor = conn.cursor()
    else:
        conn = False
        check_cursor(cursor)
    matrix = confusion_matrix(y_true, y_score, input_relation, cursor, pos_label)
    if conn:
        conn.close()
    non_pos_label = 0 if (pos_label == 1) else "Non-{}".format(pos_label)
    tn, fn, fp, tp = (
        matrix.values[non_pos_label][0],
        matrix.values[non_pos_label][1],
        matrix.values[pos_label][0],
        matrix.values[pos_label][1],
    )
    tpr = tp / (tp + fn) if (tp + fn != 0) else 0
    tnr = tn / (tn + fp) if (tn + fp != 0) else 0
    return tpr + tnr - 1


# ---#
def log_loss(y_true: str, y_score: str, input_relation: str, cursor=None, pos_label=1):
    """
---------------------------------------------------------------------------
Computes the Log Loss.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction Probability.
input_relation: str
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.
pos_label: int/float/str, optional
	To compute the log loss, one of the response column class has to be the 
	positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [str], False),
            ("input_relation", input_relation, [str], False),
        ]
    )
    if not (cursor):
        conn = read_auto_connect()
        cursor = conn.cursor()
    else:
        conn = False
        check_cursor(cursor)
    query = "SELECT AVG(CASE WHEN {} = '{}' THEN - LOG({}::float + 1e-90) else - LOG(1 - {}::float + 1e-90) END) FROM {};"
    query = query.format(y_true, pos_label, y_score, y_score, input_relation)
    cursor.execute(query)
    result = cursor.fetchone()[0]
    if conn:
        conn.close()
    return result


# ---#
def markedness(
    y_true: str, y_score: str, input_relation: str, cursor=None, pos_label=1
):
    """
---------------------------------------------------------------------------
Computes the Markedness.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.
pos_label: int/float/str, optional
	To compute the markedness, one of the response column class has to be the 
	positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [str], False),
            ("input_relation", input_relation, [str], False),
        ]
    )
    if not (cursor):
        conn = read_auto_connect()
        cursor = conn.cursor()
    else:
        conn = False
        check_cursor(cursor)
    matrix = confusion_matrix(y_true, y_score, input_relation, cursor, pos_label)
    if conn:
        conn.close()
    non_pos_label = 0 if (pos_label == 1) else "Non-{}".format(pos_label)
    tn, fn, fp, tp = (
        matrix.values[non_pos_label][0],
        matrix.values[non_pos_label][1],
        matrix.values[pos_label][0],
        matrix.values[pos_label][1],
    )
    ppv = tp / (tp + fp) if (tp + fp != 0) else 0
    npv = tn / (tn + fn) if (tn + fn != 0) else 0
    return ppv + npv - 1


# ---#
def matthews_corrcoef(
    y_true: str, y_score: str, input_relation: str, cursor=None, pos_label=1
):
    """
---------------------------------------------------------------------------
Computes the Matthews Correlation Coefficient.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.
pos_label: int/float/str, optional
	To compute the Matthews Correlation Coefficient, one of the response column 
	class has to be the positive one. The parameter 'pos_label' represents this 
	class.

Returns
-------
float
	score
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [str], False),
            ("input_relation", input_relation, [str], False),
        ]
    )
    if not (cursor):
        conn = read_auto_connect()
        cursor = conn.cursor()
    else:
        conn = False
        check_cursor(cursor)
    matrix = confusion_matrix(y_true, y_score, input_relation, cursor, pos_label)
    if conn:
        conn.close()
    non_pos_label = 0 if (pos_label == 1) else "Non-{}".format(pos_label)
    tn, fn, fp, tp = (
        matrix.values[non_pos_label][0],
        matrix.values[non_pos_label][1],
        matrix.values[pos_label][0],
        matrix.values[pos_label][1],
    )
    mcc = (
        (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if (tp + fp != 0) and (tp + fn != 0) and (tn + fp != 0) and (tn + fn != 0)
        else 0
    )
    return mcc


# ---#
def multilabel_confusion_matrix(
    y_true: str, y_score: str, input_relation: str, labels: list, cursor=None
):
    """
---------------------------------------------------------------------------
Computes the Multi Label Confusion Matrix.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
labels: list
	List of the response column categories.
cursor: DBcursor, optional
	Vertica DB cursor.

Returns
-------
tablesample
 	An object containing the result. For more information, see
 	utilities.tablesample.
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [str], False),
            ("input_relation", input_relation, [str], False),
            ("labels", labels, [list], False),
        ]
    )
    if not (cursor):
        conn = read_auto_connect()
        cursor = conn.cursor()
    else:
        conn = False
        check_cursor(cursor)
    num_classes = str(len(labels))
    query = "SELECT CONFUSION_MATRIX(obs, response USING PARAMETERS num_classes = {}) OVER() FROM (SELECT DECODE({}".format(
        num_classes, y_true
    )
    for idx, item in enumerate(labels):
        query += ", '{}', {}".format(item, idx)
    query += ") AS obs, DECODE({}".format(y_score)
    for idx, item in enumerate(labels):
        query += ", '{}', {}".format(item, idx)
    query += ") AS response FROM {}) x".format(input_relation)
    result = to_tablesample(query, cursor)
    if conn:
        conn.close()
    result.table_info = False
    del result.values["comment"]
    result = result.transpose()
    result.values["actual_class"] = labels
    result = result.transpose()
    matrix = {"index": labels}
    for elem in result.values:
        if elem != "actual_class":
            matrix[elem] = result.values[elem]
    result.values = matrix
    return result


# ---#
def negative_predictive_score(
    y_true: str, y_score: str, input_relation: str, cursor=None, pos_label=1
):
    """
---------------------------------------------------------------------------
Computes the Negative Predictive Score.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.
pos_label: int/float/str, optional
	To compute the Negative Predictive Score, one of the response column class 
	has to be the positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [str], False),
            ("input_relation", input_relation, [str], False),
        ]
    )
    if not (cursor):
        conn = read_auto_connect()
        cursor = conn.cursor()
    else:
        conn = False
        check_cursor(cursor)
    matrix = confusion_matrix(y_true, y_score, input_relation, cursor, pos_label)
    if conn:
        conn.close()
    non_pos_label = 0 if (pos_label == 1) else "Non-{}".format(pos_label)
    tn, fn, fp, tp = (
        matrix.values[non_pos_label][0],
        matrix.values[non_pos_label][1],
        matrix.values[pos_label][0],
        matrix.values[pos_label][1],
    )
    npv = tn / (tn + fn) if (tn + fn != 0) else 0
    return npv


# ---#
def prc_auc(y_true: str, y_score: str, input_relation: str, cursor=None, pos_label=1):
    """
---------------------------------------------------------------------------
Computes the PRC AUC (Area Under Curve).

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction Probability.
input_relation: str
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.
pos_label: int/float/str, optional
	To compute the PRC AUC, one of the response column class has to be the 
	positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [str], False),
            ("input_relation", input_relation, [str], False),
        ]
    )
    return prc_curve(
        y_true, y_score, input_relation, cursor, pos_label, nbins=10000, auc_prc=True
    )


# ---#
def precision_score(
    y_true: str, y_score: str, input_relation: str, cursor=None, pos_label=1
):
    """
---------------------------------------------------------------------------
Computes the Precision Score.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.
pos_label: int/float/str, optional
	To compute the Precision Score, one of the response column class has to be 
	the positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [str], False),
            ("input_relation", input_relation, [str], False),
        ]
    )
    if not (cursor):
        conn = read_auto_connect()
        cursor = conn.cursor()
    else:
        conn = False
        check_cursor(cursor)
    matrix = confusion_matrix(y_true, y_score, input_relation, cursor, pos_label)
    if conn:
        conn.close()
    non_pos_label = 0 if (pos_label == 1) else "Non-{}".format(pos_label)
    tn, fn, fp, tp = (
        matrix.values[non_pos_label][0],
        matrix.values[non_pos_label][1],
        matrix.values[pos_label][0],
        matrix.values[pos_label][1],
    )
    precision = tp / (tp + fp) if (tp + fp != 0) else 0
    return precision


# ---#
def recall_score(
    y_true: str, y_score: str, input_relation: str, cursor=None, pos_label=1
):
    """
---------------------------------------------------------------------------
Computes the Recall Score.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.
pos_label: int/float/str, optional
	To compute the Recall Score, one of the response column class has to be 
	the positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [str], False),
            ("input_relation", input_relation, [str], False),
        ]
    )
    if not (cursor):
        conn = read_auto_connect()
        cursor = conn.cursor()
    else:
        conn = False
        check_cursor(cursor)
    matrix = confusion_matrix(y_true, y_score, input_relation, cursor, pos_label)
    if conn:
        conn.close()
    non_pos_label = 0 if (pos_label == 1) else "Non-{}".format(pos_label)
    tn, fn, fp, tp = (
        matrix.values[non_pos_label][0],
        matrix.values[non_pos_label][1],
        matrix.values[pos_label][0],
        matrix.values[pos_label][1],
    )
    recall = tp / (tp + fn) if (tp + fn != 0) else 0
    return recall


# ---#
def specificity_score(
    y_true: str, y_score: str, input_relation: str, cursor=None, pos_label=1
):
    """
---------------------------------------------------------------------------
Computes the Specificity Score.

Parameters
----------
y_true: str
	Response column.
y_score: str
	Prediction.
input_relation: str
	Relation to use to do the scoring. The relation can be a view or a table
	or even a customized relation. For example, you could write:
	"(SELECT ... FROM ...) x" as long as an alias is given at the end of the
	relation.
cursor: DBcursor, optional
	Vertica DB cursor.
pos_label: int/float/str, optional
	To compute the Specificity Score, one of the response column class has to 
	be the positive one. The parameter 'pos_label' represents this class.

Returns
-------
float
	score
	"""
    check_types(
        [
            ("y_true", y_true, [str], False),
            ("y_score", y_score, [str], False),
            ("input_relation", input_relation, [str], False),
        ]
    )
    if not (cursor):
        conn = read_auto_connect()
        cursor = conn.cursor()
    else:
        conn = False
        check_cursor(cursor)
    matrix = confusion_matrix(y_true, y_score, input_relation, cursor, pos_label)
    if conn:
        conn.close()
    non_pos_label = 0 if (pos_label == 1) else "Non-{}".format(pos_label)
    tn, fn, fp, tp = (
        matrix.values[non_pos_label][0],
        matrix.values[non_pos_label][1],
        matrix.values[pos_label][0],
        matrix.values[pos_label][1],
    )
    tnr = tn / (tn + fp) if (tn + fp != 0) else 0
    return tnr
