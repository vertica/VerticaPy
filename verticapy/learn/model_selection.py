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
import statistics, random
from collections.abc import Iterable

# VerticaPy Modules
from verticapy import vDataFrame
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.errors import *
from verticapy.plot import gen_colors

# Other Python Modules
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---#
def best_k(
    input_relation: (str, vDataFrame),
    X: list = [],
    cursor=None,
    n_cluster: (tuple, list) = (1, 100),
    init: (str, list) = "kmeanspp",
    max_iter: int = 50,
    tol: float = 1e-4,
    elbow_score_stop: float = 0.8,
):
    """
---------------------------------------------------------------------------
Finds the KMeans K based on a score.

Parameters
----------
input_relation: str/vDataFrame
    Relation to use to train the model.
X: list, optional
	List of the predictor columns. If empty, all the numerical columns will
    be used.
cursor: DBcursor, optional
	Vertica DB cursor.
n_cluster: tuple/list, optional
	Tuple representing the number of cluster to start with and to end with.
	It can also be customized list with the different K to test.
init: str/list, optional
	The method to use to find the initial cluster centers.
		kmeanspp : Use the KMeans++ method to initialize the centers.
		random   : The initial centers
	It can be also a list with the initial cluster centers to use.
max_iter: int, optional
	The maximum number of iterations the algorithm performs.
tol: float, optional
	Determines whether the algorithm has converged. The algorithm is considered 
	converged after no center has moved more than a distance of 'tol' from the 
	previous iteration.
elbow_score_stop: float, optional
	Stops the Parameters Search when this Elbow score is reached.

Returns
-------
int
	the KMeans K
	"""
    check_types(
        [
            ("X", X, [list],),
            ("input_relation", input_relation, [str, vDataFrame],),
            ("n_cluster", n_cluster, [list],),
            ("init", init, ["kmeanspp", "random"],),
            ("max_iter", max_iter, [int, float],),
            ("tol", tol, [int, float],),
            ("elbow_score_stop", elbow_score_stop, [int, float],),
        ]
    )

    from verticapy.learn.cluster import KMeans

    cursor, conn = check_cursor(cursor, input_relation)[0:2]
    if isinstance(n_cluster, tuple):
        L = range(n_cluster[0], n_cluster[1])
    else:
        L = n_cluster
        L.sort()
    schema, relation = schema_relation(input_relation)
    if isinstance(input_relation, vDataFrame):
        if not (schema):
            schema = "public"
    schema = str_column(schema)
    for i in L:
        cursor.execute(
            "DROP MODEL IF EXISTS {}.__VERTICAPY_TEMP_MODEL_KMEANS_{}__".format(
                schema, get_session(cursor)
            )
        )
        model = KMeans(
            "{}.__VERTICAPY_TEMP_MODEL_KMEANS_{}__".format(schema, get_session(cursor)),
            cursor,
            i,
            init,
            max_iter,
            tol,
        )
        model.fit(input_relation, X)
        score = model.metrics_.values["value"][3]
        if score > elbow_score_stop:
            return i
        score_prev = score
    if conn:
        conn.close()
    print(
        "\u26A0 The K was not found. The last K (= {}) is returned with an elbow score of {}".format(
            i, score
        )
    )
    return i


# ---#
def cross_validate(
    estimator,
    input_relation: (str, vDataFrame),
    X: list,
    y: str,
    cv: int = 3,
    pos_label: (int, float, str) = None,
    cutoff: float = -1,
):
    """
---------------------------------------------------------------------------
Computes the K-Fold cross validation of an estimator.

Parameters
----------
estimator: object
	Vertica estimator having a fit method and a DB cursor.
input_relation: str/vDataFrame
	Relation to use to train the model.
X: list
	List of the predictor columns.
y: str
	Response Column.
cv: int, optional
	Number of folds.
pos_label: int/float/str, optional
	The main class to be considered as positive (classification only).
cutoff: float, optional
	The model cutoff (classification only).

Returns
-------
tablesample
 	An object containing the result. For more information, see
 	utilities.tablesample.
	"""
    check_types(
        [
            ("X", X, [list],),
            ("input_relation", input_relation, [str, vDataFrame],),
            ("y", y, [str],),
            ("cv", cv, [int, float],),
            ("cutoff", cutoff, [int, float],),
        ]
    )
    if isinstance(input_relation, str):
        input_relation = vdf_from_relation(input_relation, cursor=estimator.cursor)
    if cv < 2:
        raise ParameterError("Cross Validation is only possible with at least 2 folds")
    if estimator.type in (
        "RandomForestRegressor",
        "LinearSVR",
        "LinearRegression",
        "KNeighborsRegressor",
    ):
        result = {
            "index": [
                "explained_variance",
                "max_error",
                "median_absolute_error",
                "mean_absolute_error",
                "mean_squared_error",
                "r2",
            ]
        }
    elif estimator.type in (
        "NaiveBayes",
        "RandomForestClassifier",
        "LinearSVC",
        "LogisticRegression",
        "KNeighborsClassifier",
        "NearestCentroid",
    ):
        result = {
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
            ]
        }
    else:
        raise Exception(
            "Cross Validation is only possible for Regressors and Classifiers"
        )
    try:
        schema = schema_relation(estimator.name)[0]
    except:
        schema = schema_relation(input_relation)[0]
    try:
        input_relation.set_schema_writing(str_column(schema)[1:-1])
    except:
        pass
    for i in range(cv):
        try:
            estimator.drop()
        except:
            pass
        random_state = verticapy.options["random_state"]
        random_state = (
            random.randint(-10e6, 10e6) if not (random_state) else random_state + i
        )
        train, test = input_relation.train_test_split(
            test_size=float(1 / cv), order_by=[X[0]], random_state=random_state
        )
        estimator.fit(
            train, X, y, test,
        )
        if estimator.type in (
            "RandomForestRegressor",
            "LinearSVR",
            "LinearRegression",
            "KNeighborsRegressor",
        ):
            result["{}-fold".format(i + 1)] = estimator.regression_report().values[
                "value"
            ]
        else:
            if (len(estimator.classes_) > 2) and (pos_label not in estimator.classes_):
                raise ParameterError(
                    "'pos_label' must be in the estimator classes, it must be the main class to study for the Cross Validation"
                )
            elif (len(estimator.classes_) == 2) and (
                pos_label not in estimator.classes_
            ):
                pos_label = estimator.classes_[1]
            try:
                result["{}-fold".format(i + 1)] = estimator.classification_report(
                    labels=[pos_label], cutoff=cutoff
                ).values["value"][0:-1]
            except:
                result["{}-fold".format(i + 1)] = estimator.classification_report(
                    cutoff=cutoff
                ).values["value"][0:-1]
        try:
            estimator.drop()
        except:
            pass
    n = (
        6
        if (
            estimator.type
            in (
                "RandomForestRegressor",
                "LinearSVR",
                "LinearRegression",
                "KNeighborsRegressor",
            )
        )
        else 11
    )
    total = [[] for item in range(n)]
    for i in range(cv):
        for k in range(n):
            total[k] += [result["{}-fold".format(i + 1)][k]]
    result["avg"], result["std"] = [], []
    for item in total:
        result["avg"] += [statistics.mean([float(elem) for elem in item])]
        result["std"] += [statistics.stdev([float(elem) for elem in item])]
    return tablesample(values=result).transpose()


# ---#
def elbow(
    input_relation: (str, vDataFrame),
    X: list = [],
    cursor=None,
    n_cluster: (tuple, list) = (1, 15),
    init: (str, list) = "kmeanspp",
    max_iter: int = 50,
    tol: float = 1e-4,
    ax=None,
):
    """
---------------------------------------------------------------------------
Draws an Elbow Curve.

Parameters
----------
input_relation: str/vDataFrame
    Relation to use to train the model.
X: list, optional
    List of the predictor columns. If empty all the numerical vcolumns will
    be used.
cursor: DBcursor, optional
    Vertica DB cursor.
n_cluster: tuple/list, optional
    Tuple representing the number of cluster to start with and to end with.
    It can also be customized list with the different K to test.
init: str/list, optional
    The method to use to find the initial cluster centers.
        kmeanspp : Use the KMeans++ method to initialize the centers.
        random   : The initial centers
    It can be also a list with the initial cluster centers to use.
max_iter: int, optional
    The maximum number of iterations the algorithm performs.
tol: float, optional
    Determines whether the algorithm has converged. The algorithm is considered 
    converged after no center has moved more than a distance of 'tol' from the 
    previous iteration.
ax: Matplotlib axes object, optional
    The axes to plot on.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    check_types(
        [
            ("X", X, [list],),
            ("input_relation", input_relation, [str, vDataFrame],),
            ("n_cluster", n_cluster, [list],),
            ("init", init, ["kmeanspp", "random"],),
            ("max_iter", max_iter, [int, float],),
            ("tol", tol, [int, float],),
        ]
    )
    cursor, conn = check_cursor(cursor, input_relation)[0:2]
    version(cursor=cursor, condition=[8, 0, 0])
    if isinstance(n_cluster, tuple):
        L = range(n_cluster[0], n_cluster[1])
    else:
        L = n_cluster
        L.sort()
    schema, relation = schema_relation(input_relation)
    all_within_cluster_SS = []
    if isinstance(n_cluster, tuple):
        L = [i for i in range(n_cluster[0], n_cluster[1])]
    else:
        L = n_cluster
        L.sort()
    for i in L:
        cursor.execute(
            "DROP MODEL IF EXISTS {}.VERTICAPY_KMEANS_TMP_{}".format(
                schema, get_session(cursor)
            )
        )
        from verticapy.learn.cluster import KMeans

        model = KMeans(
            "{}.VERTICAPY_KMEANS_TMP_{}".format(schema, get_session(cursor)),
            cursor,
            i,
            init,
            max_iter,
            tol,
        )
        model.fit(input_relation, X)
        all_within_cluster_SS += [float(model.metrics_.values["value"][3])]
        model.drop()
    if conn:
        conn.close()
    if not (ax):
        fig, ax = plt.subplots()
        if isnotebook():
            fig.set_size_inches(8, 6)
    ax.set_facecolor("#F5F5F5")
    ax.grid()
    ax.plot(L, all_within_cluster_SS, marker="s", color=gen_colors()[0])
    ax.set_title("Elbow Curve")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Between-Cluster SS / Total SS")
    values = {"index": L, "Within-Cluster SS": all_within_cluster_SS}
    return tablesample(values=values)


# ---#
def lift_chart(
    y_true: str,
    y_score: str,
    input_relation: (str, vDataFrame),
    cursor=None,
    pos_label: (int, float, str) = 1,
    nbins: int = 1000,
    ax=None,
):
    """
---------------------------------------------------------------------------
Draws the Lift Chart.

Parameters
----------
y_true: str
    Response column.
y_score: str
    Prediction Probability.
input_relation: str/vDataFrame
    Relation to use to do the scoring. The relation can be a view or a table
    or even a customized relation. For example, you could write:
    "(SELECT ... FROM ...) x" as long as an alias is given at the end of the
    relation.
cursor: DBcursor, optional
    Vertica DB cursor.
pos_label: int/float/str, optional
    To compute the Lift Chart, one of the response column class has to be the 
    positive one. The parameter 'pos_label' represents this class.
nbins: int, optional
    Curve number of bins.
ax: Matplotlib axes object, optional
    The axes to plot on.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    check_types(
        [
            ("y_true", y_true, [str],),
            ("y_score", y_score, [str],),
            ("input_relation", input_relation, [str, vDataFrame],),
            ("nbins", nbins, [int, float],),
        ]
    )
    cursor, conn, input_relation = check_cursor(cursor, input_relation)
    version(cursor=cursor, condition=[8, 0, 0])
    query = "SELECT LIFT_TABLE(obs, prob USING PARAMETERS num_bins = {}) OVER() FROM (SELECT (CASE WHEN {} = '{}' THEN 1 ELSE 0 END) AS obs, {}::float AS prob FROM {}) AS prediction_output"
    query = query.format(nbins, y_true, pos_label, y_score, input_relation)
    executeSQL(cursor, query, "Computing the Lift Table.")
    query_result = cursor.fetchall()
    if conn:
        conn.close()
    decision_boundary, positive_prediction_ratio, lift = (
        [item[0] for item in query_result],
        [item[1] for item in query_result],
        [item[2] for item in query_result],
    )
    decision_boundary.reverse()
    if not (ax):
        fig, ax = plt.subplots()
        if isnotebook():
            fig.set_size_inches(8, 6)
    ax.set_facecolor("#F5F5F5")
    ax.set_xlabel("Cumulative Data Fraction")
    ax.plot(decision_boundary, lift, color=gen_colors()[0])
    ax.plot(decision_boundary, positive_prediction_ratio, color="#444444")
    ax.set_title("Lift Table")
    ax.set_axisbelow(True)
    ax.grid()
    color1 = mpatches.Patch(color=gen_colors()[0], label="Cumulative Lift")
    color2 = mpatches.Patch(color="#444444", label="Cumulative Capture Rate")
    ax.legend(handles=[color1, color2])
    return tablesample(
        values={
            "decision_boundary": decision_boundary,
            "positive_prediction_ratio": positive_prediction_ratio,
            "lift": lift,
        },
    )


# ---#
def prc_curve(
    y_true: str,
    y_score: str,
    input_relation: (str, vDataFrame),
    cursor=None,
    pos_label: (int, float, str) = 1,
    nbins: int = 1000,
    auc_prc: bool = False,
    ax=None,
):
    """
---------------------------------------------------------------------------
Draws the PRC Curve.

Parameters
----------
y_true: str
    Response column.
y_score: str
    Prediction Probability.
input_relation: str/vDataFrame
    Relation to use to do the scoring. The relation can be a view or a table
    or even a customized relation. For example, you could write:
    "(SELECT ... FROM ...) x" as long as an alias is given at the end of the
    relation.
cursor: DBcursor, optional
    Vertica DB cursor.
pos_label: int/float/str, optional
    To compute the PRC Curve, one of the response column class has to be the 
    positive one. The parameter 'pos_label' represents this class.
nbins: int, optional
    Curve number of bins.
auc_prc: bool, optional
    If set to True, the function will return the PRC AUC without drawing the 
    curve.
ax: Matplotlib axes object, optional
    The axes to plot on.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    check_types(
        [
            ("y_true", y_true, [str],),
            ("y_score", y_score, [str],),
            ("input_relation", input_relation, [str, vDataFrame],),
            ("nbins", nbins, [int, float],),
            ("auc_prc", auc_prc, [bool],),
        ]
    )
    cursor, conn, input_relation = check_cursor(cursor, input_relation)
    version(cursor=cursor, condition=[9, 1, 0])
    query = "SELECT PRC(obs, prob USING PARAMETERS num_bins = {}) OVER() FROM (SELECT (CASE WHEN {} = '{}' THEN 1 ELSE 0 END) AS obs, {}::float AS prob FROM {}) AS prediction_output"
    query = query.format(nbins, y_true, pos_label, y_score, input_relation)
    executeSQL(cursor, query, "Computing the PRC table.")
    query_result = cursor.fetchall()
    if conn:
        conn.close()
    threshold, recall, precision = (
        [0] + [item[0] for item in query_result] + [1],
        [1] + [item[1] for item in query_result] + [0],
        [0] + [item[2] for item in query_result] + [1],
    )
    auc = 0
    for i in range(len(recall) - 1):
        if recall[i + 1] - recall[i] != 0.0:
            a = (precision[i + 1] - precision[i]) / (recall[i + 1] - recall[i])
            b = precision[i + 1] - a * recall[i + 1]
            auc = (
                auc
                + a * (recall[i + 1] * recall[i + 1] - recall[i] * recall[i]) / 2
                + b * (recall[i + 1] - recall[i])
            )
    auc = -auc
    if auc_prc:
        return auc
    if not (ax):
        fig, ax = plt.subplots()
        if isnotebook():
            fig.set_size_inches(8, 6)
    ax.set_facecolor("#F5F5F5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.plot(recall, precision, color=gen_colors()[0])
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_title("PRC Curve\nAUC = " + str(auc))
    ax.set_axisbelow(True)
    ax.grid()
    return tablesample(
        values={"threshold": threshold, "recall": recall, "precision": precision},
    )


# ---#
def roc_curve(
    y_true: str,
    y_score: str,
    input_relation: (str, vDataFrame),
    cursor=None,
    pos_label: (int, float, str) = 1,
    nbins: int = 1000,
    auc_roc: bool = False,
    best_threshold: bool = False,
    ax=None,
):
    """
---------------------------------------------------------------------------
Draws the ROC Curve.

Parameters
----------
y_true: str
    Response column.
y_score: str
    Prediction Probability.
input_relation: str/vDataFrame
    Relation to use to do the scoring. The relation can be a view or a table
    or even a customized relation. For example, you could write:
    "(SELECT ... FROM ...) x" as long as an alias is given at the end of the
    relation.
cursor: DBcursor, optional
    Vertica DB cursor.
pos_label: int/float/str, optional
    To compute the PRC Curve, one of the response column class has to be the 
    positive one. The parameter 'pos_label' represents this class.
nbins: int, optional
    Curve number of bins.
auc_roc: bool, optional
    If set to true, the function will return the ROC AUC without drawing the 
    curve.
best_threshold: bool, optional
    If set to True, the function will return the best threshold without drawing 
    the curve. The best threshold is the threshold of the point which is the 
    farest from the random line.
ax: Matplotlib axes object, optional
    The axes to plot on.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    check_types(
        [
            ("y_true", y_true, [str],),
            ("y_score", y_score, [str],),
            ("input_relation", input_relation, [str, vDataFrame],),
            ("nbins", nbins, [int, float],),
            ("auc_roc", auc_roc, [bool],),
            ("best_threshold", best_threshold, [bool],),
        ]
    )
    cursor, conn, input_relation = check_cursor(cursor, input_relation)
    version(cursor=cursor, condition=[8, 0, 0])
    query = "SELECT ROC(obs, prob USING PARAMETERS num_bins = {}) OVER() FROM (SELECT (CASE WHEN {} = '{}' THEN 1 ELSE 0 END) AS obs, {}::float AS prob FROM {}) AS prediction_output"
    query = query.format(nbins, y_true, pos_label, y_score, input_relation)
    executeSQL(cursor, query, "Computing the ROC Table.")
    query_result = cursor.fetchall()
    if conn:
        conn.close()
    threshold, false_positive, true_positive = (
        [item[0] for item in query_result],
        [item[1] for item in query_result],
        [item[2] for item in query_result],
    )
    auc = 0
    for i in range(len(false_positive) - 1):
        if false_positive[i + 1] - false_positive[i] != 0.0:
            a = (true_positive[i + 1] - true_positive[i]) / (
                false_positive[i + 1] - false_positive[i]
            )
            b = true_positive[i + 1] - a * false_positive[i + 1]
            auc = (
                auc
                + a
                * (
                    false_positive[i + 1] * false_positive[i + 1]
                    - false_positive[i] * false_positive[i]
                )
                / 2
                + b * (false_positive[i + 1] - false_positive[i])
            )
    auc = -auc
    auc = min(auc, 1.0)
    if auc_roc:
        return auc
    if best_threshold:
        l = [abs(y - x) for x, y in zip(false_positive, true_positive)]
        best_threshold_arg = max(zip(l, range(len(l))))[1]
        best = max(threshold[best_threshold_arg], 0.001)
        best = min(best, 0.999)
        return best
    if not (ax):
        fig, ax = plt.subplots()
        if isnotebook():
            fig.set_size_inches(8, 6)
    ax.set_xlabel("False Positive Rate (1-Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.plot(false_positive, true_positive, color=gen_colors()[0])
    ax.plot([0, 1], [0, 1], color="#444444")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_title("ROC Curve\nAUC = " + str(auc))
    ax.set_axisbelow(True)
    ax.grid()
    return tablesample(
        values={
            "threshold": threshold,
            "false_positive": false_positive,
            "true_positive": true_positive,
        },
    )
