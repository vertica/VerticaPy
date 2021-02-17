# (c) Copyright [2018-2021] Micro Focus or one of its affiliates.
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
import statistics, random, time
from collections.abc import Iterable
from itertools import product
import numpy as np

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
def autoML(
    input_relation: (str, vDataFrame),
    X: list,
    y: str,
    estimator: (list, str) = "native",
    metric: str = "auto",
    cv: int = 3,
    pos_label: (int, float, str) = None,
    cutoff: float = -1,
    nbins: int = 1000,
    lmax: int = 10,
    optimized_grid: int = 2,
    cursor=None,
    print_info: bool = True,
    show: bool = True,
    ax=None,
    **style_kwds,
):
    """
---------------------------------------------------------------------------
Tests multiple models to find the ones which maximize the input score.

Parameters
----------
input_relation: str/vDataFrame
    Relation to use to train the model.
X: list
    List of the predictor columns.
y: str
    Response Column.
estimator: list / 'native' / 'all' / object
    List of Vertica estimators having a fit method and a DB cursor.
    It is also possible to set the value to 'native' (for all Vertica native
    models) or 'all' (for all VerticaPy models)
metric: str, optional
    Metric used to do the model evaluation.
        auto: logloss for classification & rmse for regression.
    For Classification:
        accuracy    : Accuracy
        auc         : Area Under the Curve (ROC)
        bm          : Informedness = tpr + tnr - 1
        csi         : Critical Success Index = tp / (tp + fn + fp)
        f1          : F1 Score 
        logloss     : Log Loss
        mcc         : Matthews Correlation Coefficient 
        mk          : Markedness = ppv + npv - 1
        npv         : Negative Predictive Value = tn / (tn + fn)
        prc_auc     : Area Under the Curve (PRC)
        precision   : Precision = tp / (tp + fp)
        recall      : Recall = tp / (tp + fn)
        specificity : Specificity = tn / (tn + fp)
    For Regression:
        max    : Max Error
        mae    : Mean Absolute Error
        median : Median Absolute Error
        mse    : Mean Squared Error
        msle   : Mean Squared Log Error
        r2     : R squared coefficient
        r2a    : R2 adjusted
        rmse   : Root Mean Squared Error
        var    : Explained Variance 
cv: int, optional
    Number of folds.
pos_label: int/float/str, optional
    The main class to be considered as positive (classification only).
cutoff: float, optional
    The model cutoff (classification only).
nbins: int, optional
    Number of bins used to compute the different parameters categories.
lmax: int, optional
    Maximum length of each parameter list.
optimized_grid: int, optional
    If set to 0, the randomness is based on the input parameters.
    If set to 1, the randomness is limited to some parameters, the other
    ones are picked based on a default grid.
    If set to 2, there is no randomness and a default grid is returned.
cursor: DBcursor, optional
    Vertica DB cursor.
print_info: bool
    If set to True, prints the model information at each step.
show: bool, optional
    If set to True, the ML Bubble Plot will be drawn using Matplotlib.
ax: Matplotlib axes object, optional
    The axes to plot on.
**style_kwds
    Any optional parameter to pass to the Matplotlib functions.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    from verticapy.learn.ensemble import RandomForestRegressor, RandomForestClassifier, XGBoostRegressor, XGBoostClassifier
    from verticapy.learn.naive_bayes import NaiveBayes
    from verticapy.learn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, LogisticRegression
    from verticapy.learn.neighbors import KNeighborsRegressor, KNeighborsClassifier, NearestCentroid
    from verticapy.learn.svm import LinearSVC, LinearSVR

    name = "verticapy_autoML_temp_name"
    conn = False
    if isinstance(estimator, str):
        estimator = estimator.lower()
        check_types([("estimator", estimator, ["native", "all"],),])
        modeltype = None
        if not(isinstance(input_relation, vDataFrame)):
            cursor, conn = check_cursor(cursor, input_relation)[0:2]
            vdf = vdf_from_relation(input_relation, cursor=cursor)
        else:
            cursor, conn = input_relation._VERTICAPY_VARIABLES_["cursor"], False
            vdf = input_relation
        if sorted(vdf[y].distinct()) == [0, 1]:
            if estimator == "native":
                estimator = [LinearSVC(name, cursor=cursor), LogisticRegression(name, cursor=cursor), RandomForestClassifier(name, cursor=cursor), XGBoostClassifier(name, cursor=cursor), NaiveBayes(name, cursor=cursor)]
            else:
                estimator = [LinearSVC(name, cursor=cursor), LogisticRegression(name, cursor=cursor), RandomForestClassifier(name, cursor=cursor), XGBoostClassifier(name, cursor=cursor), NaiveBayes(name, cursor=cursor), KNeighborsClassifier(name, cursor=cursor), NearestCentroid(name, cursor=cursor)]
        elif vdf[y].isnum():
            if estimator == "native":
                estimator = [LinearSVR(name, cursor=cursor), ElasticNet(name, cursor=cursor), LinearRegression(name, cursor=cursor), Lasso(name, cursor=cursor), Ridge(name, cursor=cursor), RandomForestRegressor(name, cursor=cursor), XGBoostRegressor(name, cursor=cursor),]
            else:
                estimator = [LinearSVR(name, cursor=cursor), ElasticNet(name, cursor=cursor), LinearRegression(name, cursor=cursor), Lasso(name, cursor=cursor), Ridge(name, cursor=cursor), RandomForestRegressor(name, cursor=cursor), XGBoostRegressor(name, cursor=cursor), KNeighborsRegressor(name, cursor=cursor),]
        else:
            if estimator == "native":
                estimator = [RandomForestClassifier(name, cursor=cursor), XGBoostClassifier(name, cursor=cursor), NaiveBayes(name, cursor=cursor)]
            else:
                estimator = [RandomForestClassifier(name, cursor=cursor), XGBoostClassifier(name, cursor=cursor), NaiveBayes(name, cursor=cursor), KNeighborsClassifier(name, cursor=cursor), NearestCentroid(name, cursor=cursor)]
    elif isinstance(estimator, (RandomForestRegressor, RandomForestClassifier, XGBoostRegressor, XGBoostClassifier, NaiveBayes, LinearRegression, ElasticNet, Lasso, Ridge, LogisticRegression, KNeighborsRegressor, KNeighborsClassifier, NearestCentroid, LinearSVC, LinearSVR)):
        estimator = [estimator]
    else:
        for elem in estimator:
            assert isinstance(elem, (RandomForestRegressor, RandomForestClassifier, XGBoostRegressor, XGBoostClassifier, NaiveBayes, LinearRegression, ElasticNet, Lasso, Ridge, LogisticRegression, KNeighborsRegressor, KNeighborsClassifier, NearestCentroid, LinearSVC, LinearSVR)), ParameterError("estimator must be a list of VerticaPy estimators. Found {}.".format(type(elem)))
    result = tablesample(
                {
                    "model_type": [],
                    "parameters": [],
                    "avg_score": [],
                    "avg_train_score": [],
                    "avg_time": [],
                    "score_std": [],
                    "score_train_std": [],
                }
            )
    for elem in estimator:
        if print_info:
            print(f"\033[1m\033[4mTesting Model - {str(elem.__class__).split('.')[-1][:-2]}\033[0m\033[0m\n")
        if category_from_model_type(elem.type)[0] == "regressor" and metric == "auto":
            metric = "rmse"
        elif metric == "auto":
            metric = "logloss"
        param_grid = gen_params_grid(elem, nbins, len(X), lmax, optimized_grid)
        gs = grid_search_cv(
            elem,
            param_grid,
            input_relation,
            X,
            y,
            metric,
            cv,
            pos_label,
            cutoff,
            True,
            "no_print",
            print_info,
        )
        if gs["parameters"] != [] and gs["avg_score"] != [] and gs["avg_train_score"] != [] and gs["avg_time"] != [] and gs["score_std"] != [] and gs["score_train_std"] != []:
            result.values["model_type"] += [str(elem.__class__).split('.')[-1][:-2]] * len(gs["parameters"])
            result.values["parameters"] += gs["parameters"]
            result.values["avg_score"] += gs["avg_score"]
            result.values["avg_train_score"] += gs["avg_train_score"]
            result.values["avg_time"] += gs["avg_time"]
            result.values["score_std"] += gs["score_std"]
            result.values["score_train_std"] += gs["score_train_std"]

    data = [(result["model_type"][i], result["parameters"][i], result["avg_score"][i], result["avg_train_score"][i], result["avg_time"][i], result["score_std"][i], result["score_train_std"][i]) for i in range(len(result["score_train_std"]))]
    reverse = True

    if metric in [
        "logloss",
        "max",
        "mae",
        "median",
        "mse",
        "msle",
        "rmse",
        "aic",
        "bic",
    ]:
        reverse = False
    data.sort(key=lambda tup: tup[2], reverse=reverse)
    result = tablesample(
        {
            "model_type": [elem[0] for elem in data],
            "parameters": [elem[1] for elem in data],
            "avg_score": [elem[2] for elem in data],
            "avg_train_score": [elem[3] for elem in data],
            "avg_time": [elem[4] for elem in data],
            "score_std": [elem[5] for elem in data],
            "score_train_std": [elem[6] for elem in data],
        }
    )
    if print_info:
        print(f"\n\n\033[1mBest_Model: {result['model_type'][0]}; Best_Parameters: {result['parameters'][0]}; \033[91mBest_Test_score: {result['avg_score'][0]}\033[0m; \033[92mTrain_score: {result['avg_train_score'][0]}\033[0m; \033[94mTime: {result['avg_time'][0]}\033[0m;\033[0m\n\n")
    if conn:
        conn.close()
    if show:
        from verticapy.learn.mlplot import plot_bubble_ml

        plot_bubble_ml(result["avg_time"], result["avg_score"], result["score_std"], result["model_type"], x_label="time", y_label="score", title= "Model Type", ax=ax, reverse=reverse, **style_kwds)
    return result


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
    if isinstance(X, str):
        X = [X]
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
    metric: (str, list) = "all",
    cv: int = 3,
    pos_label: (int, float, str) = None,
    cutoff: float = -1,
    show_time: bool = True,
    training_score: bool = False,
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
metric: str/list, optional
    Metric used to do the model evaluation. It can also be a list of metrics.
        all: The model will compute all the possible metrics.
    For Classification:
        accuracy    : Accuracy
        auc         : Area Under the Curve (ROC)
        best_cutoff : Cutoff which optimised the ROC Curve prediction.
        bm          : Informedness = tpr + tnr - 1
        csi         : Critical Success Index = tp / (tp + fn + fp)
        f1          : F1 Score 
        logloss     : Log Loss
        mcc         : Matthews Correlation Coefficient 
        mk          : Markedness = ppv + npv - 1
        npv         : Negative Predictive Value = tn / (tn + fn)
        prc_auc     : Area Under the Curve (PRC)
        precision   : Precision = tp / (tp + fp)
        recall      : Recall = tp / (tp + fn)
        specificity : Specificity = tn / (tn + fp)
    For Regression:
        aic    : Akaike’s Information Criterion
        bic    : Bayesian Information Criterion
        max    : Max Error
        mae    : Mean Absolute Error
        median : Median Absolute Error
        mse    : Mean Squared Error
        msle   : Mean Squared Log Error
        r2     : R squared coefficient
        r2a    : R2 adjusted
        rmse   : Root Mean Squared Error
        var    : Explained Variance 
cv: int, optional
	Number of folds.
pos_label: int/float/str, optional
	The main class to be considered as positive (classification only).
cutoff: float, optional
	The model cutoff (classification only).
show_time: bool, optional
    If set to True, the time and the average time will be added to the report.
training_score: bool, optional
    If set to True, the training score will be computed with the validation score.

Returns
-------
tablesample
 	An object containing the result. For more information, see
 	utilities.tablesample.
	"""
    if isinstance(X, str):
        X = [X]
    check_types(
        [
            ("X", X, [list],),
            ("input_relation", input_relation, [str, vDataFrame],),
            ("y", y, [str],),
            ("metric", metric, [str, list],),
            ("cv", cv, [int, float],),
            ("cutoff", cutoff, [int, float],),
        ]
    )
    if isinstance(input_relation, str):
        input_relation = vdf_from_relation(input_relation, cursor=estimator.cursor)
    if cv < 2:
        raise ParameterError("Cross Validation is only possible with at least 2 folds")
    if category_from_model_type(estimator.type)[0] == "regressor":
        all_metrics = [
            "explained_variance",
            "max_error",
            "median_absolute_error",
            "mean_absolute_error",
            "mean_squared_error",
            "root_mean_squared_error",
            "r2",
            "r2_adj",
            "aic",
            "bic",
        ]
    elif category_from_model_type(estimator.type)[0] == "classifier":
        all_metrics = [
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
    else:
        raise Exception(
            "Cross Validation is only possible for Regressors and Classifiers"
        )
    if metric == "all":
        final_metrics = all_metrics
    elif isinstance(metric, str):
        final_metrics = [metric]
    else:
        final_metrics = metric
    result = {"index": final_metrics}
    if training_score:
        result_train = {"index": final_metrics}
    try:
        schema = schema_relation(estimator.name)[0]
    except:
        schema = schema_relation(input_relation)[0]
    try:
        input_relation.set_schema_writing(str_column(schema)[1:-1])
    except:
        pass
    total_time = []
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
        start_time = time.time()
        estimator.fit(
            train, X, y, test,
        )
        total_time += [time.time() - start_time]
        if category_from_model_type(estimator.type)[0] == "regressor":
            if metric == "all":
                result["{}-fold".format(i + 1)] = estimator.regression_report().values[
                    "value"
                ]
                if training_score:
                    estimator.test_relation = estimator.input_relation
                    result_train[
                        "{}-fold".format(i + 1)
                    ] = estimator.regression_report().values["value"]
            elif isinstance(metric, str):
                result["{}-fold".format(i + 1)] = [estimator.score(metric)]
                if training_score:
                    estimator.test_relation = estimator.input_relation
                    result_train["{}-fold".format(i + 1)] = [estimator.score(metric)]
            else:
                result["{}-fold".format(i + 1)] = [estimator.score(m) for m in metric]
                if training_score:
                    estimator.test_relation = estimator.input_relation
                    result_train["{}-fold".format(i + 1)] = [
                        estimator.score(m) for m in metric
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
                if metric == "all":
                    result["{}-fold".format(i + 1)] = estimator.classification_report(
                        labels=[pos_label], cutoff=cutoff
                    ).values["value"][0:-1]
                    if training_score:
                        estimator.test_relation = estimator.input_relation
                        result_train[
                            "{}-fold".format(i + 1)
                        ] = estimator.classification_report(
                            labels=[pos_label], cutoff=cutoff
                        ).values[
                            "value"
                        ][
                            0:-1
                        ]

                elif isinstance(metric, str):
                    result["{}-fold".format(i + 1)] = [
                        estimator.score(metric, pos_label=pos_label, cutoff=cutoff)
                    ]
                    if training_score:
                        estimator.test_relation = estimator.input_relation
                        result_train["{}-fold".format(i + 1)] = [
                            estimator.score(metric, pos_label=pos_label, cutoff=cutoff)
                        ]
                else:
                    result["{}-fold".format(i + 1)] = [
                        estimator.score(m, pos_label=pos_label, cutoff=cutoff)
                        for m in metric
                    ]
                    if training_score:
                        estimator.test_relation = estimator.input_relation
                        result_train["{}-fold".format(i + 1)] = [
                            estimator.score(m, pos_label=pos_label, cutoff=cutoff)
                            for m in metric
                        ]
            except:
                if metric == "all":
                    result["{}-fold".format(i + 1)] = estimator.classification_report(
                        cutoff=cutoff
                    ).values["value"][0:-1]
                    if training_score:
                        estimator.test_relation = estimator.input_relation
                        result_train[
                            "{}-fold".format(i + 1)
                        ] = estimator.classification_report(cutoff=cutoff).values[
                            "value"
                        ][
                            0:-1
                        ]
                elif isinstance(metric, str):
                    result["{}-fold".format(i + 1)] = [
                        estimator.score(metric, cutoff=cutoff)
                    ]
                    if training_score:
                        estimator.test_relation = estimator.input_relation
                        result_train["{}-fold".format(i + 1)] = [
                            estimator.score(metric, cutoff=cutoff)
                        ]
                else:
                    result["{}-fold".format(i + 1)] = [
                        estimator.score(m, cutoff=cutoff) for m in metric
                    ]
                    if training_score:
                        estimator.test_relation = estimator.input_relation
                        result_train["{}-fold".format(i + 1)] = [
                            estimator.score(m, cutoff=cutoff) for m in metric
                        ]
        try:
            estimator.drop()
        except:
            pass
    n = len(final_metrics)
    total = [[] for item in range(n)]
    for i in range(cv):
        for k in range(n):
            total[k] += [result["{}-fold".format(i + 1)][k]]
    if training_score:
        total_train = [[] for item in range(n)]
        for i in range(cv):
            for k in range(n):
                total_train[k] += [result_train["{}-fold".format(i + 1)][k]]
    result["avg"], result["std"] = [], []
    if training_score:
        result_train["avg"], result_train["std"] = [], []
    for item in total:
        result["avg"] += [statistics.mean([float(elem) for elem in item])]
        result["std"] += [statistics.stdev([float(elem) for elem in item])]
    if training_score:
        for item in total_train:
            result_train["avg"] += [statistics.mean([float(elem) for elem in item])]
            result_train["std"] += [statistics.stdev([float(elem) for elem in item])]
    total_time += [
        statistics.mean([float(elem) for elem in total_time]),
        statistics.stdev([float(elem) for elem in total_time]),
    ]
    result = tablesample(values=result).transpose()
    if show_time:
        result.values["time"] = total_time
    if training_score:
        result_train = tablesample(values=result_train).transpose()
        if show_time:
            result_train.values["time"] = total_time
    if training_score:
        return result, result_train
    else:
        return result


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
    **style_kwds,
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
**style_kwds
    Any optional parameter to pass to the Matplotlib functions.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    if isinstance(X, str):
        X = [X]
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
        ax.grid(axis="y")
    param = {
        "color": gen_colors()[0],
        "marker": "o",
        "markerfacecolor": "white",
        "markersize": 7,
        "markeredgecolor": "black",
    }
    ax.plot(
        L, all_within_cluster_SS, **updated_dict(param, style_kwds),
    )
    ax.set_title("Elbow Curve")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Between-Cluster SS / Total SS")
    values = {"index": L, "Within-Cluster SS": all_within_cluster_SS}
    return tablesample(values=values)


# ---#
def gen_params_grid(estimator, 
                    nbins: int = 10, 
                    max_nfeatures: int = 3,
                    lmax: int = -1,
                    optimized_grid: int = 0,):
    """
---------------------------------------------------------------------------
Generates the estimator grid.

Parameters
----------
estimator: object
    Vertica estimator having a fit method and a DB cursor.
nbins: int, optional
    Number of bins used to discretize numberical features.
max_nfeatures: int, optional
    Maximum number of features used to compute Random Forest, PCA...
lmax: int, optional
    Maximum length of the parameter grid.
optimized_grid: int, optional
    If set to 0, the randomness is based on the input parameters.
    If set to 1, the randomness is limited to some parameters, the other
    ones are picked based on a default grid.
    If set to 2, there is no randomness and a default grid is returned.
    
Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    from verticapy.learn.cluster import KMeans, BisectingKMeans, DBSCAN
    from verticapy.learn.decomposition import PCA, SVD
    from verticapy.learn.ensemble import RandomForestRegressor, RandomForestClassifier, XGBoostRegressor, XGBoostClassifier
    from verticapy.learn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, LogisticRegression
    from verticapy.learn.naive_bayes import NaiveBayes
    from verticapy.learn.neighbors import KNeighborsRegressor, KNeighborsClassifier, LocalOutlierFactor, NearestCentroid
    from verticapy.learn.preprocessing import Normalizer, OneHotEncoder
    from verticapy.learn.svm import LinearSVC, LinearSVR
    from verticapy.learn.tree import DummyTreeRegressor, DummyTreeClassifier, DecisionTreeRegressor, DecisionTreeClassifier

    params_grid = {}
    if isinstance(estimator, (DummyTreeRegressor, DummyTreeClassifier, OneHotEncoder,)):
        return params_grid
    elif isinstance(estimator, (RandomForestRegressor, RandomForestClassifier, DecisionTreeRegressor, DecisionTreeClassifier,)):
        if optimized_grid == 0:
            params_grid = {"max_features": ["auto", "max"] + list(range(1, max_nfeatures, math.ceil(max_nfeatures / nbins))),
                           "max_leaf_nodes": list(range(1, int(1e9), math.ceil(int(1e9) / nbins))),
                           "max_depth": list(range(1, 100, math.ceil(100 / nbins))),
                           "min_samples_leaf": list(range(1, int(1e6), math.ceil(int(1e6) / nbins))),
                           "min_info_gain": [elem / 1000 for elem in range(1, 1000, math.ceil(1000 / nbins))],
                           "nbins": list(range(2, 100, math.ceil(100 / nbins))),}
            if isinstance(RandomForestRegressor, RandomForestClassifier,):
                params_grid["sample"] = [elem / 1000 for elem in range(1, 1000, math.ceil(1000 / nbins))]
                params_grid["n_estimators"] = list(range(1, 100, math.ceil(100 / nbins)))
        elif optimized_grid == 1:
            params_grid = {"max_features": ["auto", "max"],
                           "max_leaf_nodes": [32, 64, 128, 1000, 1e4, 1e6, 1e9],
                           "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50],
                           "min_samples_leaf": [1, 2, 3, 4, 5],
                           "min_info_gain": [0.0, 0.1, 0.2],
                           "nbins": [10, 15, 20, 25, 30, 35, 40],}
            if isinstance(RandomForestRegressor, RandomForestClassifier,):
                params_grid["sample"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                params_grid["n_estimators"] = [1, 5, 10, 15, 20, 30, 40, 50, 100]
        else:
            params_grid = {"max_features": ["auto", "max"],
                           "max_leaf_nodes": [32, 64, 128, 1000,],
                           "max_depth": [3, 4, 5, 6,],
                           "min_samples_leaf": [1, 2, 3,],
                           "min_info_gain": [0.0, 0.1,],
                           "nbins": [30, 40, 50,],}
            if isinstance(RandomForestRegressor, RandomForestClassifier,):
                params_grid["sample"] = [0.5, 0.6, 0.7,]
                params_grid["n_estimators"] = [5, 10, 20, 30,]
    elif isinstance(estimator, (LinearSVC, LinearSVR,)):
        if optimized_grid == 0:
            params_grid = {"tol": [1e-4, 1e-6, 1e-8],
                           "C": [elem / 1000 for elem in range(1, 5000, math.ceil(5000 / nbins))],
                           "fit_intercept": [False, True],
                           "intercept_mode": ["regularized", "unregularized"],
                           "max_iter": [100, 500, 1000],}
        elif optimized_grid == 1:
            params_grid = {"tol": [1e-8],
                           "C": [1, 2, 3, 4, 5],
                           "fit_intercept": [True],
                           "intercept_mode": ["regularized", "unregularized"],
                           "max_iter": [1000],}
        else:
            params_grid = {"tol": [1e-8],
                           "C": [0.5, 1, 2,],
                           "fit_intercept": [True],
                           "intercept_mode": ["regularized", "unregularized"],
                           "max_iter": [1000],}
    elif isinstance(estimator, (XGBoostClassifier, XGBoostRegressor,)):
        if optimized_grid == 0:
            params_grid = {"nbins": list(range(2, 100, math.ceil(100 / nbins))),
                           "max_depth": list(range(1, 20, math.ceil(100 / nbins))),
                           "weight_reg": [elem / 1000 for elem in range(1, 1000, math.ceil(1000 / nbins))],
                           "min_split_loss": [elem / 1000 for elem in range(1, 1000, math.ceil(1000 / nbins))],
                           "learning_rate": [elem / 1000 for elem in range(1, 1000, math.ceil(1000 / nbins))],
                           #"sample": [elem / 1000 for elem in range(1, 1000, math.ceil(1000 / nbins))],
                           "tol": [1e-4, 1e-6, 1e-8],
                           "max_ntree": list(range(1, 100, math.ceil(100 / nbins)))}
        elif optimized_grid == 1:
            params_grid = {"nbins": [10, 15, 20, 25, 30, 35, 40],
                           "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20,],
                           "weight_reg": [0.0, 0.5, 1.0, 2.0],
                           "min_split_loss": [0.0, 0.1, 0.25],
                           "learning_rate": [0.01, 0.05, 0.1, 1.0],
                           #"sample": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                           "tol": [1e-8],
                           "max_ntree": [1, 10, 20, 30, 40, 50, 100]}
        else:
            params_grid = {"nbins": [30, 40, 50,],
                           "max_depth": [3, 4, 5, 6,],
                           "weight_reg": [0.0, 0.25,],
                           "min_split_loss": [0.0, 0.1, 0.25,],
                           "learning_rate": [0.01, 0.05, 0.1, 1.0,],
                           #"sample": [0.5, 0.6, 0.7,],
                           "tol": [1e-8],
                           "max_ntree": [5, 10, 20,]}
    elif isinstance(estimator, NaiveBayes):
        if optimized_grid == 0:
            params_grid = {"alpha": [elem / 1000 for elem in range(1, 1000, math.ceil(1000 / nbins))]}
        elif optimized_grid == 1:
            params_grid = {"alpha": [0.5, 1.0,  1.5, 2.0, 5.0]}
        else:
            params_grid = {"alpha": [0.5, 1.0, 1.5,]}
    elif isinstance(estimator, (PCA, SVD)):
        if optimized_grid == 0:
            params_grid = {"max_features": list(range(1, max_nfeatures, math.ceil(max_nfeatures / nbins))),}
        if isinstance(estimator, (PCA,)):
            params_grid["scale"] = [False, True]
    elif isinstance(estimator, (Normalizer,)):
        params_grid = {"method": ["minmax", "robust_zscore", "zscore"]}
    elif isinstance(estimator, (KNeighborsRegressor, KNeighborsClassifier, LocalOutlierFactor, NearestCentroid,)):
        if optimized_grid == 0:
            params_grid = {"p": [1, 2] + list(range(3, 100, math.ceil(100 / (nbins - 2)))),}
            if isinstance(estimator, (KNeighborsRegressor, KNeighborsClassifier, LocalOutlierFactor,)):
                params_grid["n_neighbors"] =  list(range(1, 100, math.ceil(100 / (nbins))))
        elif optimized_grid == 1:
            params_grid = {"p": [1, 2, 3, 4],}
            if isinstance(estimator, (KNeighborsRegressor, KNeighborsClassifier, LocalOutlierFactor,)):
                params_grid["n_neighbors"] =  [1, 2, 3, 4, 5, 10, 20, 100]
        else:
            params_grid = {"p": [1, 2, 3,],}
            if isinstance(estimator, (KNeighborsRegressor, KNeighborsClassifier, LocalOutlierFactor,)):
                params_grid["n_neighbors"] =  [1, 2, 3, 4, 5, 10,]
    elif isinstance(estimator, (DBSCAN,)):
        if optimized_grid == 0:
            params_grid = {"p": [1, 2] + list(range(3, 100, math.ceil(100 / (nbins - 2)))),
                           "eps": [elem / 1000 for elem in range(1, 1000, math.ceil(1000 / nbins))],
                           "min_samples": list(range(1, 1000, math.ceil(1000 / nbins))),}
        elif optimized_grid == 1:
            params_grid = {"p": [1, 2, 3, 4],
                           "min_samples": [1, 2, 3, 4, 5, 10, 100],}
        else:
            params_grid = {"p": [1, 2, 3,],
                           "min_samples": [3, 4, 5, 10,],}
    elif isinstance(estimator, (LogisticRegression, LinearRegression, ElasticNet, Lasso, Ridge,)):
        if optimized_grid == 0:
            params_grid = {"tol": [1e-4, 1e-6, 1e-8],
                           "max_iter": [100, 500, 1000],}
            if isinstance(estimator, LogisticRegression):
                params_grid["penalty"] = ["none", "l1", "l2", "enet",]
            if isinstance(estimator, LinearRegression):
                params_grid["solver"] = ["newton", "bfgs",]
            elif isinstance(estimator, (Lasso, LogisticRegression, ElasticNet)):
                params_grid["solver"] = ["newton", "bfgs", "cgd",]
            if isinstance(estimator, (Lasso, Ridge, ElasticNet, LogisticRegression)):
                params_grid["C"] = [elem / 1000 for elem in range(1, 5000, math.ceil(5000 / nbins))]
            if isinstance(estimator, (LogisticRegression, ElasticNet)):
                params_grid["l1_ratio"] = [elem / 1000 for elem in range(1, 1000, math.ceil(1000 / nbins))]
        elif optimized_grid == 1:
            params_grid = {"tol": [1e-8],
                           "max_iter": [1000],}
            if isinstance(estimator, LogisticRegression):
                params_grid["penalty"] = ["none", "l1", "l2", "enet",]
            if isinstance(estimator, LinearRegression):
                params_grid["solver"] = ["newton", "bfgs",]
            elif isinstance(estimator, (Lasso, LogisticRegression, ElasticNet)):
                params_grid["solver"] = ["newton", "bfgs", "cgd",]
            if isinstance(estimator, (Lasso, Ridge, ElasticNet, LogisticRegression)):
                params_grid["C"] = [0.5, 1, 2, 3, 4, 5]
            if isinstance(estimator, (LogisticRegression,)):
                params_grid["penalty"] = ["none", "l1", "l2", "enet"]
            if isinstance(estimator, (LogisticRegression, ElasticNet)):
                params_grid["l1_ratio"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        else:
            params_grid = {"tol": [1e-8],
                           "max_iter": [1000],}
            if isinstance(estimator, LogisticRegression):
                params_grid["penalty"] = ["none", "l1", "l2", "enet",]
            if isinstance(estimator, LinearRegression):
                params_grid["solver"] = ["newton", "bfgs",]
            elif isinstance(estimator, (Lasso, LogisticRegression, ElasticNet)):
                params_grid["solver"] = ["newton", "bfgs", "cgd",]
            if isinstance(estimator, (Lasso, Ridge, ElasticNet, LogisticRegression)):
                params_grid["C"] = [0.5, 1, 2,]
            if isinstance(estimator, (LogisticRegression,)):
                params_grid["penalty"] = ["none", "l1", "l2", "enet"]
            if isinstance(estimator, (LogisticRegression, ElasticNet)):
                params_grid["l1_ratio"] = [0.2, 0.5, 0.8,]
    elif isinstance(estimator, KMeans):
        if optimized_grid == 0:
            params_grid = {"n_cluster": list(range(2, 100, math.ceil(100 / nbins))),
                           "init": ["kmeanspp", "random"],
                           "max_iter": [100, 500, 1000],
                           "tol": [1e-4, 1e-6, 1e-8],}
        elif optimized_grid == 1:
            params_grid = {"n_cluster": [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50, 100, 200, 300, 1000],
                           "init": ["kmeanspp", "random"],
                           "max_iter": [1000],
                           "tol": [1e-8],}
        else:
            params_grid = {"n_cluster": [2, 3, 4, 5, 10, 20, 100,],
                           "init": ["kmeanspp",],
                           "max_iter": [1000],
                           "tol": [1e-8],}
    elif isinstance(estimator, BisectingKMeans):
        if optimized_grid == 0:
            params_grid = {"n_cluster": list(range(2, 100, math.ceil(100 / nbins))),
                           "bisection_iterations": list(range(10, 1000, math.ceil(1000 / nbins))),
                           "split_method": ["size", "sum_squares"],
                           "min_divisible_cluster_size": list(range(2, 100, math.ceil(100 / nbins))),
                           "init": ["kmeanspp", "pseudo"],
                           "max_iter": [100, 500, 1000],
                           "tol": [1e-4, 1e-6, 1e-8],}
        elif optimized_grid == 1:
            params_grid = {"n_cluster": [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50, 100, 200, 300, 1000],
                           "bisection_iterations": list(range(10, 1000, math.ceil(1000 / nbins))),
                           "split_method": ["size", "sum_squares"],
                           "min_divisible_cluster_size": list(range(2, 100, math.ceil(100 / nbins))),
                           "init": ["kmeanspp", "pseudo"],
                           "max_iter": [1000],
                           "tol": [1e-8],}
        else:
            params_grid = {"n_cluster": [2, 3, 4, 5, 10, 20, 100,],
                           "bisection_iterations": [1, 2, 3,],
                           "split_method": ["sum_squares",],
                           "min_divisible_cluster_size": [2, 3, 4,],
                           "init": ["kmeanspp",],
                           "max_iter": [1000],
                           "tol": [1e-8],}
    params_grid = parameter_grid(params_grid)
    final_param_grid = []
    for param in params_grid:
        if "penalty" in param:
            if param["penalty"] in ("none", "l2") and "solver" in param and param["solver"] == "cgd":
                param["solver"] = "bfgs"
            if param["penalty"] in ("none", "l1", "l2") and "l1_ratio" in param:
                del param["l1_ratio"]
            if param["penalty"] in ("none",) and "C" in param:
                del param["C"]
            if param["penalty"] in ("l1", "enet",) and "solver" in param:
                param["solver"] = "cgd"
        if param not in final_param_grid:
            final_param_grid += [param]
    if len(final_param_grid) > lmax and lmax > 0:
        final_param_grid = random.sample(final_param_grid, lmax)
    return final_param_grid


# ---#
def grid_search_cv(
    estimator,
    param_grid: (dict, list),
    input_relation: (str, vDataFrame),
    X: list,
    y: str,
    metric: str = "auto",
    cv: int = 3,
    pos_label: (int, float, str) = None,
    cutoff: float = -1,
    training_score: bool = True,
    skip_error: bool = False,
    print_info: bool = False,
):
    """
---------------------------------------------------------------------------
Computes the K-Fold grid search of an estimator.

Parameters
----------
estimator: object
    Vertica estimator having a fit method and a DB cursor.
param_grid: dict/list
    Dictionary of the parameters to test. It can also be a list of the
    different combinations.
input_relation: str/vDataFrame
    Relation to use to train the model.
X: list
    List of the predictor columns.
y: str
    Response Column.
metric: str, optional
    Metric used to do the model evaluation.
        auto: logloss for classification & rmse for regression.
    For Classification:
        accuracy    : Accuracy
        auc         : Area Under the Curve (ROC)
        bm          : Informedness = tpr + tnr - 1
        csi         : Critical Success Index = tp / (tp + fn + fp)
        f1          : F1 Score 
        logloss     : Log Loss
        mcc         : Matthews Correlation Coefficient 
        mk          : Markedness = ppv + npv - 1
        npv         : Negative Predictive Value = tn / (tn + fn)
        prc_auc     : Area Under the Curve (PRC)
        precision   : Precision = tp / (tp + fp)
        recall      : Recall = tp / (tp + fn)
        specificity : Specificity = tn / (tn + fp)
    For Regression:
        max    : Max Error
        mae    : Mean Absolute Error
        median : Median Absolute Error
        mse    : Mean Squared Error
        msle   : Mean Squared Log Error
        r2     : R squared coefficient
        r2a    : R2 adjusted
        rmse   : Root Mean Squared Error
        var    : Explained Variance 
cv: int, optional
    Number of folds.
pos_label: int/float/str, optional
    The main class to be considered as positive (classification only).
cutoff: float, optional
    The model cutoff (classification only).
training_score: bool, optional
    If set to True, the training score will be computed with the validation score.
skip_error: bool, optional
    If set to True and an error occurs, it will be displayed and not raised.
print_info: bool
    If set to True, prints the model information at each step.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    if isinstance(X, str):
        X = [X]
    check_types(
        [
            ("metric", metric, [str]),
            ("param_grid", param_grid, [dict, list]),
            ("training_score", training_score, [bool]),
            ("skip_error", skip_error, [bool, str,]),
            ("print_info", print_info, [bool,]),
        ]
    )
    if category_from_model_type(estimator.type)[0] == "regressor" and metric == "auto":
        metric = "rmse"
    elif metric == "auto":
        metric = "logloss"
    if isinstance(param_grid, dict):
        for param in param_grid:
            assert isinstance(param_grid[param], Iterable) and not (
                isinstance(param_grid[param], str)
            ), ParameterError(
                f"When of type dictionary, the parameter 'param_grid' must be a dictionary where each value is a list of parameters, found {type(param_grid[param])} for parameter '{param}'."
            )
        all_configuration = parameter_grid(param_grid)
    else:
        for idx, param in enumerate(param_grid):
            assert isinstance(param, dict), ParameterError(
                f"When of type List, the parameter 'param_grid' must be a list of dictionaries, found {type(param)} for elem '{idx}'."
            )
        all_configuration = param_grid
    # testing all the config
    for config in all_configuration:
        estimator.set_params(config)
    # applying all the config
    data = []
    if all_configuration == []:
        all_configuration = [{}]
    for config in all_configuration:
        try:
            estimator.set_params(config)
            current_cv = cross_validate(
                estimator,
                input_relation,
                X,
                y,
                metric,
                cv,
                pos_label,
                cutoff,
                True,
                training_score,
            )
            if training_score:
                keys = [elem for elem in current_cv[0].values]
                data += [
                    (
                        estimator.get_params(),
                        current_cv[0][keys[1]][cv],
                        current_cv[1][keys[1]][cv],
                        current_cv[0][keys[2]][cv],
                        current_cv[0][keys[1]][cv + 1],
                        current_cv[1][keys[1]][cv + 1],
                    )
                ]
                if print_info:
                    print(f"Model: {str(estimator.__class__).split('.')[-1][:-2]}; Parameters: {config}; \033[91mTest_score: {current_cv[0][keys[1]][cv]}\033[0m; \033[92mTrain_score: {current_cv[1][keys[1]][cv]}\033[0m; \033[94mTime: {current_cv[0][keys[2]][cv]}\033[0m;")
            else:
                keys = [elem for elem in current_cv.values]
                data += [
                    (
                        config,
                        current_cv[keys[1]][cv],
                        current_cv[keys[2]][cv],
                        current_cv[keys[1]][cv + 1],
                    )
                ]
                if print_info:
                    print(f"Model: {str(estimator.__class__).split('.')[-1][:-2]}; Parameters: {config}; \033[91mTest_score: {current_cv[keys[1]][cv]}\033[0m; \033[94mTime:{current_cv[keys[2]][cv]}\033[0m;")
        except Exception as e:
            if skip_error and skip_error != "no_print":
                print(e)
            elif not(skip_error):
                raise (e)
    if not(data):
        if training_score:
            return tablesample(
                {
                    "parameters": [],
                    "avg_score": [],
                    "avg_train_score": [],
                    "avg_time": [],
                    "score_std": [],
                    "score_train_std": [],
                }
            )
        else:
            return tablesample(
                {
                    "parameters": [],
                    "avg_score": [],
                    "avg_time": [],
                    "score_std": [],
                }
            )
    reverse = True
    if metric in [
        "logloss",
        "max",
        "mae",
        "median",
        "mse",
        "msle",
        "rmse",
        "aic",
        "bic",
    ]:
        reverse = False
    data.sort(key=lambda tup: tup[1], reverse=reverse)
    if training_score:
        result = tablesample(
            {
                "parameters": [elem[0] for elem in data],
                "avg_score": [elem[1] for elem in data],
                "avg_train_score": [elem[2] for elem in data],
                "avg_time": [elem[3] for elem in data],
                "score_std": [elem[4] for elem in data],
                "score_train_std": [elem[5] for elem in data],
            }
        )
        if print_info:
            print(f"\n\033[1mModel: {str(estimator.__class__).split('.')[-1][:-2]}; Best_Parameters: {result['parameters'][0]}; \033[91mBest_Test_score: {result['avg_score'][0]}\033[0m; \033[92mTrain_score: {result['avg_train_score'][0]}\033[0m; \033[94mTime: {result['avg_time'][0]}\033[0m;\033[0m\n\n")
    else:
        result = tablesample(
            {
                "parameters": [elem[0] for elem in data],
                "avg_score": [elem[1] for elem in data],
                "avg_time": [elem[2] for elem in data],
                "score_std": [elem[3] for elem in data],
            }
        )
        if print_info:
            print(f"\n\033[1mModel: {str(estimator.__class__).split('.')[-1][:-2]}; Best_Parameters: {result['parameters'][0]}; \033[91mBest_Test_score: {result['avg_score'][0]}\033[0m; \033[94mTime: {result['avg_time'][0]}\033[0m\033[0m;\n")
    return result


# ---#
def learning_curve(
    estimator,
    input_relation: (str, vDataFrame),
    X: list,
    y: str,
    sizes: list = [0.1, 0.33, 0.55, 0.78, 1.0],
    method="efficiency",
    metric: str = "auto",
    cv: int = 3,
    pos_label: (int, float, str) = None,
    cutoff: float = -1,
    std_coeff: float = 1,
    ax=None,
    **style_kwds,
):
    """
---------------------------------------------------------------------------
Draws the Learning curve.

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
sizes: list, optional
    Different sizes of the dataset used to train the model. Multiple models
    will be trained using the different sizes.
method: str, optional
    Method used to plot the curve.
        efficiency  : draws train/test score vs sample size.
        performance : draws score vs time.
        scalability : draws time vs sample size.
metric: str, optional
    Metric used to do the model evaluation.
        auto: logloss for classification & rmse for regression.
    For Classification:
        accuracy    : Accuracy
        auc         : Area Under the Curve (ROC)
        bm          : Informedness = tpr + tnr - 1
        csi         : Critical Success Index = tp / (tp + fn + fp)
        f1          : F1 Score 
        logloss     : Log Loss
        mcc         : Matthews Correlation Coefficient 
        mk          : Markedness = ppv + npv - 1
        npv         : Negative Predictive Value = tn / (tn + fn)
        prc_auc     : Area Under the Curve (PRC)
        precision   : Precision = tp / (tp + fp)
        recall      : Recall = tp / (tp + fn)
        specificity : Specificity = tn / (tn + fp)
    For Regression:
        max    : Max Error
        mae    : Mean Absolute Error
        median : Median Absolute Error
        mse    : Mean Squared Error
        msle   : Mean Squared Log Error
        r2     : R squared coefficient
        r2a    : R2 adjusted
        rmse   : Root Mean Squared Error
        var    : Explained Variance 
cv: int, optional
    Number of folds.
pos_label: int/float/str, optional
    The main class to be considered as positive (classification only).
cutoff: float, optional
    The model cutoff (classification only).
std_coeff: float, optional
    Value of the standard deviation coefficient used to compute the area plot 
    around each score.
ax: Matplotlib axes object, optional
    The axes to plot on.
**style_kwds
    Any optional parameter to pass to the Matplotlib functions.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    check_types(
        [("method", method, ["efficiency", "performance", "scalability"],),]
    )
    from verticapy.plot import range_curve

    for s in sizes:
        assert 0 < s <= 1, ParameterError("Each size must be in ]0,1].")
    if category_from_model_type(estimator.type)[0] == "regressor" and metric == "auto":
        metric = "rmse"
    elif metric == "auto":
        metric = "logloss"
    if isinstance(input_relation, str):
        input_relation = vdf_from_relation(input_relation, cursor=estimator.cursor)
    lc_result_final = []
    sizes = sorted(set(sizes))
    for s in sizes:
        relation = input_relation.sample(x=s)
        lc_result = cross_validate(
            estimator, relation, X, y, metric, cv, pos_label, cutoff, True, True,
        )
        lc_result_final += [
            (
                relation.shape()[0],
                lc_result[0][metric][cv],
                lc_result[0][metric][cv + 1],
                lc_result[1][metric][cv],
                lc_result[1][metric][cv + 1],
                lc_result[0]["time"][cv],
                lc_result[0]["time"][cv + 1],
            )
        ]
    if method in ("efficiency", "scalability"):
        lc_result_final.sort(key=lambda tup: tup[0])
    else:
        lc_result_final.sort(key=lambda tup: tup[5])
    result = tablesample(
        {
            "n": [elem[0] for elem in lc_result_final],
            metric: [elem[1] for elem in lc_result_final],
            metric + "_std": [elem[2] for elem in lc_result_final],
            metric + "_train": [elem[3] for elem in lc_result_final],
            metric + "_train_std": [elem[4] for elem in lc_result_final],
            "time": [elem[5] for elem in lc_result_final],
            "time_std": [elem[6] for elem in lc_result_final],
        }
    )
    if method == "efficiency":
        X = result["n"]
        Y = [
            [
                [
                    result[metric][i] - std_coeff * result[metric + "_std"][i]
                    for i in range(len(sizes))
                ],
                result[metric],
                [
                    result[metric][i] + std_coeff * result[metric + "_std"][i]
                    for i in range(len(sizes))
                ],
            ],
            [
                [
                    result[metric + "_train"][i]
                    - std_coeff * result[metric + "_train_std"][i]
                    for i in range(len(sizes))
                ],
                result[metric + "_train"],
                [
                    result[metric + "_train"][i]
                    + std_coeff * result[metric + "_train_std"][i]
                    for i in range(len(sizes))
                ],
            ],
        ]
        x_label = "n"
        y_label = metric
        labels = [
            "test",
            "train",
        ]
    elif method == "performance":
        X = result["time"]
        Y = [
            [
                [
                    result[metric][i] - std_coeff * result[metric + "_std"][i]
                    for i in range(len(sizes))
                ],
                result[metric],
                [
                    result[metric][i] + std_coeff * result[metric + "_std"][i]
                    for i in range(len(sizes))
                ],
            ],
        ]
        x_label = "time"
        y_label = metric
        labels = []
    else:
        X = result["n"]
        Y = [
            [
                [
                    result["time"][i] - std_coeff * result["time_std"][i]
                    for i in range(len(sizes))
                ],
                result["time"],
                [
                    result["time"][i] + std_coeff * result["time_std"][i]
                    for i in range(len(sizes))
                ],
            ],
        ]
        x_label = "n"
        y_label = "time"
        labels = []
    range_curve(
        X, Y, x_label, y_label, ax, labels, **style_kwds,
    )
    return result


# ---#
def lift_chart(
    y_true: str,
    y_score: str,
    input_relation: (str, vDataFrame),
    cursor=None,
    pos_label: (int, float, str) = 1,
    nbins: int = 30,
    ax=None,
    **style_kwds,
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
**style_kwds
    Any optional parameter to pass to the Matplotlib functions.

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
    ax.set_xlabel("Cumulative Data Fraction")
    max_value = max([0 if elem != elem else elem for elem in lift])
    lift = [max_value if elem != elem else elem for elem in lift]
    param1 = {"color": gen_colors()[0]}
    ax.plot(
        decision_boundary, lift, **updated_dict(param1, style_kwds, 0),
    )
    param2 = {"color": gen_colors()[1]}
    ax.plot(
        decision_boundary,
        positive_prediction_ratio,
        **updated_dict(param2, style_kwds, 1),
    )
    color1, color2 = color_dict(style_kwds, 0), color_dict(style_kwds, 1)
    if color1 == color2:
        color2 = gen_colors()[1]
    ax.fill_between(
        decision_boundary, positive_prediction_ratio, lift, facecolor=color1, alpha=0.2
    )
    ax.fill_between(
        decision_boundary,
        [0 for elem in decision_boundary],
        positive_prediction_ratio,
        facecolor=color2,
        alpha=0.2,
    )
    ax.set_title("Lift Table")
    ax.set_axisbelow(True)
    ax.grid()
    color1 = mpatches.Patch(color=color1, label="Cumulative Lift")
    color2 = mpatches.Patch(color=color2, label="Cumulative Capture Rate")
    ax.legend(handles=[color1, color2], loc="center left", bbox_to_anchor=[1, 0.5])
    ax.set_xlim(0, 1)
    ax.set_ylim(0)
    return tablesample(
        values={
            "decision_boundary": decision_boundary,
            "positive_prediction_ratio": positive_prediction_ratio,
            "lift": lift,
        },
    )


# ---#
def parameter_grid(param_grid: dict,):
    """
---------------------------------------------------------------------------
Generates the list of the different input parameters grid combinations.

Parameters
----------
param_grid: dict
    Dictionary of parameters.

Returns
-------
list of dict
    List of the different combinations.
    """
    check_types([("param_grid", param_grid, [dict]),])
    return [dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())]


# ---#
def plot_acf_pacf(
    vdf: vDataFrame,
    column: str,
    ts: str,
    by: list = [],
    p: (int, list) = 15,
    **style_kwds,
):
    """
---------------------------------------------------------------------------
Draws the ACF and PACF Charts.

Parameters
----------
vdf: vDataFrame
    Input vDataFrame.
column: str
    Response column.
ts: str
    vcolumn used as timeline. It will be to use to order the data. 
    It can be a numerical or type date like (date, datetime, timestamp...) 
    vcolumn.
by: list, optional
    vcolumns used in the partition.
p: int/list, optional
    Int equals to the maximum number of lag to consider during the computation
    or List of the different lags to include during the computation.
    p must be positive or a list of positive integers.
**style_kwds
    Any optional parameter to pass to the Matplotlib functions.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    if isinstance(by, str):
        by = [by]
    check_types(
        [
            ("column", column, [str],),
            ("ts", ts, [str],),
            ("by", by, [list],),
            ("p", p, [int, float],),
            ("vdf", vdf, [vDataFrame,],),
        ]
    )
    tmp_style = {}
    for elem in style_kwds:
        if elem not in ("color", "colors"):
            tmp_style[elem] = style_kwds[elem]
    if "color" in style_kwds:
        color = style_kwds["color"]
    else:
        color = gen_colors()[0]
    columns_check([column, ts] + by, vdf)
    by = vdf_columns_names(by, vdf)
    column, ts = vdf_columns_names([column, ts], vdf)
    acf = vdf.acf(ts=ts, column=column, by=by, p=p, show=False)
    pacf = vdf.pacf(ts=ts, column=column, by=by, p=p, show=False)
    result = tablesample(
        {
            "index": [i for i in range(0, len(acf.values["value"]))],
            "acf": acf.values["value"],
            "pacf": pacf.values["value"],
            "confidence": pacf.values["confidence"],
        },
    )
    fig = plt.figure(figsize=(10, 6)) if isnotebook() else plt.figure(figsize=(10, 6))
    plt.rcParams["axes.facecolor"] = "#FCFCFC"
    ax1 = fig.add_subplot(211)
    x, y, confidence = (
        result.values["index"],
        result.values["acf"],
        result.values["confidence"],
    )
    plt.xlim(-1, x[-1] + 1)
    ax1.bar(
        x, y, width=0.007 * len(x), color="#444444", zorder=1, linewidth=0,
    )
    param = {
        "s": 90,
        "marker": "o",
        "facecolors": color,
        "edgecolors": "black",
        "zorder": 2,
    }
    ax1.scatter(
        x, y, **updated_dict(param, tmp_style,),
    )
    ax1.plot(
        [-1] + x + [x[-1] + 1],
        [0 for elem in range(len(x) + 2)],
        color=color,
        zorder=0,
    )
    ax1.fill_between(x, confidence, color="#FE5016", alpha=0.1)
    ax1.fill_between(x, [-elem for elem in confidence], color="#FE5016", alpha=0.1)
    ax1.set_title("Autocorrelation")
    y = result.values["pacf"]
    ax2 = fig.add_subplot(212)
    ax2.bar(x, y, width=0.007 * len(x), color="#444444", zorder=1, linewidth=0)
    ax2.scatter(
        x, y, **updated_dict(param, tmp_style,),
    )
    ax2.plot(
        [-1] + x + [x[-1] + 1],
        [0 for elem in range(len(x) + 2)],
        color=color,
        zorder=0,
    )
    ax2.fill_between(x, confidence, color="#FE5016", alpha=0.1)
    ax2.fill_between(x, [-elem for elem in confidence], color="#FE5016", alpha=0.1)
    ax2.set_title("Partial Autocorrelation")
    plt.show()
    return result


# ---#
def prc_curve(
    y_true: str,
    y_score: str,
    input_relation: (str, vDataFrame),
    cursor=None,
    pos_label: (int, float, str) = 1,
    nbins: int = 30,
    auc_prc: bool = False,
    ax=None,
    **style_kwds,
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
**style_kwds
    Any optional parameter to pass to the Matplotlib functions.

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
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    param = {"color": color_dict(style_kwds, 0)}
    ax.plot(recall, precision, **updated_dict(param, style_kwds))
    ax.fill_between(
        recall,
        [0 for item in recall],
        precision,
        facecolor=color_dict(style_kwds, 0),
        alpha=0.1,
    )
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_title("PRC Curve")
    ax.text(
        0.995,
        0,
        "AUC = " + str(round(auc, 4) * 100) + "%",
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize=11.5,
    )
    ax.set_axisbelow(True)
    ax.grid()
    return tablesample(
        values={"threshold": threshold, "recall": recall, "precision": precision},
    )


# ---#
def randomized_search_cv(
    estimator,
    input_relation: (str, vDataFrame),
    X: list,
    y: str,
    metric: str = "auto",
    cv: int = 3,
    pos_label: (int, float, str) = None,
    cutoff: float = -1,
    nbins: int = 1000,
    lmax: int = 4,
    optimized_grid: int = 1,
    print_info: bool = True,
):
    """
---------------------------------------------------------------------------
Computes the K-Fold randomized search of an estimator.

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
metric: str, optional
    Metric used to do the model evaluation.
        auto: logloss for classification & rmse for regression.
    For Classification:
        accuracy    : Accuracy
        auc         : Area Under the Curve (ROC)
        bm          : Informedness = tpr + tnr - 1
        csi         : Critical Success Index = tp / (tp + fn + fp)
        f1          : F1 Score 
        logloss     : Log Loss
        mcc         : Matthews Correlation Coefficient 
        mk          : Markedness = ppv + npv - 1
        npv         : Negative Predictive Value = tn / (tn + fn)
        prc_auc     : Area Under the Curve (PRC)
        precision   : Precision = tp / (tp + fp)
        recall      : Recall = tp / (tp + fn)
        specificity : Specificity = tn / (tn + fp)
    For Regression:
        max    : Max Error
        mae    : Mean Absolute Error
        median : Median Absolute Error
        mse    : Mean Squared Error
        msle   : Mean Squared Log Error
        r2     : R squared coefficient
        r2a    : R2 adjusted
        rmse   : Root Mean Squared Error
        var    : Explained Variance 
cv: int, optional
    Number of folds.
pos_label: int/float/str, optional
    The main class to be considered as positive (classification only).
cutoff: float, optional
    The model cutoff (classification only).
nbins: int, optional
    Number of bins used to compute the different parameters categories.
lmax: int, optional
    Maximum length of each parameter list.
optimized_grid: int, optional
    If set to 0, the randomness is based on the input parameters.
    If set to 1, the randomness is limited to some parameters, the other
    ones are picked based on a default grid.
    If set to 2, there is no randomness and a default grid is returned.
print_info: bool
    If set to True, prints the model information at each step.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    param_grid = gen_params_grid(estimator, nbins, len(X), lmax, optimized_grid)
    return grid_search_cv(
        estimator,
        param_grid,
        input_relation,
        X,
        y,
        metric,
        cv,
        pos_label,
        cutoff,
        True,
        "no_print",
        print_info,
    )


# ---#
def roc_curve(
    y_true: str,
    y_score: str,
    input_relation: (str, vDataFrame),
    cursor=None,
    pos_label: (int, float, str) = 1,
    nbins: int = 30,
    auc_roc: bool = False,
    best_threshold: bool = False,
    cutoff_curve: bool = False,
    ax=None,
    **style_kwds,
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
cutoff_curve: bool, optional
    If set to True, the Cutoff curve will be drawn.
ax: Matplotlib axes object, optional
    The axes to plot on.
**style_kwds
    Any optional parameter to pass to the Matplotlib functions.

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
            ("cutoff_curve", cutoff_curve, [bool],),
        ]
    )
    cursor, conn, input_relation = check_cursor(cursor, input_relation)
    version(cursor=cursor, condition=[8, 0, 0])
    query = "SELECT decision_boundary, false_positive_rate, true_positive_rate FROM (SELECT ROC(obs, prob USING PARAMETERS num_bins = {}) OVER() FROM (SELECT (CASE WHEN {} = '{}' THEN 1 ELSE 0 END) AS obs, {}::float AS prob FROM {}) AS prediction_output) x"
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
    color1, color2 = color_dict(style_kwds, 0), color_dict(style_kwds, 1)
    if color1 == color2:
        color2 = gen_colors()[1]
    if cutoff_curve:
        ax.plot(
            threshold,
            [1 - item for item in false_positive],
            label="Specificity",
            **updated_dict({"color": gen_colors()[0]}, style_kwds),
        )
        ax.plot(
            threshold,
            true_positive,
            label="Sensitivity",
            **updated_dict({"color": gen_colors()[1]}, style_kwds),
        )
        ax.fill_between(
            threshold,
            [1 - item for item in false_positive],
            true_positive,
            facecolor="black",
            alpha=0.02,
        )
        ax.set_xlabel("Decision Boundary")
        ax.set_title("Cutoff Curve")
        ax.legend(loc="center left", bbox_to_anchor=[1, 0.5])
    else:
        ax.set_xlabel("False Positive Rate (1-Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.plot(
            false_positive,
            true_positive,
            **updated_dict({"color": gen_colors()[0]}, style_kwds),
        )
        ax.fill_between(
            false_positive, false_positive, true_positive, facecolor=color1, alpha=0.1
        )
        ax.fill_between([0, 1], [0, 0], [0, 1], facecolor=color2, alpha=0.1)
        ax.plot([0, 1], [0, 1], color=color2)
        ax.set_title("ROC Curve")
        ax.text(
            0.995,
            0,
            "AUC = " + str(round(auc, 4) * 100) + "%",
            verticalalignment="bottom",
            horizontalalignment="right",
            fontsize=11.5,
        )
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_axisbelow(True)
    ax.grid()
    return tablesample(
        values={
            "threshold": threshold,
            "false_positive": false_positive,
            "true_positive": true_positive,
        },
    )


# ---#
def validation_curve(
    estimator,
    param_name: str,
    param_range: list,
    input_relation: (str, vDataFrame),
    X: list,
    y: str,
    metric: str = "auto",
    cv: int = 3,
    pos_label: (int, float, str) = None,
    cutoff: float = -1,
    std_coeff: float = 1,
    ax=None,
    **style_kwds,
):
    """
---------------------------------------------------------------------------
Draws the Validation curve.

Parameters
----------
estimator: object
    Vertica estimator having a fit method and a DB cursor.
param_name: str
    Parameter name.
param_range: list
    Parameter Range.
input_relation: str/vDataFrame
    Relation to use to train the model.
X: list
    List of the predictor columns.
y: str
    Response Column.
metric: str, optional
    Metric used to do the model evaluation.
        auto: logloss for classification & rmse for regression.
    For Classification:
        accuracy    : Accuracy
        auc         : Area Under the Curve (ROC)
        bm          : Informedness = tpr + tnr - 1
        csi         : Critical Success Index = tp / (tp + fn + fp)
        f1          : F1 Score 
        logloss     : Log Loss
        mcc         : Matthews Correlation Coefficient 
        mk          : Markedness = ppv + npv - 1
        npv         : Negative Predictive Value = tn / (tn + fn)
        prc_auc     : Area Under the Curve (PRC)
        precision   : Precision = tp / (tp + fp)
        recall      : Recall = tp / (tp + fn)
        specificity : Specificity = tn / (tn + fp)
    For Regression:
        max    : Max Error
        mae    : Mean Absolute Error
        median : Median Absolute Error
        mse    : Mean Squared Error
        msle   : Mean Squared Log Error
        r2     : R squared coefficient
        r2a    : R2 adjusted
        rmse   : Root Mean Squared Error
        var    : Explained Variance 
cv: int, optional
    Number of folds.
pos_label: int/float/str, optional
    The main class to be considered as positive (classification only).
cutoff: float, optional
    The model cutoff (classification only).
std_coeff: float, optional
    Value of the standard deviation coefficient used to compute the area plot 
    around each score.
ax: Matplotlib axes object, optional
    The axes to plot on.
**style_kwds
    Any optional parameter to pass to the Matplotlib functions.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    if not (isinstance(param_range, Iterable)) or isinstance(param_range, str):
        param_range = [param_range]
    from verticapy.plot import range_curve

    gs_result = grid_search_cv(
        estimator,
        {param_name: param_range},
        input_relation,
        X,
        y,
        metric,
        cv,
        pos_label,
        cutoff,
        True,
        False,
    )
    gs_result_final = [
        (
            gs_result["parameters"][i][param_name],
            gs_result["avg_score"][i],
            gs_result["avg_train_score"][i],
            gs_result["score_std"][i],
            gs_result["score_train_std"][i],
        )
        for i in range(len(param_range))
    ]
    gs_result_final.sort(key=lambda tup: tup[0])
    X = [elem[0] for elem in gs_result_final]
    Y = [
        [
            [elem[2] - std_coeff * elem[4] for elem in gs_result_final],
            [elem[2] for elem in gs_result_final],
            [elem[2] + std_coeff * elem[4] for elem in gs_result_final],
        ],
        [
            [elem[1] - std_coeff * elem[3] for elem in gs_result_final],
            [elem[1] for elem in gs_result_final],
            [elem[1] + std_coeff * elem[3] for elem in gs_result_final],
        ],
    ]
    result = tablesample(
        {
            param_name: X,
            "training_score_lower": Y[0][0],
            "training_score": Y[0][1],
            "training_score_upper": Y[0][2],
            "test_score_lower": Y[1][0],
            "test_score": Y[1][1],
            "test_score_upper": Y[1][2],
        }
    )
    range_curve(
        X, Y, param_name, metric, ax, ["train", "test"], **style_kwds,
    )
    return result
