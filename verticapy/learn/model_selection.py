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
from verticapy.learn.tools import does_model_exist
from verticapy.learn.mlplot import plot_bubble_ml, plot_stepwise_ml, plot_importance

# Other Python Modules
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---#
def bayesian_search_cv(
    estimator,
    input_relation: (str, vDataFrame),
    X: list,
    y: str,
    metric: str = "auto",
    cv: int = 3,
    pos_label: (int, float, str) = None,
    cutoff: float = -1,
    param_grid: (dict, list) = {},
    random_nbins: int = 16,
    bayesian_nbins: int = None,
    random_grid: bool = False,
    lmax: int = 15,
    nrows: int = 100000,
    k_tops: int = 10,
    RFmodel_params: dict = {},
    print_info: bool = True,
    **kwargs,
):
    """
---------------------------------------------------------------------------
Computes the k-fold bayesian search of an estimator using a random
forest model to estimate a probable optimal set of parameters.

Parameters
----------
estimator: object
    Vertica estimator with a fit method and a database cursor.
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
        max    : Max error
        mae    : Mean absolute error
        median : Median absolute error
        mse    : Mean squared error
        msle   : Mean squared log error
        r2     : R-squared coefficient
        r2a    : R2 adjusted
        rmse   : Root-mean-squared error
        var    : Explained variance
cv: int, optional
    Number of folds.
pos_label: int/float/str, optional
    The main class to be considered as positive (classification only).
cutoff: float, optional
    The model cutoff (classification only).
param_grid: dict/list, optional
    Dictionary of the parameters to test. It can also be a list of the
    different combinations. If empty, a parameter grid will be generated.
random_nbins: int, optional
    Number of bins used to compute the different parameters categories
    in the random parameters generation.
bayesian_nbins: int, optional
    Number of bins used to compute the different parameters categories
    in the bayesian table generation.
random_grid: bool, optional
    If True, the rows used to find the optimal function will be
    used randomnly. Otherwise, they will be regularly spaced. 
lmax: int, optional
    Maximum length of each parameter list.
nrows: int, optional
    Number of rows to use when performing the bayesian search.
k_tops: int, optional
    When performing the bayesian search, the final stage will be to retrain the top
    possible combinations. 'k_tops' represents the number of models to train at
    this stage to find the most efficient model.
RFmodel_params: dict, optional
    Dictionary of the random forest model parameters used to estimate a probable 
    optimal set of parameters.
print_info: bool, optional
    If True, prints the model information at each step.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    if print_info:
        print(f"\033[1m\033[4mStarting Bayesian Search\033[0m\033[0m\n")
        print(f"\033[1m\033[4mStep 1 - Computing Random Models using Grid Search\033[0m\033[0m\n")
    if not(param_grid):
        param_grid = gen_params_grid(estimator, random_nbins, len(X), lmax, 0)
    param_gs = grid_search_cv(
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
        final_print="no_print",
    )
    if "enet" not in kwargs:
        params = []
        for param_grid in param_gs["parameters"]:
            params += [elem for elem in param_grid]
        all_params = list(dict.fromkeys(params))
    else:
        all_params = ["C", "l1_ratio",]
    if not(bayesian_nbins):
        bayesian_nbins = max(int(np.exp(np.log(nrows) / len(all_params))), 1)
    result = {}
    for elem in all_params:
        result[elem] = []
    for param_grid in param_gs["parameters"]:
        for item in all_params:
            if item in param_grid:
                result[item] += [param_grid[item]]
            else:
                result[item] += [None]
    result["score"] = param_gs["avg_score"]
    result = tablesample(result).to_sql()
    if isinstance(input_relation, str):
        schema, relation = schema_relation(input_relation)
    else:
        if input_relation._VERTICAPY_VARIABLES_["schema_writing"]:
            schema = input_relation._VERTICAPY_VARIABLES_["schema_writing"]
        else:
            schema, relation = schema_relation(input_relation.__genSQL__())
    relation = "{}.verticapy_temp_table_bayesian_{}".format(schema, get_session(estimator.cursor))
    model_name = "{}.verticapy_temp_rf_{}".format(schema, get_session(estimator.cursor))
    estimator.cursor.execute("DROP TABLE IF EXISTS {}".format(relation))
    estimator.cursor.execute("CREATE TABLE {} AS {}".format(relation, result))
    if print_info:
        print(f"\033[1m\033[4mStep 2 - Fitting the RF model with the hyperparameters data\033[0m\033[0m\n")
    if verticapy.options["tqdm"] and print_info:
        from tqdm.auto import tqdm
        loop = tqdm(range(1))
    else:
        loop = range(1)
    for j in loop:
        if "enet" not in kwargs:
            model_grid = gen_params_grid(estimator, nbins=bayesian_nbins, max_nfeatures=len(all_params), optimized_grid=-666)
        else:
            model_grid = {"C": {"type": float, "range": [0.0, 10], "nbins": bayesian_nbins}, "l1_ratio": {"type": float, "range": [0.0, 1.0], "nbins": bayesian_nbins},}
        all_params = list(dict.fromkeys(model_grid))
        from verticapy.learn.ensemble import RandomForestRegressor
        hyper_param_estimator = RandomForestRegressor(name=estimator.name, cursor=estimator.cursor, **RFmodel_params,)
        hyper_param_estimator.fit(relation, all_params, "score")
        from verticapy.datasets import gen_meshgrid, gen_dataset
        if random_grid:
            vdf = gen_dataset(model_grid, estimator.cursor, nrows=nrows,)
        else:
            vdf = gen_meshgrid(model_grid, estimator.cursor,)
        estimator.cursor.execute("DROP TABLE IF EXISTS {}".format(relation))
        vdf.to_db(relation, relation_type="table", inplace=True)
        vdf = hyper_param_estimator.predict(vdf, name="score")
        reverse = reverse_score(metric)
        vdf.sort({"score": "desc" if reverse else "asc"})
        result = vdf.head(limit = k_tops)
        new_param_grid = []
        for i in range(k_tops):
            param_tmp_grid = {}
            for elem in result.values:
                if elem != "score":
                    param_tmp_grid[elem] = result[elem][i]
            new_param_grid += [param_tmp_grid]
    if print_info:
        print(f"\033[1m\033[4mStep 3 - Computing Most Probable Good Models using Grid Search\033[0m\033[0m\n")
    result = grid_search_cv(
        estimator,
        new_param_grid,
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
        final_print="no_print",
    )
    for elem in result.values:
        result.values[elem] += param_gs[elem]
    data = []
    keys = [elem for elem in result.values]
    for i in range(len(result[keys[0]])):
        data += [tuple([result[elem][i] for elem in result.values])]
    data.sort(key=lambda tup: tup[1], reverse=reverse)
    for idx, elem in enumerate(result.values):
        result.values[elem] = [item[idx] for item in data]
    hyper_param_estimator.drop()
    if print_info:
        print("\033[1mBayesian Search Selected Model\033[0m")
        print(f"Parameters: {result['parameters'][0]}; \033[91mTest_score: {result['avg_score'][0]}\033[0m; \033[92mTrain_score: {result['avg_train_score'][0]}\033[0m; \033[94mTime: {result['avg_time'][0]}\033[0m;")
    estimator.cursor.execute("DROP TABLE IF EXISTS {}".format(relation))

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
    **kwargs,
):
    """
---------------------------------------------------------------------------
Finds the k-means k based on a score.

Parameters
----------
input_relation: str/vDataFrame
    Relation to use to train the model.
X: list, optional
	List of the predictor columns. If empty, all numerical columns will
    be used.
cursor: DBcursor, optional
	Vertica database cursor.
n_cluster: tuple/list, optional
	Tuple representing the number of clusters to start and end with.
    This can also be customized list with various k values to test.
init: str/list, optional
	The method to use to find the initial cluster centers.
		kmeanspp : Use the k-means++ method to initialize the centers.
        random   : Randomly subsamples the data to find initial centers.
	It can be also a list with the initial cluster centers to use.
max_iter: int, optional
	The maximum number of iterations for the algorithm.
tol: float, optional
	Determines whether the algorithm has converged. The algorithm is considered 
	converged after no center has moved more than a distance of 'tol' from the 
	previous iteration.
elbow_score_stop: float, optional
	Stops searching for parameters when the specified elbow score is reached.

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
    if verticapy.options["tqdm"] and ("tqdm" not in kwargs or ("tqdm" in kwargs and kwargs["tqdm"])):
        from tqdm.auto import tqdm

        loop = tqdm(L)
    else:
        loop = L
    for i in loop:
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
    **kwargs,
):
    """
---------------------------------------------------------------------------
Computes the K-Fold cross validation of an estimator.

Parameters
----------
estimator: object
	Vertica estimator with a fit method and a database cursor.
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
        aic    : Akaike’s information criterion
        bic    : Bayesian information criterion
        max    : Max error
        mae    : Mean absolute error
        median : Median absolute error
        mse    : Mean squared error
        msle   : Mean squared log error
        r2     : R-squared coefficient
        r2a    : R2 adjusted
        rmse   : Root-mean-squared error
        var    : Explained variance
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
    if verticapy.options["tqdm"] and ("tqdm" not in kwargs or ("tqdm" in kwargs and kwargs["tqdm"])):
        from tqdm.auto import tqdm

        loop = tqdm(range(cv))
    else:
        loop = range(cv)
    for i in loop:
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
Draws an elbow curve.

Parameters
----------
input_relation: str/vDataFrame
    Relation to use to train the model.
X: list, optional
    List of the predictor columns. If empty all the numerical vcolumns will
    be used.
cursor: DBcursor, optional
    Vertica database cursor.
n_cluster: tuple/list, optional
    Tuple representing the number of cluster to start with and to end with.
    It can also be customized list with the different K to test.
init: str/list, optional
    The method to use to find the initial cluster centers.
        kmeanspp : Use the k-means++ method to initialize the centers.
        random   : Randomly subsamples the data to find initial centers.
    Alternatively, you can specify a list with the initial custer centers.
max_iter: int, optional
    The maximum number of iterations for the algorithm.
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
    if verticapy.options["tqdm"]:
        from tqdm.auto import tqdm

        loop = tqdm(L)
    else:
        loop = L
    for i in loop:
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
def enet_search_cv(
    input_relation: (str, vDataFrame),
    X: list,
    y: str,
    metric: str = "auto",
    cv: int = 3,
    estimator_type: str = "auto",
    cutoff: float = -1,
    cursor=None,
    print_info: bool = True,
    **kwargs,
):
    """
---------------------------------------------------------------------------
Computes the k-fold grid search using multiple ENet models.

Parameters
----------
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
        max    : Max error
        mae    : Mean absolute error
        median : Median absolute error
        mse    : Mean squared error
        msle   : Mean squared log error
        r2     : R-squared coefficient
        r2a    : R2 adjusted
        rmse   : Root-mean-squared error
        var    : Explained variance
cv: int, optional
    Number of folds.
estimator_type: str, optional
    Estimator Type.
        auto : detects if it is a Logit Model or ENet.
        logit: Logistic Regression
        enet : ElasticNet
cutoff: float, optional
    The model cutoff (logit only).
cursor: DBcursor, optional
    Vertica database cursor.
print_info: bool, optional
    If set to True, prints the model information at each step.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    check_types([("estimator_type", estimator_type, ["logit", "enet", "auto",]),])
    cursor, conn, input_relation = check_cursor(cursor, input_relation)
    param_grid = parameter_grid({"solver": ["cgd",], 
                                 "penalty": ["enet",],
                                 "C": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 5.0, 10.0, 50.0, 100.0] if "small" not in kwargs else [1e-1, 1.0, 10.0,],
                                 "l1_ratio": [0.1 * i for i in range(0, 10)] if "small" not in kwargs else [0.1, 0.5, 0.9,]})

    from verticapy.learn.linear_model import LogisticRegression, ElasticNet

    if estimator_type == "auto":
        if not(isinstance(input_relation, vDataFrame)):
            vdf = vdf_from_relation(input_relation, cursor=cursor)
        else:
            vdf = input_relation
        if sorted(vdf[y].distinct()) == [0, 1]:
            estimator_type = "logit"
        else:
            estimator_type = "enet"
    if estimator_type == "logit":
        estimator = LogisticRegression("verticapy_enet_search_{}".format(get_session(cursor)), cursor=cursor)
    else:
        estimator = ElasticNet("verticapy_enet_search_{}".format(get_session(cursor)), cursor=cursor)
    result = bayesian_search_cv(
        estimator,
        input_relation,
        X,
        y,
        metric,
        cv,
        None,
        cutoff,
        param_grid,
        random_grid=False,
        bayesian_nbins=1000,
        print_info=print_info,
        enet=True,
    )
    if conn:
        conn.close()
    return result


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
    Vertica estimator with a fit method and a database cursor.
nbins: int, optional
    Number of bins used to discretize numberical features.
max_nfeatures: int, optional
    Maximum number of features used to compute Random Forest, PCA...
lmax: int, optional
    Maximum length of the parameter grid.
optimized_grid: int, optional
    If set to 0, the randomness is based on the input parameters.
    If set to 1, the randomness is limited to some parameters while others
    are picked based on a default grid.
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
        elif optimized_grid == 2:
            params_grid = {"max_features": ["auto", "max"],
                           "max_leaf_nodes": [32, 64, 128, 1000,],
                           "max_depth": [4, 5, 6,],
                           "min_samples_leaf": [1, 2,],
                           "min_info_gain": [0.0,],
                           "nbins": [32,],}
            if isinstance(RandomForestRegressor, RandomForestClassifier,):
                params_grid["sample"] = [0.7,]
                params_grid["n_estimators"] = [20,]
        elif optimized_grid == -666:
            result = {"max_features": {"type": int, "range": [1, len(all_params),], "nbins": nbins,},
                      "max_leaf_nodes": {"type": int, "range": [32, 1e9,], "nbins": nbins,},
                      "max_depth": {"type": int, "range": [2, 30,], "nbins": nbins,},
                      "min_samples_leaf": {"type": int, "range": [1, 15,], "nbins": nbins,},
                      "min_samples_leaf": {"type": float, "range": [0.0, 0.1,], "nbins": nbins,},
                      "nbins": {"type": int, "range": [10, 1000,], "nbins": nbins,},}
            if isinstance(RandomForestRegressor, RandomForestClassifier,):
                result["sample"] = {"type": float, "range": [0.1, 1.0,], "nbins": nbins,}
                result["n_estimators"] = {"type": int, "range": [1, 100,], "nbins": nbins,}
            return result
    elif isinstance(estimator, (LinearSVC, LinearSVR,)):
        if optimized_grid == 0:
            params_grid = {"tol": [1e-4, 1e-6, 1e-8],
                           "C": [elem / 1000 for elem in range(1, 5000, math.ceil(5000 / nbins))],
                           "fit_intercept": [False, True],
                           "intercept_mode": ["regularized", "unregularized"],
                           "max_iter": [100, 500, 1000],}
        elif optimized_grid == 1:
            params_grid = {"tol": [1e-6],
                           "C": [1e-1, 0.0, 1.0, 10.0,],
                           "fit_intercept": [True],
                           "intercept_mode": ["regularized", "unregularized"],
                           "max_iter": [100],}
        elif optimized_grid == 2:
            params_grid = {"tol": [1e-6],
                           "C": [0.0, 1.0,],
                           "fit_intercept": [True],
                           "intercept_mode": ["regularized", "unregularized"],
                           "max_iter": [100],}
        elif optimized_grid == -666:
            return {"tol": {"type": float, "range": [1e-8, 1e-2,], "nbins": nbins,},
                    "C": {"type": float, "range": [0.0, 1000.0,], "nbins": nbins,},
                    "fit_intercept": {"type": bool,},
                    "intercept_mode": {"type": str, "values": ["regularized", "unregularized"]},
                    "max_iter": {"type": int, "range": [10, 1000,], "nbins": nbins,},}
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
        elif optimized_grid == 2:
            params_grid = {"nbins": [32,],
                           "max_depth": [3, 4, 5,],
                           "weight_reg": [0.0, 0.25,],
                           "min_split_loss": [0.0,],
                           "learning_rate": [0.05, 0.1, 1.0,],
                           #"sample": [0.5, 0.6, 0.7,],
                           "tol": [1e-8],
                           "max_ntree": [20,]}
        elif optimized_grid == -666:
            return {"nbins": {"type": int, "range": [2, 100,], "nbins": nbins,},
                    "max_depth": {"type": int, "range": [1, 20,], "nbins": nbins,},
                    "weight_reg": {"type": float, "range": [0.0, 1.0,], "nbins": nbins,},
                    "min_split_loss": {"type": float, "values": [0.0, 0.25,], "nbins": nbins,},
                    "learning_rate": {"type": float, "range": [0.0, 1.0,], "nbins": nbins,},
                    "sample": {"type": float, "range": [0.0, 1.0,], "nbins": nbins,},
                    "tol": {"type": float, "range": [1e-8, 1e-2,], "nbins": nbins,},
                    "max_ntree": {"type": int, "range": [1, 20,], "nbins": nbins,},}
    elif isinstance(estimator, NaiveBayes):
        if optimized_grid == 0:
            params_grid = {"alpha": [elem / 1000 for elem in range(1, 1000, math.ceil(1000 / nbins))]}
        elif optimized_grid == 1:
            params_grid = {"alpha": [0.01, 0.1, 1.0,  5.0, 10.0,]}
        elif optimized_grid == 2:
            params_grid = {"alpha": [0.01, 1.0,  10.0,]}
        elif optimized_grid == -666:
            return {"alpha": {"type": float, "range": [0.00001, 1000.0,], "nbins": nbins,},}
    elif isinstance(estimator, (PCA, SVD)):
        if optimized_grid == 0:
            params_grid = {"max_features": list(range(1, max_nfeatures, math.ceil(max_nfeatures / nbins))),}
        if isinstance(estimator, (PCA,)):
            params_grid["scale"] = [False, True]
        if optimized_grid == -666:
            return {"scale": {"type": bool,}, "max_features": {"type": int, "range": [1, max_nfeatures,], "nbins": nbins,},}
    elif isinstance(estimator, (Normalizer,)):
        params_grid = {"method": ["minmax", "robust_zscore", "zscore"]}
        if optimized_grid == -666:
            return {"method": {"type": str, "values": ["minmax", "robust_zscore", "zscore"]},}
    elif isinstance(estimator, (KNeighborsRegressor, KNeighborsClassifier, LocalOutlierFactor, NearestCentroid,)):
        if optimized_grid == 0:
            params_grid = {"p": [1, 2] + list(range(3, 100, math.ceil(100 / (nbins - 2)))),}
            if isinstance(estimator, (KNeighborsRegressor, KNeighborsClassifier, LocalOutlierFactor,)):
                params_grid["n_neighbors"] =  list(range(1, 100, math.ceil(100 / (nbins))))
        elif optimized_grid == 1:
            params_grid = {"p": [1, 2, 3, 4],}
            if isinstance(estimator, (KNeighborsRegressor, KNeighborsClassifier, LocalOutlierFactor,)):
                params_grid["n_neighbors"] =  [1, 2, 3, 4, 5, 10, 20, 100]
        elif optimized_grid == 2:
            params_grid = {"p": [1, 2,],}
            if isinstance(estimator, (KNeighborsRegressor, KNeighborsClassifier, LocalOutlierFactor,)):
                params_grid["n_neighbors"] =  [5, 10,]
        elif optimized_grid == -666:
            return {"p": {"type": int, "range": [1, 10,], "nbins": nbins,},
                    "n_neighbors": {"type": int, "range": [1, 100,], "nbins": nbins,},}
    elif isinstance(estimator, (DBSCAN,)):
        if optimized_grid == 0:
            params_grid = {"p": [1, 2] + list(range(3, 100, math.ceil(100 / (nbins - 2)))),
                           "eps": [elem / 1000 for elem in range(1, 1000, math.ceil(1000 / nbins))],
                           "min_samples": list(range(1, 1000, math.ceil(1000 / nbins))),}
        elif optimized_grid == 1:
            params_grid = {"p": [1, 2, 3, 4],
                           "min_samples": [1, 2, 3, 4, 5, 10, 100],}
        elif optimized_grid == 2:
            params_grid = {"p": [1, 2,],
                           "min_samples": [5, 10,],}
        elif optimized_grid == -666:
            return {"p": {"type": int, "range": [1, 10,], "nbins": nbins,},
                    "min_samples": {"type": int, "range": [1, 100,], "nbins": nbins,},}
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
            params_grid = {"tol": [1e-6],
                           "max_iter": [100],}
            if isinstance(estimator, LogisticRegression):
                params_grid["penalty"] = ["none", "l1", "l2", "enet",]
            if isinstance(estimator, LinearRegression):
                params_grid["solver"] = ["newton", "bfgs",]
            elif isinstance(estimator, (Lasso, LogisticRegression, ElasticNet)):
                params_grid["solver"] = ["newton", "bfgs", "cgd",]
            if isinstance(estimator, (Lasso, Ridge, ElasticNet, LogisticRegression)):
                params_grid["C"] = [1e-1, 0.0, 1.0, 10.0,]
            if isinstance(estimator, (LogisticRegression,)):
                params_grid["penalty"] = ["none", "l1", "l2", "enet"]
            if isinstance(estimator, (LogisticRegression, ElasticNet)):
                params_grid["l1_ratio"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        elif optimized_grid == 2:
            params_grid = {"tol": [1e-6],
                           "max_iter": [100],}
            if isinstance(estimator, LogisticRegression):
                params_grid["penalty"] = ["none", "l1", "l2", "enet",]
            if isinstance(estimator, LinearRegression):
                params_grid["solver"] = ["newton", "bfgs",]
            elif isinstance(estimator, (Lasso, LogisticRegression, ElasticNet)):
                params_grid["solver"] = ["bfgs", "cgd",]
            if isinstance(estimator, (Lasso, Ridge, ElasticNet, LogisticRegression)):
                params_grid["C"] = [1.0,]
            if isinstance(estimator, (LogisticRegression,)):
                params_grid["penalty"] = ["none", "l1", "l2", "enet"]
            if isinstance(estimator, (LogisticRegression, ElasticNet)):
                params_grid["l1_ratio"] = [0.5,]
        elif optimized_grid == -666:
            result = {"tol": {"type": float, "range": [1e-8, 1e-2,], "nbins": nbins,},
                      "max_iter": {"type": int, "range": [1, 1000,], "nbins": nbins,},}
            if isinstance(estimator, LogisticRegression):
                result["penalty"] = {"type": str, "values": ["none", "l1", "l2", "enet",]}
            if isinstance(estimator, LinearRegression):
                result["solver"] = {"type": str, "values": ["newton", "bfgs",]}
            elif isinstance(estimator, (Lasso, LogisticRegression, ElasticNet)):
                result["solver"] = {"type": str, "values": ["bfgs", "cgd",]}
            if isinstance(estimator, (Lasso, Ridge, ElasticNet, LogisticRegression)):
                result["C"] = {"type": float, "range": [0.0, 1000.0,], "nbins": nbins,}
            if isinstance(estimator, (LogisticRegression,)):
                result["penalty"] = {"type": str, "values": ["none", "l1", "l2", "enet",]}
            if isinstance(estimator, (LogisticRegression, ElasticNet)):
                result["l1_ratio"] = {"type": float, "range": [0.0, 1.0,], "nbins": nbins,}
            return result
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
        elif optimized_grid == 2:
            params_grid = {"n_cluster": [2, 3, 4, 5, 10, 20, 100,],
                           "init": ["kmeanspp",],
                           "max_iter": [1000],
                           "tol": [1e-8],}
        elif optimized_grid == -666:
            return {"tol": {"type": float, "range": [1e-2, 1e-8,], "nbins": nbins,},
                    "max_iter": {"type": int, "range": [1, 1000,], "nbins": nbins,},
                    "n_cluster": {"type": int, "range": [1, 10000,], "nbins": nbins,},
                    "init": {"type": str, "values": ["kmeanspp", "random"],},}
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
        elif optimized_grid == 2:
            params_grid = {"n_cluster": [2, 3, 4, 5, 10, 20, 100,],
                           "bisection_iterations": [1, 2, 3,],
                           "split_method": ["sum_squares",],
                           "min_divisible_cluster_size": [2, 3, 4,],
                           "init": ["kmeanspp",],
                           "max_iter": [1000],
                           "tol": [1e-8],}
        elif optimized_grid == -666:
            return {"tol": {"type": float, "range": [1e-8, 1e-2,], "nbins": nbins,},
                    "max_iter": {"type": int, "range": [1, 1000,], "nbins": nbins,},
                    "bisection_iterations": {"type": int, "range": [1, 1000,], "nbins": nbins,},
                    "split_method": {"type": str, "values": ["sum_squares",],},
                    "n_cluster": {"type": int, "range": [1, 10000,], "nbins": nbins,},
                    "init": {"type": str, "values": ["kmeanspp", "pseudo"],},}
    params_grid = parameter_grid(params_grid)
    final_param_grid = []
    for param in params_grid:
        if "C" in param and param["C"] == 0:
            del param["C"]
            if "l1_ratio" in param:
                del param["l1_ratio"]
            if "penalty" in param:
                param["penalty"] = "none"
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
    skip_error: bool = True,
    print_info: bool = True,
    **kwargs,
):
    """
---------------------------------------------------------------------------
Computes the k-fold grid search of an estimator.

Parameters
----------
estimator: object
    Vertica estimator with a fit method and a database cursor.
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
        max    : Max error
        mae    : Mean absolute error
        median : Median absolute error
        mse    : Mean squared error
        msle   : Mean squared log error
        r2     : R-squared coefficient
        r2a    : R2 adjusted
        rmse   : Root-mean-squared error
        var    : Explained variance
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
print_info: bool, optional
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
    if verticapy.options["tqdm"] and ("tqdm" not in kwargs or ("tqdm" in kwargs and kwargs["tqdm"])) and print_info:
        from tqdm.auto import tqdm

        loop = tqdm(all_configuration)
    else:
        loop = all_configuration
    for config in loop:
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
                tqdm=False,
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
    reverse = reverse_score(metric)
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
        if print_info and ("final_print" not in kwargs or kwargs["final_print"] != "no_print"):
            print("\033[1mGrid Search Selected Model\033[0m")
            print(f"{str(estimator.__class__).split('.')[-1][:-2]}; Parameters: {result['parameters'][0]}; \033[91mTest_score: {result['avg_score'][0]}\033[0m; \033[92mTrain_score: {result['avg_train_score'][0]}\033[0m; \033[94mTime: {result['avg_time'][0]}\033[0m;")
    else:
        result = tablesample(
            {
                "parameters": [elem[0] for elem in data],
                "avg_score": [elem[1] for elem in data],
                "avg_time": [elem[2] for elem in data],
                "score_std": [elem[3] for elem in data],
            }
        )
        if print_info and ("final_print" not in kwargs or kwargs["final_print"] != "no_print"):
            print("\033[1mGrid Search Selected Model\033[0m")
            print(f"{str(estimator.__class__).split('.')[-1][:-2]}; Parameters: {result['parameters'][0]}; \033[91mTest_score: {result['avg_score'][0]}\033[0m; \033[94mTime: {result['avg_time'][0]}\033[0m;")
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
Draws the learning curve.

Parameters
----------
estimator: object
    Vertica estimator with a fit method and a database cursor.
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
        max    : Max error
        mae    : Mean absolute error
        median : Median absolute error
        mse    : Mean squared error
        msle   : Mean squared log error
        r2     : R-squared coefficient
        r2a    : R2 adjusted
        rmse   : Root-mean-squared error
        var    : Explained variance
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
    if verticapy.options["tqdm"]:
        from tqdm.auto import tqdm

        loop = tqdm(sizes)
    else:
        loop = sizes
    for s in loop:
        relation = input_relation.sample(x=s)
        lc_result = cross_validate(
            estimator, relation, X, y, metric, cv, pos_label, cutoff, True, True, tqdm=False,
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
    Vertica database cursor.
pos_label: int/float/str, optional
    To compute the Lift Chart, one of the response column classes must be the
    positive one. The parameter 'pos_label' represents this class.
nbins: int, optional
    The number of bins.
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
Generates the list of the different combinations of input parameters.

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
    Vertica database cursor.
pos_label: int/float/str, optional
    To compute the PRC Curve, one of the response column classes must be the
    positive one. The parameter 'pos_label' represents this class.
nbins: int, optional
    The number of bins.
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
def randomized_features_search_cv(
    estimator,
    input_relation: (str, vDataFrame),
    X: list,
    y: str,
    metric: str = "auto",
    cv: int = 3,
    pos_label: (int, float, str) = None,
    cutoff: float = -1,
    training_score: bool = True,
    comb_limit: int = 100,
    skip_error: bool = True,
    print_info: bool = True,
    **kwargs,
):
    """
---------------------------------------------------------------------------
Computes the k-fold grid search of an estimator using different features
combinations. It can be used to find the parameters which will optimize
the model.

Parameters
----------
estimator: object
    Vertica estimator with a fit method and a database cursor.
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
        max    : Max error
        mae    : Mean absolute error
        median : Median absolute error
        mse    : Mean squared error
        msle   : Mean squared log error
        r2     : R-squared coefficient
        r2a    : R2 adjusted
        rmse   : Root-mean-squared error
        var    : Explained variance
cv: int, optional
    Number of folds.
pos_label: int/float/str, optional
    The main class to be considered as positive (classification only).
cutoff: float, optional
    The model cutoff (classification only).
training_score: bool, optional
    If set to True, the training score will be computed with the validation score.
comb_limit: int, optional
    Maximum number of features combinations used to train the model.
skip_error: bool, optional
    If set to True and an error occurs, it will be displayed and not raised.
print_info: bool, optional
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
            ("training_score", training_score, [bool]),
            ("skip_error", skip_error, [bool, str,]),
            ("print_info", print_info, [bool,]),
            ("comb_limit", comb_limit, [int,]),
        ]
    )
    if category_from_model_type(estimator.type)[0] == "regressor" and metric == "auto":
        metric = "rmse"
    elif metric == "auto":
        metric = "logloss"
    if len(X) < 20:
        all_configuration = all_comb(X)
        if len(all_configuration) > comb_limit and comb_limit > 0:
            all_configuration = random.sample(all_configuration, comb_limit)
    else:
        all_configuration = []
        for k in range(max(comb_limit, 1)):
            config = sorted(random.sample(X, random.randint(1, len(X))))
            if config not in all_configuration:
                all_configuration += [config]
    if verticapy.options["tqdm"] and ("tqdm" not in kwargs or ("tqdm" in kwargs and kwargs["tqdm"])) and print_info:
        from tqdm.auto import tqdm

        loop = tqdm(all_configuration)
    else:
        loop = all_configuration
    data = []
    for config in loop:
        if config:
            config = list(config)
            try:
                current_cv = cross_validate(
                    estimator,
                    input_relation,
                    config,
                    y,
                    metric,
                    cv,
                    pos_label,
                    cutoff,
                    True,
                    training_score,
                    tqdm=False,
                )
                if training_score:
                    keys = [elem for elem in current_cv[0].values]
                    data += [
                        (
                            config,
                            current_cv[0][keys[1]][cv],
                            current_cv[1][keys[1]][cv],
                            current_cv[0][keys[2]][cv],
                            current_cv[0][keys[1]][cv + 1],
                            current_cv[1][keys[1]][cv + 1],
                        )
                    ]
                    if print_info:
                        print(f"Model: {str(estimator.__class__).split('.')[-1][:-2]}; Features: {config}; \033[91mTest_score: {current_cv[0][keys[1]][cv]}\033[0m; \033[92mTrain_score: {current_cv[1][keys[1]][cv]}\033[0m; \033[94mTime: {current_cv[0][keys[2]][cv]}\033[0m;")
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
                        print(f"Model: {str(estimator.__class__).split('.')[-1][:-2]}; Features: {config}; \033[91mTest_score: {current_cv[keys[1]][cv]}\033[0m; \033[94mTime:{current_cv[keys[2]][cv]}\033[0m;")
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
    reverse = reverse_score(metric)
    data.sort(key=lambda tup: tup[1], reverse=reverse)
    if training_score:
        result = tablesample(
            {
                "features": [elem[0] for elem in data],
                "avg_score": [elem[1] for elem in data],
                "avg_train_score": [elem[2] for elem in data],
                "avg_time": [elem[3] for elem in data],
                "score_std": [elem[4] for elem in data],
                "score_train_std": [elem[5] for elem in data],
            }
        )
        if print_info and ("final_print" not in kwargs or kwargs["final_print"] != "no_print"):
            print("\033[1mRandomized Features Search Selected Model\033[0m")
            print(f"{str(estimator.__class__).split('.')[-1][:-2]}; Features: {result['features'][0]}; \033[91mTest_score: {result['avg_score'][0]}\033[0m; \033[92mTrain_score: {result['avg_train_score'][0]}\033[0m; \033[94mTime: {result['avg_time'][0]}\033[0m;")
    else:
        result = tablesample(
            {
                "features": [elem[0] for elem in data],
                "avg_score": [elem[1] for elem in data],
                "avg_time": [elem[2] for elem in data],
                "score_std": [elem[3] for elem in data],
            }
        )
        if print_info and ("final_print" not in kwargs or kwargs["final_print"] != "no_print"):
            print("\033[1mRandomized Features Search Selected Model\033[0m")
            print(f"{str(estimator.__class__).split('.')[-1][:-2]}; Features: {result['features'][0]}; \033[91mTest_score: {result['avg_score'][0]}\033[0m; \033[94mTime: {result['avg_time'][0]}\033[0m;")
    return result


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
    Vertica estimator with a fit method and a database cursor.
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
        max    : Max error
        mae    : Mean absolute error
        median : Median absolute error
        mse    : Mean squared error
        msle   : Mean squared log error
        r2     : R-squared coefficient
        r2a    : R2 adjusted
        rmse   : Root-mean-squared error
        var    : Explained variance
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
    If set to 1, the randomness is limited to some parameters while others
    are picked based on a default grid.
    If set to 2, there is no randomness and a default grid is returned.
print_info: bool, optional
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
    Vertica database cursor.
pos_label: int/float/str, optional
    To compute the PRC Curve, one of the response column classes must be the
    positive one. The parameter 'pos_label' represents this class.
nbins: int, optional
    The number of bins.
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
def stepwise(
    estimator,
    input_relation: (str, vDataFrame),
    X: list,
    y: str,
    criterion: str = "bic",
    direction: str = "backward",
    max_steps: int = 100,
    criterion_threshold: int = 3,
    drop_final_estimator: bool = True,
    x_order: str = "pearson",
    print_info: bool = True,
    show: bool = True,
    ax=None,
    **style_kwds,
):
    """
---------------------------------------------------------------------------
Uses the Stepwise algorithm to find the most suitable number of features
when fitting the estimator.

Parameters
----------
estimator: object
    Vertica estimator with a fit method and a database cursor.
input_relation: str/vDataFrame
    Relation to use to train the model.
X: list
    List of the predictor columns.
y: str
    Response Column.
criterion: str, optional
    Criterion used to evaluate the model.
        aic : Akaike’s Information Criterion
        bic : Bayesian Information Criterion
direction: str, optional
    How to start the stepwise search. Can be done 'backward' or 'forward'.
max_steps: int, optional
    The maximum number of steps to be considered.
criterion_threshold: int, optional
    Threshold used when comparing the models criterions. If the difference
    is lesser than the threshold then the current 'best' model is changed.
drop_final_estimator: bool, optional
    If set to True, the final estimator will be dropped.
x_order: str, optional
    How to preprocess X before using the stepwise algorithm.
        pearson  : X is ordered based on the Pearson's correlation coefficient.
        spearman : X is ordered based on the Spearman's correlation coefficient.
        random   : Shuffles the vector X before applying the stepwise algorithm.
        none     : Does not change the order of X.
print_info: bool, optional
    If set to True, prints the model information at each step.
show: bool, optional
    If set to True, the stepwise graphic will be drawn.
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
    from verticapy.learn.metrics import aic_bic

    if isinstance(X, str):
        X = [X]
    if isinstance(x_order, str):
        x_order = x_order.lower()
    assert len(X) >= 1, ParameterError("Vector X must have at least one element.")
    check_types(
        [
            ("criterion", criterion, ["aic", "bic",]),
            ("direction", direction, ["forward", "backward",]),
            ("max_steps", max_steps, [int, float,]),
            ("print_info", print_info, [bool,]),
            ("x_order", x_order, ["pearson", "spearman", "random", "none",]),
        ]
    )
    does_model_exist(name=estimator.name, cursor=estimator.cursor, raise_error=True)
    result, current_step = [], 0
    table = input_relation if isinstance(input_relation, str) else input_relation.__genSQL__()
    estimator.cursor.execute(f"SELECT AVG({y}) FROM {table}")
    avg = estimator.cursor.fetchone()[0]
    k = 0 if criterion == "aic" else 1
    if x_order == "random":
        random.shuffle(X)
    elif x_order in ("spearman", "pearson"):
        if isinstance(input_relation, str):
            vdf = vdf_from_relation(input_relation, cursor=estimator.cursor)
        else:
            vdf = input_relation
        X = [elem for elem in vdf.corr(method=x_order, focus=y, columns=X, show=False,)["index"]]
        if direction == "backward":
            X.reverse()
    if print_info:
        print("\033[1m\033[4mStarting Stepwise\033[0m\033[0m")
    if verticapy.options["tqdm"] and print_info:
        from tqdm.auto import tqdm

        loop = tqdm(range(len(X)))
    else:
        loop = range(len(X))
    model_id = 0
    if direction == "backward":
        X_current = [elem for elem in X]
        estimator.drop()
        estimator.fit(input_relation, X, y)
        current_score = estimator.score(criterion)
        result += [(X_current, current_score, None, None, 0, None)]
        for idx in loop:
            if print_info and idx == 0:
                print(f"\033[1m[Model 0]\033[0m \033[92m{criterion}: {current_score}\033[0m; Variables: {X_current}")
            if current_step >= max_steps:
                break
            X_test = [elem for elem in X_current]
            X_test.remove(X[idx])
            if len(X_test) != 0:
                estimator.drop()
                estimator.fit(input_relation, X_test, y)
                test_score = estimator.score(criterion,)
            else:
                test_score = aic_bic(y, str(avg), input_relation, estimator.cursor, 0)[k]
            score_diff = test_score - current_score
            if test_score - current_score < criterion_threshold:
                sign = "-"
                model_id += 1
                current_score = test_score
                X_current = [elem for elem in X_test]
                if print_info:
                    print(f"\033[1m[Model {model_id}]\033[0m \033[92m{criterion}: {test_score}\033[0m; \033[91m(-) Variable: {X[idx]}\033[0m")
            else:
                sign = "+"
            result += [(X_test, test_score, sign, X[idx], idx + 1, score_diff)]
            current_step += 1
    else:
        X_current = []
        current_score = aic_bic(y, str(avg), input_relation, estimator.cursor, 0)[k]
        result += [(X_current, current_score, None, None, 0, None)]
        for idx in loop:
            if print_info and idx == 0:
                print(f"\033[1m[Model 0]\033[0m \033[92m{criterion}: {current_score}\033[0m; Variables: {X_current}")
            if current_step >= max_steps:
                break
            X_test = [elem for elem in X_current] + [X[idx]]
            estimator.drop()
            estimator.fit(input_relation, X_test, y)
            test_score = estimator.score(criterion,)
            score_diff = current_score - test_score
            if current_score - test_score > criterion_threshold:
                sign = "+"
                model_id += 1
                current_score = test_score
                X_current = [elem for elem in X_test]
                if print_info:
                    print(f"\033[1m[Model {model_id}]\033[0m \033[92m{criterion}: {test_score}\033[0m; \033[91m(+) Variable: {X[idx]}\033[0m")
            else:
                sign = "-"
            result += [(X_test, test_score, sign, X[idx], idx + 1, score_diff)]
            current_step += 1
    if print_info:
        print(f"\033[1m\033[4mSelected Model\033[0m\033[0m\n")
        print(f"\033[1m[Model {model_id}]\033[0m \033[92m{criterion}: {current_score}\033[0m; Variables: {X_current}")
    features = [elem[0] for elem in result]
    for idx, elem in enumerate(features):
        features[idx] = [item.replace('"', '') for item in elem]
    importance = [elem[5] if (elem[5]) and elem[5] > 0 else 0 for elem in result]
    importance = [100 * elem / sum(importance) for elem in importance]
    result = tablesample({"index": [elem[4] for elem in result], "features": features, criterion: [elem[1] for elem in result], "change": [elem[2] for elem in result], "variable": [elem[3] for elem in result], "importance": importance})
    estimator.drop()
    if not(drop_final_estimator):
        estimator.fit(input_relation, X_current, y)
    result.best_list_ = X_current
    if show:
        plot_stepwise_ml([len(elem) for elem in result["features"]], result[criterion], result["variable"], result["change"], [result["features"][0], X_current], x_label="n_features", y_label=criterion, direction=direction, ax=ax, **style_kwds,)
        coeff_importances = {}
        for idx in range(len(importance)):
            if result["variable"][idx] != None:
                coeff_importances[result["variable"][idx]] = importance[idx]
        plot_importance(coeff_importances, print_legend=False, ax=ax, **style_kwds,)
    return result


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
Draws the validation curve.

Parameters
----------
estimator: object
    Vertica estimator with a fit method and a database cursor.
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
        max    : Max error
        mae    : Mean absolute error
        median : Median absolute error
        mse    : Mean squared error
        msle   : Mean squared log error
        r2     : R-squared coefficient
        r2a    : R2 adjusted
        rmse   : Root-mean-squared error
        var    : Explained variance
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
