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
import random, datetime
import numpy as np

# VerticaPy Modules
from verticapy import vDataFrame
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.errors import *
from verticapy.learn.ensemble import *
from verticapy.learn.naive_bayes import *
from verticapy.learn.linear_model import *
from verticapy.learn.decomposition import *
from verticapy.learn.cluster import *
from verticapy.learn.neighbors import *
from verticapy.learn.svm import *
from verticapy.learn.mlplot import plot_bubble_ml
from verticapy.learn.vmodel import *


class vAuto(vModel):
    # ---#
    def set_params(self, parameters: dict = {}):
        """
    ---------------------------------------------------------------------------
    Sets the parameters of the model.

    Parameters
    ----------
    parameters: dict, optional
        New parameters.
        """
        for elem in self.parameters:
            if elem not in parameters:
                parameters[elem] = self.parameters[elem]
        self.__init__(self.name, self.cursor, **parameters)


class AutoDataPrep(vAuto):
    """
---------------------------------------------------------------------------
Automatically find relations between the different features to preprocess
the data according to each column type.

Parameters
----------
name: str, optional
    Name of the model in which to store the output relation in the
    Vertica database.
cursor: DBcursor, optional
    Vertica database cursor.
cat_method: str, optional
    Method for encoding categorical features. This can be set to 'label' for
    label encoding and 'ooe' for One-Hot Encoding.
num_method: str, optional
    [Only used for non-time series datasets]
    Method for encoding numerical features. This can be set to 'same_freq' to
    encode using frequencies, 'same_width' to encode using regular bins, or
    'none' to not encode numerical features.
nbins: int, optional
    [Only used for non-time series datasets]
    Number of bins used to discretize numerical features.
outliers_threshold: float, optional
    [Only used for non-time series datasets]
    How to deal with outliers. If a number is used, all elements with an absolute 
    z-score greater than the threshold will be converted to NULL values. Otherwise,
    outliers are treated as regular values.
na_method: str, optional
    Method for handling missing values.
        auto: Mean for the numerical features and creates a new category for the 
              categorical vColumns. For time series datasets, 'constant' interpolation 
              is used for categorical features and 'linear' for the others.
        drop: Drops the missing values.
cat_topk: int, optional
    Keeps the top-k most frequent categories and merges the others into one unique 
    category. If unspecified, all categories are kept.
normalize: bool, optional
    If True, the data will be normalized using the z-score. The 'num_method' parameter
    must be set to 'none'.
normalize_min_cat: int, optional
    Minimum feature cardinality before using normalization.
id_method: str, optional
    Method for handling ID features.
        drop: Drops any feature detected as ID.
        none: Does not change ID features.
apply_pca: bool, optional
    [Only used for non-time series datasets]
    If True, a PCA is applied at the end of the preprocessing.
rule: str / time, optional
    [Only used for time series datasets]
    Interval to use to slice the time. For example, '5 minutes' will create records
    separated by '5 minutes' time interval.
    If set to auto, the rule will be detected using aggregations.
identify_ts: bool, optional
    If True and parameter 'ts' is undefined when fitting the model, the function will
    try to automatically detect the parameter 'ts'.
save: bool, optional
    If True, saves the final relation inside the database.

Attributes
----------
X_in: list
    Variables used to fit the model.
X_out: list
    Variables created by the model.
ts: str
    TS component.
by: list
    vcolumns used in the partition.
sql_: str
    SQL needed to deploy the model.
final_relation_: vDataFrame
    Relation created after fitting the model.
model_grid_ : tablesample
    Grid containing the different models information.
    """
    # ---#
    def __init__(self,
                 name: str = "",
                 cursor=None,
                 cat_method: str = "ooe",
                 num_method: str = "none",
                 nbins: int = 20,
                 outliers_threshold: float = 4.0,
                 na_method: str = "auto",
                 cat_topk: int = 10,
                 normalize: bool = True,
                 normalize_min_cat: int = 6,
                 id_method: int = "drop",
                 apply_pca: bool = False,
                 rule: (str, datetime.timedelta) = "auto",
                 identify_ts: bool = True,
                 save: bool = True,):
        check_types([("name", name, [str],), 
                     ("cat_method", cat_method, ["label", "ooe"],),
                     ("num_method", num_method, ["same_freq", "same_width", "none",],),
                     ("nbins", nbins, [int, float,],),
                     ("outliers_threshold", outliers_threshold, [int, float,],),
                     ("na_method", na_method, ["auto", "drop",],),
                     ("cat_topk", cat_topk, [int, float,],),
                     ("rule", rule, [str, datetime.timedelta,],),
                     ("normalize", normalize, [bool,],),
                     ("id_method", id_method, ["none", "drop",],),
                     ("apply_pca", apply_pca, [bool,],),
                     ("normalize_min_cat", normalize_min_cat, [int, float,],),
                     ("identify_ts", identify_ts, [bool,],),
                     ("save", save, [bool,],),])
        self.type, self.name = "AutoDataPrep", name
        if not(self.name):
            self.name = "AutoDataPrep_{}".format(get_session(cursor))
        self.parameters = {"cat_method": cat_method,
                           "num_method": num_method,
                           "nbins": nbins,
                           "outliers_threshold": outliers_threshold,
                           "na_method": na_method,
                           "cat_topk": cat_topk,
                           "rule": rule,
                           "normalize": normalize,
                           "normalize_min_cat": normalize_min_cat,
                           "apply_pca": apply_pca,
                           "id_method": id_method,
                           "identify_ts": identify_ts,
                           "save": save,}
        cursor = check_cursor(cursor)[0]
        self.cursor = cursor

    # ---#
    def fit(
        self,
        input_relation: (str, vDataFrame),
        X: list = [],
        ts: str = "",
        by: list = [],
    ):
        """
    ---------------------------------------------------------------------------
    Trains the model.

    Parameters
    ----------
    input_relation: str/vDataFrame
        Training Relation.
    X: list, optional
        List of the features to preprocess.
    ts: str, optional
        Time series vcolumn to use to order the data. The vcolumn type must be
        date-like (date, datetime, timestamp...)
    by: list, optional
        vcolumns used in the partition.

    Returns
    -------
    object
        the cleaned relation
        """
        current_print_info = verticapy.options["print_info"]
        verticapy.options["print_info"] = False
        assert not(by) or (ts), ParameterError("Parameter 'by' must be empty if 'ts' is not defined.")
        if isinstance(input_relation, str):
            vdf = vdf_from_relation(input_relation, cursor=self.cursor)
        else:
            vdf = input_relation.copy()
        if not(X):
            X = vdf.get_columns()
        if not(ts) and self.parameters["identify_ts"]:
            nb_date, nb_num, nb_others = 0, 0, 0
            for elem in X:
                if vdf[elem].isnum() and not(vdf[elem].isbool()):
                    nb_num += 1
                elif vdf[elem].isdate():
                    nb_date += 1
                    ts_tmp = elem
                else:
                    nb_others += 1
                    cat_tmp = elem
            if nb_date == 1 and nb_others <= 1:
                ts = ts_tmp
            if nb_date == 1 and nb_others == 1:
                by = [cat_tmp]
        columns_check(X, vdf)
        X = vdf_columns_names(X, vdf)
        X_diff = vdf.get_columns(exclude_columns = X)
        columns_to_drop = []
        n = vdf.shape()[0]
        for elem in X:
            is_id = not(vdf[elem].isnum()) and not(vdf[elem].isdate()) and 0.9 * n <= vdf[elem].nunique()
            if self.parameters["id_method"] == "drop" and is_id and (not(by) or elem not in by):
                columns_to_drop += [elem]
                X_diff += [elem]
            elif not(is_id) and (not(by) or elem not in by):
                if not(vdf[elem].isdate()):
                    if vdf[elem].isnum():
                        if (self.parameters["outliers_threshold"]) and self.parameters["outliers_threshold"] > 0:
                            vdf[elem].fill_outliers(method = "null", threshold = self.parameters["outliers_threshold"])
                        if self.parameters["num_method"] in ("none",) and (self.parameters["normalize"]) and (self.parameters["normalize_min_cat"] < 2 or (vdf[elem].nunique() > self.parameters["normalize_min_cat"])):
                            vdf[elem].normalize(method = "zscore")
                        if self.parameters["na_method"] == "auto":
                            vdf[elem].fillna(method = "mean")
                        else:
                            vdf[elem].dropna()
                    if vdf[elem].isnum() and not(ts) and self.parameters["num_method"] in ("same_width", "same_freq",):
                        vdf[elem].discretize(method = self.parameters["num_method"], bins = self.parameters["nbins"],)
                    elif vdf[elem].nunique() > self.parameters["cat_topk"] and not(vdf[elem].isnum()):
                        if self.parameters["na_method"] == "auto":
                            vdf[elem].fillna("NULL")
                        else:
                            vdf[elem].dropna()
                        vdf[elem].discretize(method = "topk", k = self.parameters["cat_topk"],)
                    if (self.parameters["cat_method"] == 'ooe' and not(vdf[elem].isnum())) or (vdf[elem].isnum() and not(ts) and self.parameters["num_method"] in ("same_width", "same_freq",)):
                        vdf[elem].get_dummies(drop_first=False)
                        columns_to_drop += [elem]
                    elif (self.parameters["cat_method"] == 'label' and not(vdf[elem].isnum())) or (vdf[elem].isnum() and not(ts) and self.parameters["num_method"] in ("same_width", "same_freq",)):
                        vdf[elem].label_encode()
                elif not(ts):
                    vdf[elem.replace('"', '') + "_year"] = f"YEAR({elem})"
                    vdf[elem.replace('"', '') + "_dayofweek"] = f"DAYOFWEEK({elem})"
                    vdf[elem.replace('"', '') + "_month"] = f"MONTH({elem})"
                    vdf[elem.replace('"', '') + "_hour"] = f"HOUR({elem})"
                    vdf[elem.replace('"', '') + "_quarter"] = f"QUARTER({elem})"
                    vdf[elem.replace('"', '') + "_trend"] = f"({elem}::timestamp - MIN({elem}::timestamp) OVER ()) / '1 second'::interval"
                    columns_to_drop += [elem]
        if columns_to_drop:
            vdf.drop(columns_to_drop)
        if ts:
            columns_check([ts] + by, vdf)
            ts = vdf_columns_names([ts], vdf)[0]
            by = vdf_columns_names(by, vdf)
            if self.parameters["rule"] == "auto":
                vdf_tmp = vdf[[ts] + by]
                by_tmp = "PARTITION BY {} ".format(", ".join(by)) if (by) else ""
                vdf_tmp["verticapy_time_delta"] = f"({ts}::timestamp - (LAG({ts}) OVER ({by_tmp}ORDER BY {ts}))::timestamp) / '00:00:01'"
                vdf_tmp = vdf_tmp.groupby(["verticapy_time_delta"], ["COUNT(*) AS cnt"])
                self.cursor.execute("SELECT verticapy_time_delta FROM {} ORDER BY cnt DESC LIMIT 1".format(vdf_tmp.__genSQL__()))
                rule = datetime.timedelta(seconds = self.cursor.fetchone()[0])
            method = {}
            X_tmp = []
            for elem in X:
                if elem != ts and elem not in by:
                    if vdf[elem].isnum() and not(vdf[elem].isbool()):
                        method[elem] = "linear"
                    else:
                        method[elem] = "ffill"
            vdf = vdf.asfreq(ts=ts, rule=rule, method=method, by=by,)
            vdf.dropna()
        self.X_in = [elem for elem in X]
        self.X_out = vdf.get_columns(exclude_columns = by + [ts] + X_diff if ts else by + X_diff)
        self.by = by
        self.ts = ts
        if self.parameters["apply_pca"] and not(ts):
            model_pca = PCA(self.name + "_pca", cursor=self.cursor)
            model_pca.drop()
            model_pca.fit(vdf, self.X_out)
            vdf = model_pca.transform()
            self.X_out = vdf.get_columns(exclude_columns = by + [ts] + X_diff if ts else by + X_diff)
        self.sql_ = vdf.__genSQL__()
        if self.parameters["save"]:
            vdf.to_db(name = self.name, relation_type = "table", inplace=True,)
        self.final_relation_ = vdf
        verticapy.options["print_info"] = current_print_info
        return self.final_relation_


class AutoClustering(vAuto):
    """
---------------------------------------------------------------------------
Automatically creates k different groups with which to generalize the data.

Parameters
----------
name: str
    Name of the model.
cursor: DBcursor, optional
    Vertica database cursor.
n_cluster: int, optional
    Number of clusters. If empty, an optimal number of clusters will be
    determined using multiple k-means models.
init: str/list, optional
    The method for finding the initial cluster centers.
        kmeanspp : Uses the k-means++ method to initialize the centers.
        random   : Randomly subsamples the data to find initial centers.
    Alternatively, you can specify a list with the initial custer centers.
max_iter: int, optional
    The maximum number of iterations for the algorithm.
tol: float, optional
    Determines whether the algorithm has converged. The algorithm is considered 
    converged after no center has moved more than a distance of 'tol' from the 
    previous iteration.
preprocess_data: bool, optional
    If True, the data will be preprocessed.
preprocess_dict: dict, optional
    Dictionary to pass to the AutoDataPrep class in order to 
    preprocess the data before the clustering.
print_info: bool
    If True, prints the model information at each step.

Attributes
----------
preprocess_: object
    Model used to preprocess the data.
model_: object
    Final model used for the clustering.
    """
    # ---#
    def __init__(self,
                 name: str,
                 cursor=None,
                 n_cluster: int = None,
                 init: str = "kmeanspp",
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 preprocess_data: bool = True,
                 preprocess_dict: dict = {"identify_ts": False, "normalize_min_cat": 0, "outliers_threshold": 3.0, "na_method": "drop",},
                 print_info: bool = True,):
        check_types([("name", name, [str],),
                     ("n_cluster", n_cluster, [int],),
                     ("init", init, [str, list],),
                     ("max_iter", max_iter, [int],),
                     ("tol", tol, [float],),
                     ("preprocess_data", preprocess_data, [bool,]),
                     ("preprocess_dict", preprocess_dict, [dict,]),
                     ("print_info", print_info, [bool,]),])
        self.type, self.name = "AutoClustering", name
        self.parameters = {"n_cluster": n_cluster,
                           "init": init,
                           "max_iter": max_iter,
                           "tol": tol,
                           "print_info": print_info,
                           "preprocess_data": preprocess_data,
                           "preprocess_dict": preprocess_dict,}
        cursor = check_cursor(cursor)[0]
        self.cursor = cursor

    # ---#
    def fit(
        self,
        input_relation: (str, vDataFrame),
        X: list = [],
    ):
        """
    ---------------------------------------------------------------------------
    Trains the model.

    Parameters
    ----------
    input_relation: str/vDataFrame
        Training Relation.
    X: list, optional
        List of the predictors.

    Returns
    -------
    object
        clustering model
        """
        if self.parameters["print_info"]:
            print(f"\033[1m\033[4mStarting AutoClustering\033[0m\033[0m\n")
        if self.parameters["preprocess_data"]:
            model_preprocess = AutoDataPrep(cursor=self.cursor, **self.parameters["preprocess_dict"],)
            input_relation = model_preprocess.fit(input_relation, X=X,)
            X = [elem for elem in model_preprocess.X_out]
            self.preprocess_ = model_preprocess
        else:
            self.preprocess_ = None
        if not(self.parameters["n_cluster"]):
            if self.parameters["print_info"]:
                print(f"\033[1m\033[4mFinding a suitable number of clusters\033[0m\033[0m\n")
            self.parameters["n_cluster"] = best_k(input_relation=input_relation,
                                                  X=X,
                                                  cursor=self.cursor,
                                                  n_cluster=(1, 100),
                                                  init=self.parameters["init"],
                                                  max_iter=self.parameters["max_iter"],
                                                  tol=self.parameters["tol"],
                                                  elbow_score_stop=0.9,
                                                  tqdm=self.parameters["print_info"],)
        if self.parameters["print_info"]:
            print(f"\033[1m\033[4mBuilding the Final Model\033[0m\033[0m\n")
        if verticapy.options["tqdm"] and self.parameters["print_info"]:
            from tqdm.auto import tqdm

            loop = tqdm(range(1))
        else:
            loop = range(1)
        for i in loop:
            self.model_ = KMeans(self.name, cursor=self.cursor, n_cluster=self.parameters["n_cluster"], init=self.parameters["init"], max_iter=self.parameters["max_iter"], tol=self.parameters["tol"],)
            self.model_.fit(input_relation, X=X,)
        return self.model_

class AutoML(vAuto):
    """
---------------------------------------------------------------------------
Tests multiple models to find those that maximize the input score.

Parameters
----------
name: str
    Name of the model.
cursor: DBcursor, optional
    Vertica database cursor.
estimator: list / 'native' / 'all' / 'fast' / object
    List of Vertica estimators with a fit method and a database cursor.
    Alternatively, you can specify 'native' for all native Vertica models,
    'all' for all VerticaPy models and 'fast' for quick modeling.
estimator_type: str, optional
    Estimator Type.
        auto      : Automatically detects the estimator type.
        regressor : The estimator will be used to perform a regression.
        binary    : The estimator will be used to perform a binary classification.
        multi     : The estimator will be used to perform a multiclass classification.
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
    Number of bins used to compute the different parameter categories.
lmax: int, optional
    Maximum length of each parameter list.
optimized_grid: int, optional
    If set to 0, the randomness is based on the input parameters.
    If set to 1, the randomness is limited to some parameters while others
    are picked based on a default grid.
    If set to 2, no randomness is used and a default grid is returned.
stepwise: bool, optional
    If True, the stepwise algorithm will be used to determine the
    final model list of parameters.
stepwise_criterion: str, optional
    Criterion used when doing the final estimator stepwise.
        aic : Akaike’s information criterion
        bic : Bayesian information criterion
stepwise_direction: str, optional
    Which direction to start the stepwise search. Can be done 'backward' or 'forward'.
stepwise_max_steps: int, optional
    The maximum number of steps to be considered when doing the final estimator 
    stepwise.
x_order: str, optional
    Method for preprocessing X before using the stepwise algorithm.
        pearson  : X is ordered based on the Pearson's correlation coefficient.
        spearman : X is ordered based on Spearman's rank correlation coefficient.
        random   : Shuffles the vector X before applying the stepwise algorithm.
        none     : Does not change the order of X.
preprocess_data: bool, optional
    If True, the data will be preprocessed.
preprocess_dict: dict, optional
    Dictionary to pass to the AutoDataPrep class in order to 
    preprocess the data before the clustering.
print_info: bool
    If True, prints the model information at each step.

Attributes
----------
preprocess_: object
    Model used to preprocess the data.
best_model_: object
    Most efficient models found during the search.
model_grid_ : tablesample
    Grid containing the different models information.
    """
    # ---#
    def __init__(self,
                 name: str,
                 cursor=None,
                 estimator: (list, str) = "fast",
                 estimator_type: str = "auto",
                 metric: str = "auto",
                 cv: int = 3,
                 pos_label: (int, float, str) = None,
                 cutoff: float = -1,
                 nbins: int = 100,
                 lmax: int = 5,
                 optimized_grid: int = 2,
                 stepwise: bool = True,
                 stepwise_criterion: str = "aic",
                 stepwise_direction: str = "backward",
                 stepwise_max_steps: int = 100,
                 stepwise_x_order: str = "pearson",
                 preprocess_data: bool = True,
                 preprocess_dict: dict = {"identify_ts": False,},
                 print_info: bool = True,):
        check_types([("name", name, [str],), 
                     ("estimator_type", estimator_type, [str],),
                     ("metric", metric, [str],),
                     ("cv", cv, [int],),
                     ("pos_label", pos_label, [int, float, str],),
                     ("cutoff", cutoff, [int, float],),
                     ("nbins", nbins, [int],),
                     ("optimized_grid", optimized_grid, [int],),
                     ("print_info", print_info, [bool],),
                     ("stepwise", stepwise, [bool],),
                     ("stepwise_criterion", stepwise_criterion, ["aic", "bic",]),
                     ("stepwise_direction", stepwise_direction, ["forward", "backward",]),
                     ("stepwise_max_steps", stepwise_max_steps, [int, float,]),
                     ("stepwise_x_order", stepwise_x_order, ["pearson", "spearman", "random", "none",]),
                     ("preprocess_data", preprocess_data, [bool,]),
                     ("preprocess_dict", preprocess_dict, [dict,]),])
        assert optimized_grid in [0, 1, 2,], ParameterError("Optimized Grid must be an integer between 0 and 2.")
        self.type, self.name = "AutoML", name
        self.parameters = {"estimator": estimator,
                           "estimator_type": estimator_type,
                           "metric": metric,
                           "cv": cv,
                           "pos_label": pos_label,
                           "cutoff": cutoff,
                           "nbins": nbins,
                           "lmax": lmax,
                           "optimized_grid": optimized_grid,
                           "print_info": print_info,
                           "stepwise": stepwise,
                           "stepwise_criterion": stepwise_criterion,
                           "stepwise_direction": stepwise_direction,
                           "stepwise_max_steps": stepwise_max_steps,
                           "stepwise_x_order": stepwise_x_order,
                           "preprocess_data": preprocess_data,
                           "preprocess_dict": preprocess_dict,}
        cursor = check_cursor(cursor)[0]
        self.cursor = cursor

    # ---#
    def fit(
        self,
        input_relation: (str, vDataFrame),
        X: list = [],
        y: str = "",
    ):
        """
    ---------------------------------------------------------------------------
    Trains the model.

    Parameters
    ----------
    input_relation: str/vDataFrame
        Training Relation.
    X: list, optional
        List of the predictors.
    y: str, optional
        Response column.
    Returns
    -------
    object
        model grid
        """
        if not(X):
            if not(y):
                exclude_columns = []
            else:
                exclude_columns = [y]
            if not(isinstance(input_relation, vDataFrame)):
                X = vdf_from_relation(input_relation, cursor=self.cursor).get_columns(exclude_columns = exclude_columns)
            else:
                X = input_relation.get_columns(exclude_columns = exclude_columns)
        if isinstance(self.parameters["estimator"], str):
            self.parameters["estimator"] = self.parameters["estimator"].lower()
            check_types([("estimator", self.parameters["estimator"], ["native", "all", "fast",],),])
            modeltype = None
            if not(isinstance(input_relation, vDataFrame)):
                vdf = vdf_from_relation(input_relation, cursor=self.cursor)
            else:
                vdf = input_relation
            if self.parameters["estimator_type"].lower() == "binary" or (self.parameters["estimator_type"].lower() == "auto" and sorted(vdf[y].distinct()) == [0, 1]):
                self.parameters["estimator_type"] = "binary"
                if self.parameters["estimator"] == "native":
                    self.parameters["estimator"] = [LinearSVC(self.name, cursor=self.cursor), LogisticRegression(self.name, cursor=self.cursor), RandomForestClassifier(self.name, cursor=self.cursor), XGBoostClassifier(self.name, cursor=self.cursor), NaiveBayes(self.name, cursor=self.cursor)]
                elif self.parameters["estimator"] == "fast":
                    self.parameters["estimator"] = [LogisticRegression(self.name, cursor=self.cursor), RandomForestClassifier(self.name, cursor=self.cursor), NaiveBayes(self.name, cursor=self.cursor)]
                else:
                    self.parameters["estimator"] = [LinearSVC(self.name, cursor=self.cursor), LogisticRegression(self.name, cursor=self.cursor), RandomForestClassifier(self.name, cursor=self.cursor), XGBoostClassifier(self.name, cursor=self.cursor), NaiveBayes(self.name, cursor=self.cursor), KNeighborsClassifier(self.name, cursor=self.cursor), NearestCentroid(self.name, cursor=self.cursor)]
            elif self.parameters["estimator_type"].lower() == "regressor" or (self.parameters["estimator_type"].lower() == "auto" and vdf[y].isnum()):
                self.parameters["estimator_type"] = "regressor"
                if self.parameters["estimator"] == "native":
                    self.parameters["estimator"] = [LinearSVR(self.name, cursor=self.cursor), ElasticNet(self.name, cursor=self.cursor), LinearRegression(self.name, cursor=self.cursor), Lasso(self.name, cursor=self.cursor), Ridge(self.name, cursor=self.cursor), RandomForestRegressor(self.name, cursor=self.cursor), XGBoostRegressor(self.name, cursor=self.cursor),]
                elif self.parameters["estimator"] == "fast":
                    self.parameters["estimator"] = [LinearRegression(self.name, cursor=self.cursor), ElasticNet(self.name, cursor=self.cursor), RandomForestRegressor(self.name, cursor=self.cursor),]
                else:
                    self.parameters["estimator"] = [LinearSVR(self.name, cursor=self.cursor), ElasticNet(self.name, cursor=self.cursor), LinearRegression(self.name, cursor=self.cursor), Lasso(self.name, cursor=self.cursor), Ridge(self.name, cursor=self.cursor), RandomForestRegressor(self.name, cursor=self.cursor), XGBoostRegressor(self.name, cursor=self.cursor), KNeighborsRegressor(self.name, cursor=self.cursor),]
            elif self.parameters["estimator_type"].lower() in ("multi", "auto",):
                self.parameters["estimator_type"] = "multi"
                if self.parameters["estimator"] == "native":
                    self.parameters["estimator"] = [RandomForestClassifier(self.name, cursor=self.cursor), XGBoostClassifier(self.name, cursor=self.cursor), NaiveBayes(self.name, cursor=self.cursor)]
                elif self.parameters["estimator"] == "fast":
                    self.parameters["estimator"] = [RandomForestClassifier(self.name, cursor=self.cursor), NaiveBayes(self.name, cursor=self.cursor)]
                else:
                    self.parameters["estimator"] = [RandomForestClassifier(self.name, cursor=self.cursor), XGBoostClassifier(self.name, cursor=self.cursor), NaiveBayes(self.name, cursor=self.cursor), KNeighborsClassifier(self.name, cursor=self.cursor), NearestCentroid(self.name, cursor=self.cursor)]
            else:
                raise ParameterError(f"Parameter 'estimator_type' must be in auto|binary|multi|regressor. Found {estimator_type}.")
        elif isinstance(self.parameters["estimator"], (RandomForestRegressor, RandomForestClassifier, XGBoostRegressor, XGBoostClassifier, NaiveBayes, LinearRegression, ElasticNet, Lasso, Ridge, LogisticRegression, KNeighborsRegressor, KNeighborsClassifier, NearestCentroid, LinearSVC, LinearSVR)):
            self.parameters["estimator"] = [self.parameters["estimator"]]
        else:
            for elem in self.parameters["estimator"]:
                assert isinstance(elem, (RandomForestRegressor, RandomForestClassifier, XGBoostRegressor, XGBoostClassifier, NaiveBayes, LinearRegression, ElasticNet, Lasso, Ridge, LogisticRegression, KNeighborsRegressor, KNeighborsClassifier, NearestCentroid, LinearSVC, LinearSVR)), ParameterError("estimator must be a list of VerticaPy estimators. Found {}.".format(type(elem)))
        if self.parameters["estimator_type"] == "auto":
            self.parameters["estimator_type"] = self.parameters["estimator"][0].type
        for elem in self.parameters["estimator"]:
            cat = category_from_model_type(elem.type)
            assert self.parameters["estimator_type"] in ("binary", "multi") and cat[0] == "classifier" or self.parameters["estimator_type"] in ("regressor",) and cat[0] == "regressor", ParameterError("Incorrect list for parameter 'estimator'. Expected type '{}', found type '{}'.".format(self.parameters["estimator_type"], cat[0]))
        if self.parameters["estimator_type"] == "regressor" and self.parameters["metric"] == "auto":
            self.parameters["metric"] = "rmse"
        elif self.parameters["metric"] == "auto":
            self.parameters["metric"] = "logloss"
        result = tablesample(
                    {
                        "model_type": [],
                        "parameters": [],
                        "avg_score": [],
                        "avg_train_score": [],
                        "avg_time": [],
                        "score_std": [],
                        "score_train_std": [],
                        "model_class": [],
                    }
                )
        if self.parameters["preprocess_data"]:
            model_preprocess = AutoDataPrep(cursor=self.cursor, **self.parameters["preprocess_dict"],)
            input_relation = model_preprocess.fit(input_relation, X=X,)
            X = [elem for elem in model_preprocess.X_out]
            self.preprocess_ = model_preprocess
        else:
            self.preprocess_ = None
        if self.parameters["print_info"]:
            print(f"\033[1m\033[4mStarting AutoML\033[0m\033[0m\n")
        if verticapy.options["tqdm"] and self.parameters["print_info"]:
            from tqdm.auto import tqdm
            loop = tqdm(self.parameters["estimator"])
        else:
            loop = self.parameters["estimator"]
        for elem in loop:
            if self.parameters["print_info"]:
                print(f"\n\033[1m\033[4mTesting Model - {str(elem.__class__).split('.')[-1][:-2]}\033[0m\033[0m\n")
            param_grid = gen_params_grid(elem, self.parameters["nbins"], len(X), self.parameters["lmax"], self.parameters["optimized_grid"])
            gs = grid_search_cv(
                elem,
                param_grid,
                input_relation,
                X,
                y,
                self.parameters["metric"],
                self.parameters["cv"],
                self.parameters["pos_label"],
                self.parameters["cutoff"],
                True,
                "no_print",
                self.parameters["print_info"],
            )
            if gs["parameters"] != [] and gs["avg_score"] != [] and gs["avg_train_score"] != [] and gs["avg_time"] != [] and gs["score_std"] != [] and gs["score_train_std"] != []:
                result.values["model_type"] += [str(elem.__class__).split('.')[-1][:-2]] * len(gs["parameters"])
                result.values["parameters"] += gs["parameters"]
                result.values["avg_score"] += gs["avg_score"]
                result.values["avg_train_score"] += gs["avg_train_score"]
                result.values["avg_time"] += gs["avg_time"]
                result.values["score_std"] += gs["score_std"]
                result.values["score_train_std"] += gs["score_train_std"]
                result.values["model_class"] += [elem.__class__] * len(gs["parameters"])

        data = [(result["model_type"][i], result["parameters"][i], result["avg_score"][i], result["avg_train_score"][i], result["avg_time"][i], result["score_std"][i], result["score_train_std"][i], result["model_class"][i]) for i in range(len(result["score_train_std"]))]
        reverse = reverse_score(self.parameters["metric"])
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
                "model_class": [elem[7] for elem in data],
            }
        )
        if self.parameters["print_info"]:
            print(f"\033[1m\033[4mFinal Model\033[0m\033[0m\n")
            print(f"{result['model_type'][0]}; Best_Parameters: {result['parameters'][0]}; \033[91mBest_Test_score: {result['avg_score'][0]}\033[0m; \033[92mTrain_score: {result['avg_train_score'][0]}\033[0m; \033[94mTime: {result['avg_time'][0]}\033[0m;\n\n")
        best_model = result["model_class"][0](self.name, self.cursor)
        best_model.set_params(result["parameters"][0])
        self.stepwise_ = None
        if self.parameters["stepwise"]:
            self.stepwise_ = stepwise(best_model, input_relation, X, y, criterion=self.parameters["stepwise_criterion"], direction=self.parameters["stepwise_direction"], max_steps=self.parameters["stepwise_max_steps"],  x_order=self.parameters["stepwise_x_order"], print_info=self.parameters["print_info"], drop_final_estimator=False, show=False, criterion_threshold=2,)
        else:
            best_model.fit(input_relation, X, y)
        self.best_model_ = best_model
        self.model_grid_ = result
        self.parameters["reverse"] = not(reverse)
        if self.preprocess_ != None:
            self.preprocess_.drop()
            self.preprocess_.final_relation_ = vdf_from_relation(self.preprocess_.sql_, cursor=self.cursor)
        return self.model_grid_

    # ---#
    def plot(
        self,
        mltype: str = "champion",
        ax=None,
        **style_kwds,
    ):
        """
    ---------------------------------------------------------------------------
    Draws the AutoML plot.

    Parameters
    ----------
    mltype: str, optional
        The plot type.
            champion: champion challenger plot.
            step    : stepwise plot.
    ax: Matplotlib axes object, optional
        The axes to plot on.
    **style_kwds
        Any optional parameter to pass to the Matplotlib functions.

    Returns
    -------
    ax
        Matplotlib axes object
        """
        if mltype == "champion":
            return plot_bubble_ml(self.model_grid_["avg_time"], self.model_grid_["avg_score"], self.model_grid_["score_std"], self.model_grid_["model_type"], x_label="time", y_label="score", title= "Model Type", ax=ax, reverse=(True, self.parameters["reverse"],), **style_kwds,)
        else:
            return plot_stepwise_ml([len(elem) for elem in self.stepwise_["features"]], self.stepwise_[self.parameters["stepwise_criterion"]], self.stepwise_["variable"], self.stepwise_["change"], [self.stepwise_["features"][0], self.stepwise_.best_list_], x_label="n_features", y_label=self.parameters["stepwise_criterion"], direction=self.parameters["stepwise_direction"], ax=ax, **style_kwds,)
