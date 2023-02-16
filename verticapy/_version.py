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
from functools import wraps

from verticapy.errors import VersionError

__version__ = "0.13.0"
MINIMUM_VERTICA_VERSION = {
    "Balance": [8, 1, 1],
    "BernoulliNB": [8, 0, 0],
    "BisectingKMeans": [9, 3, 1],
    "CategoricalNB": [8, 0, 0],
    "confusion_matrix": [8, 0, 0],
    "DecisionTreeClassifier": [8, 1, 1],
    "DecisionTreeRegressor": [9, 0, 1],
    "DummyTreeClassifier": [8, 1, 1],
    "DummyTreeRegressor": [9, 0, 1],
    "edit_distance": [10, 1, 0],
    "ElasticNet": [8, 0, 0],
    "GaussianNB": [8, 0, 0],
    "gen_dataset": [9, 3, 0],
    "get_tree": [9, 1, 1],
    "IsolationForest": [12, 0, 0],
    "jaro_distance": [12, 0, 2],
    "jaro_winkler_distance": [12, 0, 2],
    "Lasso": [8, 0, 0],
    "lift_chart": [8, 0, 0],
    "LinearRegression": [8, 0, 0],
    "LinearSVC": [8, 1, 0],
    "LinearSVR": [8, 1, 1],
    "LogisticRegression": [8, 0, 0],
    "KMeans": [8, 0, 0],
    "KPrototypes": [12, 0, 3],
    "MCA": [9, 1, 0],
    "MinMaxScaler": [8, 1, 0],
    "multilabel_confusion_matrix": [8, 0, 0],
    "MultinomialNB": [8, 0, 0],
    "NaiveBayes": [8, 0, 0],
    "Normalizer": [8, 1, 0],
    "OneHotEncoder": [9, 0, 0],
    "PCA": [9, 1, 0],
    "prc_curve": [9, 1, 0],
    "RandomForestClassifier": [8, 1, 1],
    "RandomForestRegressor": [9, 0, 1],
    "read_file": [11, 1, 1],
    "Ridge": [8, 0, 0],
    "RobustScaler": [8, 1, 0],
    "roc_curve": [8, 0, 0],
    "SARIMAX": [8, 0, 0],
    "soundex": [10, 1, 0],
    "soundex_matches": [10, 1, 0],
    "StandardScaler": [8, 1, 0],
    "SVD": [9, 1, 0],
    "VAR": [8, 0, 0],
    "XGBoostClassifier": [10, 1, 0],
    "XGBoostRegressor": [10, 1, 0],
}
VERTICA_VERSION = None


def check_minimum_version(func):
    """
check_minimum_version decorator. It simplifies the code by checking if the
feature is available in the user's version.
    """

    @wraps(func)
    def func_prec_check_minimum_version(*args, **kwargs):
        fun_name, object_name, condition = func.__name__, "", []
        if len(args) > 0:
            object_name = type(args[0]).__name__
        name = object_name if fun_name == "__init__" else fun_name
        vertica_version(MINIMUM_VERTICA_VERSION[name])

        return func(*args, **kwargs)

    return func_prec_check_minimum_version


def vertica_version(condition: list = []):
    """
Returns the Vertica Version.

Parameters
----------
condition: list, optional
    List of the minimal version information. If the current version is not
    greater or equal to this one, it will raise an error.

Returns
-------
list
    List containing the version information.
    [MAJOR, MINOR, PATCH, POST]
    """
    from verticapy._utils._sql import _executeSQL

    global VERTICA_VERSION

    if condition:
        condition = condition + [0 for elem in range(4 - len(condition))]
    if not (VERTICA_VERSION):
        current_version = _executeSQL(
            "SELECT /*+LABEL('utilities.version')*/ version();",
            title="Getting the version.",
            method="fetchfirstelem",
        ).split("Vertica Analytic Database v")[1]
        current_version = current_version.split(".")
        result = []
        try:
            result += [int(current_version[0])]
            result += [int(current_version[1])]
            result += [int(current_version[2].split("-")[0])]
            result += [int(current_version[2].split("-")[1])]
        except:
            pass
        VERTICA_VERSION = result
    else:
        result = VERTICA_VERSION
    if condition:
        if condition[0] < result[0]:
            test = True
        elif condition[0] == result[0]:
            if condition[1] < result[1]:
                test = True
            elif condition[1] == result[1]:
                if condition[2] <= result[2]:
                    test = True
                else:
                    test = False
            else:
                test = False
        else:
            test = False
        if not (test):
            v0, v1, v2 = result[0], result[1], str(result[2]).split("-")[0]
            v = ".".join([str(c) for c in condition[:3]])
            raise VersionError(
                (
                    "This Function is not available for Vertica version "
                    f"{v0}.{v1}.{v2}.\nPlease upgrade your Vertica "
                    f"version to at least {v} to get this functionality."
                )
            )
    return result
