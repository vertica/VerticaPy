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
# VerticaPy Modules
from verticapy import vDataFrame
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.errors import *
from verticapy.learn.vmodel import *

# ---#
class Pipeline:
    """
---------------------------------------------------------------------------
Creates a Pipeline object. Sequentially apply a list of transforms and a 
final estimator. The intermediate steps must implement a transform method.

Parameters
----------
steps: list
    List of (name, transform) tuples (implementing fit/transform) that are chained, 
    in the order in which they are chained, with the last object an estimator.
	"""

    def __init__(
        self, steps: list,
    ):
        check_types([("steps", steps, [list],)])
        self.type = "Pipeline"
        self.steps = []
        for idx, elem in enumerate(steps):
            if len(elem) != 2:
                raise ParameterError(
                    "The steps of the Pipeline must be composed of 2 elements (name, transform). Found {}.".format(
                        len(elem)
                    )
                )
            elif not (isinstance(elem[0], str)):
                raise ParameterError(
                    "The steps 'name' of the Pipeline must be of type str. Found {}.".format(
                        type(elem[0])
                    )
                )
            else:
                try:
                    if idx < len(steps) - 1:
                        elem[1].transform
                    elem[1].fit
                except:
                    if idx < len(steps) - 1:
                        raise ParameterError(
                            "The estimators of the Pipeline must have a 'transform' and a 'fit' method."
                        )
                    else:
                        raise ParameterError(
                            "The last estimator of the Pipeline must have a 'fit' method."
                        )
            self.steps += [elem]
        self.cursor = self.steps[-1][1].cursor

    # ---#
    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.steps[index]
        elif isinstance(index, int):
            return self.steps[index][1]
        else:
            return getattr(self, index)

    # ---#
    def drop(self):
        """
    ---------------------------------------------------------------------------
    Drops the model from the Vertica database.
        """
        for step in self.steps:
            step[1].drop()

    # ---#
    def fit(
        self,
        input_relation: (str, vDataFrame),
        X: list,
        y: str = "",
        test_relation: (str, vDataFrame) = "",
    ):
        """
    ---------------------------------------------------------------------------
    Trains the model.

    Parameters
    ----------
    input_relation: str/vDataFrame
        Training relation.
    X: list
        List of the predictors.
    y: str, optional
        Response column.
    test_relation: str/vDataFrame, optional
        Relation used to test the model.

    Returns
    -------
    object
        model
        """
        if isinstance(X, str):
            X = [X]
        if isinstance(input_relation, str):
            vdf = vdf_from_relation(
                relation=input_relation, cursor=self.steps[0][1].cursor
            )
        else:
            vdf = input_relation
        X_new = [elem for elem in X]
        current_vdf = vdf
        for idx, step in enumerate(self.steps):
            if (idx == len(self.steps) - 1) and (y):
                step[1].fit(current_vdf, X_new, y, test_relation)
            else:
                step[1].fit(current_vdf, X_new)
            if idx < len(self.steps) - 1:
                current_vdf = step[1].transform(current_vdf, X_new)
                X_new = step[1].get_names(X=X)
        self.input_relation = self.steps[0][1].input_relation
        self.X = [column for column in self.steps[0][1].X]
        try:
            self.y = self.steps[-1][1].y
            self.test_relation = self.steps[-1][1].test_relation
        except:
            pass

    # ---#
    def get_params(self):
        """
    ---------------------------------------------------------------------------
    Returns the models Parameters.

    Returns
    -------
    dict
        models parameters
        """
        params = {}
        for step in self.steps:
            params[step[0]] = step[1].get_params()
        return params

    # ---#
    def predict(
        self, vdf: (str, vDataFrame) = None, X: list = [], name: str = "estimator",
    ):
        """
    ---------------------------------------------------------------------------
    Applies the model on a vDataFrame.

    Parameters
    ----------
    vdf: str/vDataFrame, optional
        Input vDataFrame. You can also specify a customized relation, 
        but you must enclose it with an alias. For example "(SELECT 1) x" is 
        correct whereas "(SELECT 1)" and "SELECT 1" are incorrect.
    X: list, optional
        List of the input vcolumns.
    name: str, optional
        Name of the added vcolumn.

    Returns
    -------
    vDataFrame
        object result of the model transformation.
        """
        if isinstance(X, str):
            X = [X]
        try:
            self.steps[-1][1].predict
        except:
            raise ModelError(
                "The last estimator of the Pipeline has no 'predict' method."
            )
        if not (vdf):
            vdf = self.input_relation
        if isinstance(vdf, str):
            vdf = vdf_from_relation(relation=vdf, cursor=self.steps[0][1].cursor)
        X_new, X_all = [elem for elem in X], []
        current_vdf = vdf
        for idx, step in enumerate(self.steps):
            if idx == len(self.steps) - 1:
                try:
                    current_vdf = step[1].predict(
                        current_vdf, X_new, name=name, inplace=False
                    )
                except:
                    current_vdf = step[1].predict(current_vdf, X_new, name=name)
            else:
                current_vdf = step[1].transform(current_vdf, X_new)
                X_new = step[1].get_names(X=X)
                X_all += X_new
        return current_vdf[vdf.get_columns() + [name]]

    # ---#
    def report(self):
        """
    ---------------------------------------------------------------------------
    Computes a regression/classification report using multiple metrics to evaluate 
    the model depending on its type. 

    Returns
    -------
    tablesample
        An object containing the result. For more information, see
        utilities.tablesample.
        """
        if isinstance(self.steps[-1][1], Regressor):
            return self.steps[-1][1].regression_report()
        else:
            return self.steps[-1][1].classification_report()

    # ---#
    def score(self, method: str = ""):
        """
    ---------------------------------------------------------------------------
    Computes the model score.

    Parameters
    ----------
    method: str, optional
        The method to use to compute the score.
        Depends on the final estimator type (classification or regression).

    Returns
    -------
    float
        score
        """
        if not (method):
            if isinstance(self.steps[-1][1], Regressor):
                method = "r2"
            else:
                method = "accuracy"
        return self.steps[-1][1].score(method)

    # ---#
    def transform(self, vdf: (str, vDataFrame) = None, X: list = []):
        """
    ---------------------------------------------------------------------------
    Applies the model on a vDataFrame.

    Parameters
    ----------
    vdf: str/vDataFrame, optional
        Input vDataFrame. You can also specify a customized relation, 
        but you must enclose it with an alias. For example "(SELECT 1) x" is 
        correct whereas "(SELECT 1)" and "SELECT 1" are incorrect.
    X: list, optional
        List of the input vcolumns.

    Returns
    -------
    vDataFrame
        object result of the model transformation.
        """
        if isinstance(X, str):
            X = [X]
        try:
            self.steps[-1][1].transform
        except:
            raise ModelError(
                "The last estimator of the Pipeline has no 'transform' method."
            )
        if not (vdf):
            vdf = self.input_relation
        if isinstance(vdf, str):
            vdf = vdf_from_relation(relation=vdf, cursor=self.steps[0][1].cursor)
        X_new, X_all = [elem for elem in X], []
        current_vdf = vdf
        for idx, step in enumerate(self.steps):
            current_vdf = step[1].transform(current_vdf, X_new)
            X_new = step[1].get_names(X=X)
            X_all += X_new
        return current_vdf

    # ---#
    def inverse_transform(self, vdf: (str, vDataFrame) = None, X: list = []):
        """
    ---------------------------------------------------------------------------
    Applies the inverse model transformation on a vDataFrame.

    Parameters
    ----------
    vdf: str/vDataFrame, optional
        Input vDataFrame. You can also specify a customized relation, 
        but you must enclose it with an alias. For example "(SELECT 1) x" is 
        correct whereas "(SELECT 1)" and "SELECT 1" are incorrect.
    X: list, optional
        List of the input vcolumns.

    Returns
    -------
    vDataFrame
        object result of the model inverse transformation.
        """
        if isinstance(X, str):
            X = [X]
        try:
            for idx in range(len(self.steps)):
                self.steps[idx][1].inverse_transform
        except:
            raise ModelError(
                "The estimator [{}] of the Pipeline has no 'inverse_transform' method.".format(
                    idx
                )
            )
        if not (vdf):
            vdf = self.input_relation
        if isinstance(vdf, str):
            vdf = vdf_from_relation(relation=vdf, cursor=self.steps[0][1].cursor)
        X_new, X_all = [elem for elem in X], []
        current_vdf = vdf
        for idx in range(1, len(self.steps) + 1):
            step = self.steps[-idx]
            current_vdf = step[1].inverse_transform(current_vdf, X_new)
            X_new = step[1].get_names(inverse=True, X=X)
            X_all += X_new
        return current_vdf

    # ---#
    def set_cursor(self, cursor):
        """
    ---------------------------------------------------------------------------
    Sets a new database cursor. It can be very usefull if the connection to the DB is 
    lost.

    Parameters
    ----------
    cursor: DBcursor
        New cursor.

    Returns
    -------
    model
        self
        """
        for step in self.steps:
            step[1].set_cursor(cursor)
        return self

    # ---#
    def set_params(self, parameters: dict = {}):
        """
    ---------------------------------------------------------------------------
    Sets the parameters of the model.

    Parameters
    ----------
    parameters: dict, optional
        New parameters. It must be a dictionary with as keys the Pipeline names
        and as value the parameters dictionary.
        """
        for param in parameters:
            for step in self.steps:
                if param.lower() == step[0].lower():
                    step[1].set_params(parameters[param])

    # ---#
    def to_sklearn(self):
        """
    ---------------------------------------------------------------------------
    Converts the Vertica Model to sklearn model.

    Returns
    -------
    object
        sklearn model.
        """
        import sklearn.pipeline as skp

        steps = []
        for step in self.steps:
            steps += [(step[0], step[1].to_sklearn())]
        return skp.Pipeline(steps)
