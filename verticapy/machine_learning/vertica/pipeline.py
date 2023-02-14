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

#
#
# Modules
#
# VerticaPy Modules
from verticapy._utils._collect import save_verticapy_logs
from verticapy.core.vdataframe.vdataframe import vDataFrame
from verticapy.sql.read import vDataFrameSQL
from verticapy.errors import ParameterError, ModelError
from verticapy.machine_learning.vertica.vmodel import Regressor
from verticapy._config.config import OPTIONS

# Standard Python Modules
from typing import Union


class Pipeline:
    """
Creates a Pipeline object. Sequentially apply a list of transforms and a 
final estimator. The intermediate steps must implement a transform method.

Parameters
----------
steps: list
    List of (name, transform) tuples (implementing fit/transform) that are chained, 
    in the order in which they are chained, with the last object an estimator.
	"""

    @save_verticapy_logs
    def __init__(self, steps: list):
        self.type = "Pipeline"
        self.steps = []
        for idx, s in enumerate(steps):
            if len(s) != 2:
                raise ParameterError(
                    "The steps of the Pipeline must be composed of 2 elements "
                    f"(name, transform). Found {len(s)}."
                )
            elif not (isinstance(s[0], str)):
                raise ParameterError(
                    "The steps 'name' of the Pipeline must be of "
                    f"type str. Found {type(s[0])}."
                )
            else:
                try:
                    if idx < len(steps) - 1:
                        s[1].transform
                    s[1].fit
                except:
                    if idx < len(steps) - 1:
                        raise ParameterError(
                            "The estimators of the Pipeline must have a "
                            "'transform' and a 'fit' method."
                        )
                    else:
                        raise ParameterError(
                            "The last estimator of the Pipeline must have a "
                            "'fit' method."
                        )
            self.steps += [s]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.steps[index]
        elif isinstance(index, int):
            return self.steps[index][1]
        else:
            return getattr(self, index)

    def drop(self):
        """
    Drops the model from the Vertica database.
        """
        for step in self.steps:
            step[1].drop()

    def fit(
        self,
        input_relation: Union[str, vDataFrame],
        X: list,
        y: str = "",
        test_relation: Union[str, vDataFrame] = "",
    ):
        """
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
            vdf = vDataFrameSQL(relation=input_relation)
        else:
            vdf = input_relation
        if OPTIONS["overwrite_model"]:
            self.drop()
        else:
            does_model_exist(name=self.name, raise_error=True)
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
        return self

    def get_params(self):
        """
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

    def predict(
        self, vdf: Union[str, vDataFrame] = None, X: list = [], name: str = "estimator",
    ):
        """
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
            vdf = vDataFrameSQL(relation=vdf)
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

    def report(self):
        """
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

    def score(self, method: str = ""):
        """
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

    def transform(self, vdf: Union[str, vDataFrame] = None, X: list = []):
        """
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
            vdf = vDataFrameSQL(relation=vdf)
        X_new, X_all = [elem for elem in X], []
        current_vdf = vdf
        for idx, step in enumerate(self.steps):
            current_vdf = step[1].transform(current_vdf, X_new)
            X_new = step[1].get_names(X=X)
            X_all += X_new
        return current_vdf

    def inverse_transform(self, vdf: Union[str, vDataFrame] = None, X: list = []):
        """
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
                f"The estimator [{idx}] of the Pipeline has "
                "no 'inverse_transform' method."
            )
        if not (vdf):
            vdf = self.input_relation
        if isinstance(vdf, str):
            vdf = vDataFrameSQL(relation=vdf)
        X_new, X_all = [elem for elem in X], []
        current_vdf = vdf
        for idx in range(1, len(self.steps) + 1):
            step = self.steps[-idx]
            current_vdf = step[1].inverse_transform(current_vdf, X_new)
            X_new = step[1].get_names(inverse=True, X=X)
            X_all += X_new
        return current_vdf

    def set_params(self, parameters: dict = {}):
        """
    Sets the parameters of the model.

    Parameters
    ----------
    parameters: dict, optional
        New parameters. It must be a dictionary with as keys the Pipeline 
        names and as value the parameters dictionary.
        """
        for param in parameters:
            for step in self.steps:
                if param.lower() == step[0].lower():
                    step[1].set_params(parameters[param])

    def to_python(
        self,
        name: str = "predict",
        return_proba: bool = False,
        return_distance_clusters: bool = False,
        return_str: bool = False,
    ):
        """
    Returns the Python code needed to deploy the pipeline without using 
    built-in Vertica functions.

    Parameters
    ----------
    name: str, optional
        Function Name.
    return_proba: bool, optional
        If set to True and the model is a classifier, the function 
        returns the model probabilities.
    return_distance_clusters: bool, optional
        If set to True and the model type is KPrototypes / KMeans 
        or NearestCentroids, the function returns the model clusters 
        distances.
    return_str: bool, optional
        If set to True, the function str will be returned.


    Returns
    -------
    str / func
        Python function
        """
        if not (return_str):
            func = self.to_python(
                name=name,
                return_proba=return_proba,
                return_distance_clusters=return_distance_clusters,
                return_str=True,
            )
            _locals = locals()
            exec(func, globals(), _locals)
            return _locals[name]
        str_representation = f"def {name}(X):\n"
        final_function = "X"
        for idx, step in enumerate(self.steps):
            str_representation += "\t"
            str_representation += (
                step[1]
                .to_python(
                    name=step[0],
                    return_proba=return_proba,
                    return_distance_clusters=return_distance_clusters,
                    return_str=True,
                )
                .replace("\n", "\n\t")
            )
            str_representation += "\n"
            final_function = f"{step[0]}({final_function})"
        str_representation += f"\treturn {final_function}"
        return str_representation
