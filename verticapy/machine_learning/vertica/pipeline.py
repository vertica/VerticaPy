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
from typing import Any, Literal, Union

import verticapy._config.config as conf
from verticapy._typing import SQLRelation
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy.errors import ParameterError, ModelError

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

from verticapy.machine_learning.vertica.base import Regressor, VerticaModel

"""
General Class.
"""


class Pipeline:
    """
    Creates a Pipeline object. Sequentially apply a 
    list of transforms and a final estimator. The 
    intermediate steps must implement a transform 
    method.

    Parameters
    ----------
    steps: list
        List of (name, transform) tuples (implementing 
        fit/transform) that are chained, in the order 
        in which they are chained, with the last object 
        an estimator.
	"""

    # Properties.

    @property
    def _is_native(self) -> Literal[False]:
        return False

    @property
    def _vertica_fit_sql(self) -> Literal[""]:
        return ""

    @property
    def _vertica_predict_sql(self) -> Literal[""]:
        return ""

    @property
    def _model_category(self) -> Literal[""]:
        return ""

    @property
    def _model_subcategory(self) -> Literal[""]:
        return ""

    @property
    def _model_type(self) -> Literal["Pipeline"]:
        return "Pipeline"

    @property
    def _attributes(self) -> None:
        raise NotImplementedError

    # System & Special Methods.

    @save_verticapy_logs
    def __init__(self, steps: list) -> None:
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
        return None

    def __getitem__(self, index) -> VerticaModel:
        if isinstance(index, slice):
            return self.steps[index]
        elif isinstance(index, int):
            return self.steps[index][1]
        else:
            return getattr(self, index)

    def drop(self) -> None:
        """
        Drops the model from the Vertica database.
        """
        for step in self.steps:
            step[1].drop()

    # Parameters Methods.

    def get_params(self) -> dict[dict]:
        """
        Returns the models Parameters.

        Returns
        -------
        dict
            models parameters.
        """
        params = {}
        for step in self.steps:
            params[step[0]] = step[1].get_params()
        return params

    def set_params(self, parameters: dict[dict] = {}, **kwds) -> None:
        """
        Sets the parameters of the model.

        Parameters
        ----------
        parameters: dict, optional
            New parameters. It must be a dictionary with as keys the 
            Pipeline names and as value the parameters dictionary.
        **kwds
            New parameters can also be passed as arguments
            Example: set_params(pipeline1 = dict1, 
                                pipeline2 = dict2)
        """
        for param in {**parameters, **kwds}:
            for step in self.steps:
                if param.lower() == step[0].lower():
                    step[1].set_params(parameters[param])
        return None

    # Model Fitting Method.

    def fit(
        self,
        input_relation: SQLRelation,
        X: list,
        y: str = "",
        test_relation: SQLRelation = "",
    ) -> None:
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
            model.
        """
        if isinstance(X, str):
            X = [X]
        if isinstance(input_relation, str):
            vdf = vDataFrame(input_relation)
        else:
            vdf = input_relation
        if conf.get_option("overwrite_model"):
            self.drop()
        X_new = [elem for elem in X]
        current_vdf = vdf
        for idx, step in enumerate(self.steps):
            if (idx == len(self.steps) - 1) and (y):
                step[1].fit(current_vdf, X_new, y, test_relation)
            else:
                step[1].fit(current_vdf, X_new)
            if idx < len(self.steps) - 1:
                current_vdf = step[1].transform(current_vdf, X_new)
                X_new = step[1]._get_names(X=X)
        self.input_relation = self.steps[0][1].input_relation
        self.X = [column for column in self.steps[0][1].X]
        try:
            self.y = self.steps[-1][1].y
            self.test_relation = self.steps[-1][1].test_relation
        except:
            pass
        return None

    # Model Evaluation Methods.

    def report(self) -> TableSample:
        """
        Computes a regression/classification report using 
        multiple metrics to evaluate the model depending 
        on its type. 

        Returns
        -------
        TableSample
            report.
        """
        if isinstance(self.steps[-1][1], Regressor):
            return self.steps[-1][1].regression_report()
        else:
            return self.steps[-1][1].classification_report()

    def score(self, method: str = "") -> float:
        """
        Computes the model score.

        Parameters
        ----------
        method: str, optional
            The method to use to compute the score.
            Depends on the final estimator type 
            (classification or regression).

        Returns
        -------
        float
            score.
        """
        if not (method):
            if isinstance(self.steps[-1][1], Regressor):
                method = "r2"
            else:
                method = "accuracy"
        return self.steps[-1][1].score(method)

    # Prediction / Transformation Methods.

    def predict(
        self, vdf: SQLRelation = None, X: list = [], name: str = "estimator",
    ) -> vDataFrame:
        """
        Applies the model on a vDataFrame.

        Parameters
        ----------
        vdf: str/vDataFrame, optional
            Input vDataFrame. You can also specify a 
            customized relation, but you must enclose 
            it with an alias. For example "(SELECT 1) x" 
            is correct whereas "(SELECT 1)" and "SELECT 1" 
            are incorrect.
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
            vdf = vDataFrame(vdf)
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
                X_new = step[1]._get_names(X=X)
                X_all += X_new
        return current_vdf[vdf.get_columns() + [name]]

    def transform(self, vdf: SQLRelation = None, X: list = []) -> vDataFrame:
        """
        Applies the model on a vDataFrame.

        Parameters
        ----------
        vdf: str/vDataFrame, optional
            Input vDataFrame. You can also specify a 
            customized relation, but you must enclose 
            it with an alias. For example "(SELECT 1) x" 
            is correct whereas "(SELECT 1)" and "SELECT 1" 
            are incorrect.
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
            vdf = vDataFrame(vdf)
        X_new, X_all = [elem for elem in X], []
        current_vdf = vdf
        for idx, step in enumerate(self.steps):
            current_vdf = step[1].transform(current_vdf, X_new)
            X_new = step[1]._get_names(X=X)
            X_all += X_new
        return current_vdf

    def inverse_transform(self, vdf: SQLRelation = None, X: list = []) -> vDataFrame:
        """
        Applies the inverse model transformation on a vDataFrame.

        Parameters
        ----------
        vdf: str/vDataFrame, optional
            Input vDataFrame. You can also specify a customized 
            relation, but you must enclose it with an alias. 
            For example "(SELECT 1) x" is correct whereas 
            "(SELECT 1)" and "SELECT 1" are incorrect.
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
            vdf = vDataFrame(vdf)
        X_new, X_all = [elem for elem in X], []
        current_vdf = vdf
        for idx in range(1, len(self.steps) + 1):
            step = self.steps[-idx]
            current_vdf = step[1].inverse_transform(current_vdf, X_new)
            X_new = step[1]._get_names(inverse=True, X=X)
            X_all += X_new
        return current_vdf
