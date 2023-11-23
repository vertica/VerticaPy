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
import copy

from typing import Literal, Optional

import verticapy._config.config as conf
from verticapy._typing import NoneType, SQLColumns, SQLRelation
from verticapy._utils._sql._format import format_type
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy.errors import ModelError

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

from verticapy.machine_learning.vertica.base import Regressor, VerticaModel

"""
General Class.
"""


class Pipeline:
    """
    Creates a Pipeline object, which sequentially
    applies a list of transforms and a final estimator.
    The intermediate steps must  implement a transform
    method.

    Parameters
    ----------
    steps: list
        List of (name, transform)  tuples (implementing
        fit/transform) that  are chained, in  the order
        in which they are chained, where the last object
        is an estimator.
    overwrite_model: bool, optional
        If set to True, training a model in the pipeline with the same
        name as an existing model overwrites the existing model.
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
    def __init__(self, steps: list, overwrite_model: bool = False) -> None:
        self.steps = []
        self.overwrite_model = overwrite_model
        for idx, s in enumerate(steps):
            if len(s) != 2:
                raise ValueError(
                    "The steps of the Pipeline must be composed of 2 elements "
                    f"(name, transform). Found {len(s)}."
                )
            elif not isinstance(s[0], str):
                raise ValueError(
                    "The steps 'name' of the Pipeline must be of "
                    f"type str. Found {type(s[0])}."
                )
            elif idx < len(steps) - 1 and (
                not hasattr(s[1], "transform") or not hasattr(s[1], "fit")
            ):
                raise AttributeError(
                    "The estimators of the Pipeline must have a "
                    "'transform' and a 'fit' method."
                )
            elif not hasattr(s[1], "fit"):
                raise ValueError(
                    "The last estimator of the Pipeline must have a " "'fit' method."
                )
            self.steps += [s]

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
        Returns the model's Parameters.

        Returns
        -------
        dict
            model's parameters.
        """
        params = {}
        for step in self.steps:
            params[step[0]] = step[1].get_params()
        return params

    def set_params(self, parameters: Optional[dict[dict]] = None, **kwargs) -> None:
        """
        Sets the parameters of the model.

        Parameters
        ----------
        parameters: dict, optional
            New parameters.  It must be a  dictionary with
            the  Pipeline names as keys and the parameter
            dictionary as values.
        **kwargs
            New parameters can also be passed as arguments.
            Example: set_params(pipeline1 = dict1,
                                pipeline2 = dict2)
        """
        parameters = format_type(parameters, dtype=dict)
        for param in {**parameters, **kwargs}:
            for step in self.steps:
                if param.lower() == step[0].lower():
                    step[1].set_params(parameters[param])

    # Model Fitting Method.

    def fit(
        self,
        input_relation: SQLRelation,
        X: list,
        y: Optional[str] = None,
        test_relation: SQLRelation = "",
        return_report: bool = False,
    ) -> None:
        """
        Trains the model.

        Parameters
        ----------
        input_relation: SQLRelation
            Training relation.
        X: list
            List of the predictors.
        y: str, optional
            Response column.
        test_relation: SQLRelation, optional
            Relation used to test the model.

        Returns
        -------
        object
            model.
        """
        X = format_type(X, dtype=list)
        if isinstance(input_relation, str):
            vdf = vDataFrame(input_relation)
        else:
            vdf = input_relation
        if self.overwrite_model:
            self.drop()
        X_new = copy.deepcopy(X)
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
        if hasattr(self.steps[-1][1], "y"):
            self.y = self.steps[-1][1].y
        if hasattr(self.steps[-1][1], "test_relation"):
            self.test_relation = self.steps[-1][1].test_relation

    # Model Evaluation Methods.

    def report(self) -> TableSample:
        """
        Computes a regression/classification report using
        multiple metrics to  evaluate the model depending
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

    def score(self, metric: Optional[str] = None) -> float:
        """
        Computes the model score.

        Parameters
        ----------
        metric: str, optional
            The metric used to compute the score.
            Depends  on  the  final estimator type
            (classification or regression).

        Returns
        -------
        float
            score.
        """
        if isinstance(metric, NoneType):
            if isinstance(self.steps[-1][1], Regressor):
                metric = "r2"
            else:
                metric = "accuracy"
        return self.steps[-1][1].score(metric=metric)

    # Prediction / Transformation Methods.

    def predict(
        self,
        vdf: SQLRelation = None,
        X: Optional[SQLColumns] = None,
        name: str = "estimator",
    ) -> vDataFrame:
        """
        Applies the model on a vDataFrame.

        Parameters
        ----------
        vdf: SQLRelation, optional
            Input  vDataFrame.  You  can  also  specify  a
            customized  relation,  but  you  must  enclose
            it with an alias.  For example: "(SELECT 1) x"
            is valid whereas "(SELECT 1)" and "SELECT 1"
            are invalid.
        X: SQLColumns, optional
            List of the input vDataColumns.
        name: str, optional
            Name of the added vDataColumn.

        Returns
        -------
        vDataFrame
            object result of the model transformation.
        """
        X = format_type(X, dtype=list)
        if not hasattr(self.steps[-1][1], "predict"):
            raise ModelError(
                "The last estimator of the Pipeline has no 'predict' method."
            )
        if not vdf:
            vdf = self.input_relation
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)
        X_new, X_all = copy.deepcopy(X), []
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

    def transform(
        self, vdf: SQLRelation = None, X: Optional[SQLColumns] = None
    ) -> vDataFrame:
        """
        Applies the model on a vDataFrame.

        Parameters
        ----------
        vdf: SQLRelation, optional
            Input  vDataFrame.  You  can  also  specify  a
            customized  relation,  but  you  must  enclose
            it with an alias. For  example: "(SELECT 1) x"
            is valid whereas "(SELECT 1)" and "SELECT 1"
            are invalid.
        X: SQLColumns, optional
            List of the input vDataColumns.

        Returns
        -------
        vDataFrame
            object result of the model transformation.
        """
        X = format_type(X, dtype=list)
        if not hasattr(self.steps[-1][1], "transform"):
            raise ModelError(
                "The last estimator of the Pipeline has no 'transform' method."
            )
        if not vdf:
            vdf = self.input_relation
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)
        X_new, X_all = copy.deepcopy(X), []
        current_vdf = vdf
        for step in self.steps:
            current_vdf = step[1].transform(current_vdf, X_new)
            X_new = step[1]._get_names(X=X)
            X_all += X_new
        return current_vdf

    def inverse_transform(
        self, vdf: SQLRelation = None, X: Optional[SQLColumns] = None
    ) -> vDataFrame:
        """
        Applies  the  inverse model transformation  on  a
        vDataFrame.

        Parameters
        ----------
        vdf: SQLRelation, optional
            Input  vDataFrame.  You  can  also  specify  a
            customized  relation,  but  you  must  enclose
            it with an alias. For  example: "(SELECT 1) x"
            is valid whereas "(SELECT 1)" and "SELECT 1"
            are invalid.
        X: SQLColumns, optional
            List of the input vDataColumns.

        Returns
        -------
        vDataFrame
            object result of the model inverse transformation.
        """
        X = format_type(X, dtype=list)
        for idx in range(len(self.steps)):
            if not hasattr(self.steps[idx][1], "inverse_transform"):
                raise ModelError(
                    f"The estimator [{idx}] of the Pipeline has "
                    "no 'inverse_transform' method."
                )
        if not vdf:
            vdf = self.input_relation
        if isinstance(vdf, str):
            vdf = vDataFrame(vdf)
        X_new, X_all = copy.deepcopy(X), []
        current_vdf = vdf
        for idx in range(1, len(self.steps) + 1):
            step = self.steps[-idx]
            current_vdf = step[1].inverse_transform(current_vdf, X_new)
            X_new = step[1]._get_names(inverse=True, X=X)
            X_all += X_new
        return current_vdf
