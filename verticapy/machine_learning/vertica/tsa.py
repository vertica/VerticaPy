"""
Copyright  (c)  2018-2023 Open Text  or  one  of its
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
from abc import abstractmethod
import copy
from typing import Literal, Optional, Union

import numpy as np

from verticapy._typing import (
    PlottingObject,
    PythonNumber,
    NoneType,
    SQLRelation,
)
from verticapy._utils._gen import gen_name, gen_tmp_name
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import (
    clean_query,
    quote_ident,
    schema_relation,
)
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._vertica_version import (
    check_minimum_version,
)

from verticapy.core.vdataframe.base import TableSample, vDataFrame

from verticapy.machine_learning.vertica.base import VerticaModel

from verticapy.sql.drop import drop

"""
General Classes.
"""


class TimeSeriesModelBase(VerticaModel):
    """
    Base Class for Vertica Time Series Models.
    """

    # Properties.

    @property
    def _model_category(self) -> Literal["TIMESERIES"]:
        return "TIMESERIES"

    @property
    def _attributes(self) -> list[str]:
        common_params = [
            "mse_",
        ]
        if self._model_type == "ARIMA":
            return [
                "phi_",
                "theta_",
                "mean_",
            ] + common_params
        elif self._model_type == "AR":
            return [
                "phi_",
                "intercept_",
                "features_importance_",
            ] + common_params
        else:
            return [
                "theta_",
                "mu_",
                "mean_",
            ] + common_params

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        coefficients = self.get_vertica_attributes("coefficients")
        i = 1
        if "p" in self.parameters:
            p = self.parameters["p"]
            self.intercept_ = coefficients["value"][0]
        else:
            self.mean_ = self.get_vertica_attributes("mean")["mean"][0]
            if "order" in self.parameters:
                p = self.parameters["order"][0]
                i = 0
            else:
                p = 0
            self.mu_ = coefficients["value"][0]
        self.phi_ = np.array(coefficients["value"][i : p + i])
        self.theta_ = np.array(coefficients["value"][p + i :])
        try:
            self.mse_ = self.get_vertica_attributes("mean_squared_error")[
                "mean_squared_error"
            ][0]
        except:
            self.mse_ = None

    # System & Special Methods.

    @abstractmethod
    def __init__(self, name: str, overwrite_model: bool = False) -> None:
        """Must be overridden in the child class"""
        super().__init__(name, overwrite_model)

    # Model Fitting Method.

    def fit(
        self,
        input_relation: SQLRelation,
        ts: str,
        y: str,
        test_relation: SQLRelation = "",
        return_report: bool = False,
    ) -> Optional[str]:
        """
        Trains the model.

        Parameters
        ----------
        input_relation: SQLRelation
            Training relation.
        ts: str
            TS (Time Series)  vDataColumn used to order
            the data.  The vDataColumn type must be  date
            (date, datetime, timestamp...) or numerical.
        y: str
            Response column.
        test_relation: SQLRelation, optional
            Relation used to test the model.
        return_report: bool, optional
            [For native models]
            When set to True, the model summary
            will be returned. Otherwise, it will
            be printed.

        Returns
        -------
        str
            model's summary.
        """

        # Initialization
        if self.overwrite_model:
            self.drop()
        else:
            self._is_already_stored(raise_error=True)
        self.ts = quote_ident(ts)
        self.y = quote_ident(y)
        tmp_view = False
        if isinstance(input_relation, vDataFrame) and self._is_native:
            tmp_view = True
            if isinstance(input_relation, vDataFrame):
                self.input_relation = input_relation.current_relation()
            else:
                self.input_relation = input_relation
            relation = gen_tmp_name(
                schema=schema_relation(self.model_name)[0], name="view"
            )
            _executeSQL(
                query=f"""
                    CREATE OR REPLACE VIEW {relation} AS 
                        SELECT 
                            /*+LABEL('learn.VerticaModel.fit')*/ 
                            {self.ts}, {self.y}
                        FROM {self.input_relation}""",
                title="Creating a temporary view to fit the model.",
            )
        else:
            self.input_relation = input_relation
            relation = input_relation
        if isinstance(test_relation, vDataFrame):
            self.test_relation = test_relation.current_relation()
        elif test_relation:
            self.test_relation = test_relation
        else:
            self.test_relation = self.input_relation
        # Fitting
        if self._is_native:
            parameters = self._get_vertica_param_dict()
            if "order" in parameters:
                parameters["p"] = parameters["order"][0]
                parameters["q"] = parameters["order"][-1]
                if len(parameters["order"]) == 3:
                    parameters["d"] = parameters["order"][1]
                del parameters["order"]
            query = f"""
                SELECT 
                    /*+LABEL('learn.VerticaModel.fit')*/ 
                    {self._vertica_fit_sql}
                    ('{self.model_name}', 
                     '{relation}',
                     '{self.y}',
                     '{self.ts}' 
                     USING PARAMETERS 
                     {', '.join([f"{p} = {parameters[p]}" for p in parameters])})"""
            try:
                _executeSQL(query, title="Fitting the model.")
            finally:
                if tmp_view:
                    drop(relation, method="view")
        self._compute_attributes()
        if self._is_native:
            report = self.summarize()
            if return_report:
                return report
            print(report)
        return None

    # I/O Methods.

    def deploySQL(
        self,
        ts: Optional[str] = None,
        y: Optional[str] = None,
        start: Optional[int] = None,
        npredictions: int = 10,
        output_standard_errors: bool = False,
    ) -> str:
        """
        Returns the SQL code needed to deploy the model.

        Parameters
        ----------
        ts: str, optional
            TS (Time Series)  vDataColumn used to order
            the data.  The vDataColumn type must be  date
            (date, datetime, timestamp...) or numerical.
        y: str, optional
            Response column.
        start: int, optional
            The behavior of the start parameter and its
            range of accepted values depends on whether
            you provide a timeseries-column (ts):

              - No provided timeseries-column: start must
                be an integer greater or equal to 0, where
                zero indicates to start prediction at the
                end of the in-sample data. If start is a
                positive value, the function predicts the
                values between the end of the in-sample
                data and the start index, and then uses the
                predicted values as time series inputs for
                the subsequent npredictions.
              - timeseries-column provided: start must be an
                integer greater or equal to 1 and identifies
                the index (row) of the timeseries-column at
                which to begin prediction. If the start index
                is greater than the number of rows, N, in the
                input data, the function predicts the values
                between N and start and uses the predicted
                values as time series inputs for the subsequent
                npredictions.

            Default:

              - No provided timeseries-column: prediction begins
                from the end of the in-sample data.
              - timeseries-column provided: prediction begins from
                the end of the provided input data.
        npredictions: int, optional
            Integer greater or equal to 1, the number of predicted
            timesteps.
        output_standard_errors
            Boolean,  whether to return estimates  of the standard
            error of each prediction.

        Returns
        -------
        str
            the SQL code needed to deploy the model.
        """
        if self._vertica_predict_sql:
            # Initialization
            if isinstance(ts, NoneType):
                ts = ""
            else:
                ts = "ORDER BY " + quote_ident(ts)
            if isinstance(y, NoneType):
                y = ""
            else:
                y = quote_ident(y)
            if isinstance(start, NoneType):
                start = ""
            else:
                start = f"start = {start},"
            if output_standard_errors:
                output_standard_errors = (
                    f", output_standard_errors = {output_standard_errors}"
                )
            else:
                output_standard_errors = ""
            # Deployment
            sql = f"""
                {self._vertica_predict_sql}({y}
                                            USING PARAMETERS 
                                            model_name = '{self.model_name}',
                                            add_mean = False,
                                            {start}
                                            npredictions = {npredictions}
                                            {output_standard_errors}) 
                                            OVER ({ts})"""
            return clean_query(sql)
        else:
            raise AttributeError(
                f"Method 'deploySQL' does not exist for {self._model_type} models."
            )

    # Prediction / Transformation Methods.

    def predict(
        self,
        vdf: Optional[SQLRelation] = None,
        ts: Optional[str] = None,
        y: Optional[str] = None,
        start: Optional[int] = None,
        npredictions: int = 10,
        output_standard_errors: bool = False,
    ) -> vDataFrame:
        """
        Predicts using the input relation.

        Parameters
        ----------
        vdf: SQLRelation
            Object  used to run  the prediction.  You can
            also  specify a  customized  relation,  but you
            must  enclose  it with an alias.  For  example,
            "(SELECT 1) x" is valid, whereas "(SELECT 1)"
            and "SELECT 1" are invalid.
        ts: str, optional
            TS (Time Series)  vDataColumn used to order
            the data.  The vDataColumn type must be  date
            (date, datetime, timestamp...) or numerical.
        y: str, optional
            Response column.
        start: int, optional
            The behavior of the start parameter and its
            range of accepted values depends on whether
            you provide a timeseries-column (ts):

              - No provided timeseries-column: start must
                be an integer greater or equal to 0, where
                zero indicates to start prediction at the
                end of the in-sample data. If start is a
                positive value, the function predicts the
                values between the end of the in-sample
                data and the start index, and then uses the
                predicted values as time series inputs for
                the subsequent npredictions.
              - timeseries-column provided: start must be an
                integer greater or equal to 1 and identifies
                the index (row) of the timeseries-column at
                which to begin prediction. If the start index
                is greater than the number of rows, N, in the
                input data, the function predicts the values
                between N and start and uses the predicted
                values as time series inputs for the subsequent
                npredictions.

            Default:

              - No provided timeseries-column: prediction begins
                from the end of the in-sample data.
              - timeseries-column provided: prediction begins from
                the end of the provided input data.
        npredictions: int, optional
            Integer greater or equal to 1, the number of predicted
            timesteps.
        output_standard_errors
            Boolean,  whether to return estimates  of the standard
            error of each prediction.

        Returns
        -------
        vDataFrame
            a new object.
        """
        if self._model_type in (
            "AR",
            "MA",
        ):
            if isinstance(vdf, NoneType):
                vdf = self.input_relation
            if isinstance(ts, NoneType):
                ts = self.ts
            if isinstance(y, NoneType):
                y = self.y
        sql = "SELECT " + self.deploySQL(
            ts=ts,
            y=y,
            start=start,
            npredictions=npredictions,
            output_standard_errors=output_standard_errors,
        )
        if not (isinstance(vdf, NoneType)):
            sql += f" FROM {vdf}"
        return vDataFrame(sql)

    # Plotting Methods.

    def plot(
        self,
        vdf: Optional[SQLRelation] = None,
        ts: Optional[str] = None,
        y: Optional[str] = None,
        start: Optional[int] = None,
        npredictions: int = 10,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the model.

        Parameters
        ----------
        vdf: SQLRelation
            Object  used to run  the prediction.  You can
            also  specify a  customized  relation,  but you
            must  enclose  it with an alias.  For  example,
            "(SELECT 1) x" is valid, whereas "(SELECT 1)"
            and "SELECT 1" are invalid.
        ts: str, optional
            TS (Time Series)  vDataColumn used to order
            the data.  The vDataColumn type must be  date
            (date, datetime, timestamp...) or numerical.
        y: str, optional
            Response column.
        start: int, optional
            The behavior of the start parameter and its
            range of accepted values depends on whether
            you provide a timeseries-column (ts):

              - No provided timeseries-column: start must
                be an integer greater or equal to 0, where
                zero indicates to start prediction at the
                end of the in-sample data. If start is a
                positive value, the function predicts the
                values between the end of the in-sample
                data and the start index, and then uses the
                predicted values as time series inputs for
                the subsequent npredictions.
              - timeseries-column provided: start must be an
                integer greater or equal to 1 and identifies
                the index (row) of the timeseries-column at
                which to begin prediction. If the start index
                is greater than the number of rows, N, in the
                input data, the function predicts the values
                between N and start and uses the predicted
                values as time series inputs for the subsequent
                npredictions.

            Default:

              - No provided timeseries-column: prediction begins
                from the end of the in-sample data.
              - timeseries-column provided: prediction begins from
                the end of the provided input data.
        npredictions: int, optional
            Integer greater or equal to 1, the number of predicted
            timesteps.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to pass to the
            Plotting functions.

        Returns
        -------
        object
            Plotting Object.
        """
        dataset_provided = True
        if isinstance(vdf, NoneType):
            dataset_provided = False 
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="TSPlot",
            chart=chart,
            style_kwargs=style_kwargs,
        )
        return vpy_plt.TSPlot(
            vdf=vDataFrame(self.input_relation),
            columns=self.y,
            order_by=self.ts,
            prediction=self.predict(
                vdf=vdf,
                ts=ts,
                y=y,
                start=start,
                npredictions=npredictions,
                output_standard_errors=True,
            ),
            start=start,
            dataset_provided=dataset_provided,
        ).draw(**kwargs)


class ARIMA(TimeSeriesModelBase):
    """
    Creates a inDB ARIMA model.

    Parameters
    ----------
    name: str, optional
        Name of the model. The  model is stored  in the
        database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    order: tuple, optional
        The (p,d,q) order of the model for the autoregressive,
        differences, and moving average components.
    tol: float, optional
        Determines  whether the algorithm has reached
        the specified accuracy result.
    max_iter: int, optional
        Determines  the maximum number of  iterations
        the  algorithm performs before  achieving the
        specified accuracy result.
    init: str, optional
        Initialization method, one of the following:

        - 'zero': Coefficients are initialized to zero.
        - 'hr': Coefficients are initialized using the
            Hannan-Rissanen algorithm.

    missing: str, optional
        Method for handling missing values, one of the
        following strings:

        - 'drop': Missing values are ignored.
        - 'raise': Missing values raise an error.
        - 'zero': Missing values are set to zero.
        - 'linear_interpolation': Missing values are
            replaced by a linearly interpolated value
            based on the nearest valid entries before
            and after the missing value. In cases
            where the first or last values in a
            dataset are missing, the function errors.

    Examples
    ---------

    The following examples provide a basic understanding of usage.
    For more detailed examples, please refer to the
    :ref:`user_guide.machine_learning` or the
    `Examples <https://www.vertica.com/python/examples/>`_
    section on the website.

    ...
    """

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["ARIMA"]:
        return "ARIMA"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_ARIMA"]:
        return "PREDICT_ARIMA"

    @property
    def _model_subcategory(self) -> Literal["TIMESERIES"]:
        return "TIMESERIES"

    @property
    def _model_type(self) -> Literal["ARIMA"]:
        return "ARIMA"

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        order: Union[tuple[int], list[int]] = (0, 0, 0),
        tol: float = 1e-6,
        max_iter: int = 100,
        init: Literal["zero", "hr"] = "zero",
        missing: Literal[
            "drop", "raise", "zero", "linear_interpolation"
        ] = "linear_interpolation",
    ) -> None:
        super().__init__(name, overwrite_model)
        if not (isinstance(order, (tuple, list)) or len(order)) != 3:
            raise ValueError(
                "Parameter 'order' must be a tuple or a list of 3 elements."
            )
        for x in order:
            if not (isinstance(x, int)):
                raise ValueError(
                    "Parameter 'order' must be a tuple or a list of integers."
                )
        self.parameters = {
            "order": order,
            "tol": tol,
            "max_iter": max_iter,
            "init": str(init).lower(),
            "missing": str(missing).lower(),
        }


class ARMA(TimeSeriesModelBase):
    """
    Creates a inDB ARMA model.

    Parameters
    ----------
    name: str, optional
        Name of the model. The  model is stored  in the
        database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    order: tuple, optional
        The (p,q) order of the model for the autoregressive,
        and moving average components.
    tol: float, optional
        Determines  whether the algorithm has reached
        the specified accuracy result.
    max_iter: int, optional
        Determines  the maximum number of  iterations
        the  algorithm performs before  achieving the
        specified accuracy result.
    init: str, optional
        Initialization method, one of the following:

        - 'zero': Coefficients are initialized to zero.
        - 'hr': Coefficients are initialized using the
            Hannan-Rissanen algorithm.

    missing: str, optional
        Method for handling missing values, one of the
        following strings:

        - 'drop': Missing values are ignored.
        - 'raise': Missing values raise an error.
        - 'zero': Missing values are set to zero.
        - 'linear_interpolation': Missing values are
            replaced by a linearly interpolated value
            based on the nearest valid entries before
            and after the missing value. In cases
            where the first or last values in a
            dataset are missing, the function errors.

    Examples
    ---------

    The following examples provide a basic understanding of usage.
    For more detailed examples, please refer to the
    :ref:`user_guide.machine_learning` or the
    `Examples <https://www.vertica.com/python/examples/>`_
    section on the website.

    ...
    """

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["ARIMA"]:
        return "ARIMA"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_ARIMA"]:
        return "PREDICT_ARIMA"

    @property
    def _model_subcategory(self) -> Literal["TIMESERIES"]:
        return "TIMESERIES"

    @property
    def _model_type(self) -> Literal["ARMA"]:
        return "ARMA"

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        order: Union[tuple[int], list[int]] = (0, 0),
        tol: float = 1e-6,
        max_iter: int = 100,
        init: Literal["zero", "hr"] = "zero",
        missing: Literal[
            "drop", "raise", "zero", "linear_interpolation"
        ] = "linear_interpolation",
    ) -> None:
        super().__init__(name, overwrite_model)
        if not (isinstance(order, (tuple, list)) or len(order)) != 3:
            raise ValueError(
                "Parameter 'order' must be a tuple or a list of 2 elements."
            )
        for x in order:
            if not (isinstance(x, int)):
                raise ValueError(
                    "Parameter 'order' must be a tuple or a list of integers."
                )
        self.parameters = {
            "order": order,
            "tol": tol,
            "max_iter": max_iter,
            "init": str(init).lower(),
            "missing": str(missing).lower(),
        }


class AR(TimeSeriesModelBase):
    """
    Creates a inDB Autoregressor model.

    Parameters
    ----------
    name: str, optional
        Name of the model. The  model is stored  in the
        database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    p: int, optional
        Integer in the range [1, 1999], the number of
        lags to consider in the computation. Larger
        values for p weaken the correlation.
    method: str, optional
        One of the following algorithms for training the
        model:

        - ols:
            Ordinary Least Squares
        - yule-walker:
            Yule-Walker
    penalty: str, optional
        Method of regularization.

        - none:
            No regularization.
        - l2:
            L2 regularization.
    C: PythonNumber, optional
        The regularization parameter value. The value
        must be zero or non-negative.

    missing: str, optional
        Method for handling missing values, one of the
        following strings:

        - 'drop': Missing values are ignored.
        - 'raise': Missing values raise an error.
        - 'zero': Missing values are set to zero.
        - 'linear_interpolation': Missing values are
            replaced by a linearly interpolated value
            based on the nearest valid entries before
            and after the missing value. In cases
            where the first or last values in a
            dataset are missing, the function errors.

    Examples
    ---------

    The following examples provide a basic understanding of usage.
    For more detailed examples, please refer to the
    :ref:`user_guide.machine_learning` or the
    `Examples <https://www.vertica.com/python/examples/>`_
    section on the website.

    ...
    """

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["AUTOREGRESSOR"]:
        return "AUTOREGRESSOR"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_AUTOREGRESSOR"]:
        return "PREDICT_AUTOREGRESSOR"

    @property
    def _model_subcategory(self) -> Literal["TIMESERIES"]:
        return "TIMESERIES"

    @property
    def _model_type(self) -> Literal["AR"]:
        return "AR"

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        p: int = 3,
        method: Literal["ols", "yule-walker"] = "ols",
        penalty: Literal[None, "none", "l2"] = "none",
        C: PythonNumber = 1.0,
        missing: Literal[
            "drop", "raise", "zero", "linear_interpolation"
        ] = "linear_interpolation",
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "p": int(p),
            "method": str(method).lower(),
            "penalty": str(penalty).lower(),
            "C": C,
            "missing": str(missing).lower(),
            "compute_mse": True,
        }

    # Features Importance Methods.

    def _compute_features_importance(self) -> None:
        """
        Computes the features importance.
        """
        self.features_importance_ = self.phi_ / sum(abs(self.phi_))

    def _get_features_importance(self) -> np.ndarray:
        """
        Returns the features' importance.
        """
        if not hasattr(self, "features_importance_"):
            self._compute_features_importance()
        return copy.deepcopy(self.features_importance_)

    def features_importance(
        self, show: bool = True, chart: Optional[PlottingObject] = None, **style_kwargs
    ) -> PlottingObject:
        """
        Computes the model's features importance.

        Parameters
        ----------
        show: bool
            If set to True,  draw the feature's importance.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to pass to the Plotting
            functions.

        Returns
        -------
        obj
            features importance.
        """
        fi = self._get_features_importance()
        columns = [
            copy.deepcopy(self.y) + f"[t-{i + 1}]"
            for i in range(self.get_params()["p"])
        ]
        if show:
            data = {
                "importance": fi,
            }
            layout = {"columns": columns}
            vpy_plt, kwargs = self.get_plotting_lib(
                class_name="ImportanceBarChart",
                chart=chart,
                style_kwargs=style_kwargs,
            )
            return vpy_plt.ImportanceBarChart(data=data, layout=layout).draw(**kwargs)
        importances = {
            "index": [quote_ident(x)[1:-1].lower() for x in columns],
            "importance": list(abs(fi)),
            "sign": list(np.sign(fi)),
        }
        return TableSample(values=importances).sort(column="importance", desc=True)


class MA(TimeSeriesModelBase):
    """
    Creates a inDB Moving Average model.

    Parameters
    ----------
    name: str, optional
        Name of the model. The  model is stored  in the
        database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    q: int, optional
        Integer in the range [1, 67), the number of lags
        to consider in the computation.
    penalty: str, optional
        Method of regularization.

        - none:
            No regularization.
        - l2:
            L2 regularization.
    C: PythonNumber, optional
        The regularization parameter value. The value
        must be zero or non-negative.

    missing: str, optional
        Method for handling missing values, one of the
        following strings:

        - 'drop': Missing values are ignored.
        - 'raise': Missing values raise an error.
        - 'zero': Missing values are set to zero.
        - 'linear_interpolation': Missing values are
            replaced by a linearly interpolated value
            based on the nearest valid entries before
            and after the missing value. In cases
            where the first or last values in a
            dataset are missing, the function errors.

    Examples
    ---------

    The following examples provide a basic understanding of usage.
    For more detailed examples, please refer to the
    :ref:`user_guide.machine_learning` or the
    `Examples <https://www.vertica.com/python/examples/>`_
    section on the website.

    ...
    """

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["MOVING_AVERAGE"]:
        return "MOVING_AVERAGE"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_MOVING_AVERAGE"]:
        return "PREDICT_MOVING_AVERAGE"

    @property
    def _model_subcategory(self) -> Literal["TIMESERIES"]:
        return "TIMESERIES"

    @property
    def _model_type(self) -> Literal["MA"]:
        return "MA"

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        q: int = 1,
        penalty: Literal[None, "none", "l2"] = "none",
        C: PythonNumber = 1.0,
        missing: Literal[
            "drop", "raise", "zero", "linear_interpolation"
        ] = "linear_interpolation",
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {
            "q": int(q),
            "penalty": str(penalty).lower(),
            "C": C,
            "missing": str(missing).lower(),
            "compute_mse": True,
        }
