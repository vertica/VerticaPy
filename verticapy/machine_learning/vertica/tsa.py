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
from verticapy._utils._sql._vertica_version import check_minimum_version

from verticapy.core.vdataframe.base import TableSample, vDataFrame

import verticapy.machine_learning.metrics as mt
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
            "n_",
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
        self.n_ = self.get_vertica_attributes("accepted_row_count")[
            "accepted_row_count"
        ][0]

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
        output_index: bool = False,
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

              - No provided timeseries-column:
                    start must be an integer greater or equal
                    to 0, where zero indicates to start prediction
                    at the end of the in-sample data. If start is a
                    positive value, the function predicts the values
                    between the end of the in-sample data and the
                    start index, and then uses the predicted values
                    as time series inputs for the subsequent
                    npredictions.
              - timeseries-column provided:
                    start must be an integer greater or equal to 1
                    and identifies the index (row) of the timeseries
                    -column at which to begin prediction. If the start
                    index is greater than the number of rows, N, in the
                    input data, the function predicts the values between
                    N and start and uses the predicted values as time
                    series inputs for the subsequent npredictions.

            Default:

              - No provided timeseries-column:
                    prediction begins from the end of the in-sample
                    data.
              - timeseries-column provided:
                    prediction begins from the end of the provided
                    input data.
        npredictions: int, optional
            Integer greater or equal to 1, the number of predicted
            timesteps.
        output_standard_errors: bool, optional
            Boolean,  whether to return estimates  of the standard
            error of each prediction.
        output_index: bool, optional
            Boolean,  whether to return the index of each position.

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
            if output_standard_errors or output_index:
                output_standard_errors = f", output_standard_errors = true"
            else:
                output_standard_errors = ""
            # Deployment
            sql = f"""
                {self._vertica_predict_sql}({y}
                                            USING PARAMETERS 
                                            model_name = '{self.model_name}',
                                            add_mean = True,
                                            {start}
                                            npredictions = {npredictions}
                                            {output_standard_errors}) 
                                            OVER ({ts})"""
            return clean_query(sql)
        else:
            raise AttributeError(
                f"Method 'deploySQL' does not exist for {self._model_type} models."
            )

    # Features Importance Methods.

    def _compute_features_importance(self) -> None:
        """
        Computes the features importance.
        """
        if self._model_type == "MA" or (
            self._model_type in ("ARMA", "ARIMA") and self.get_params()["order"][0] == 0
        ):
            raise AttributeError(
                "Features Importance can not be computed for Moving Averages."
            )
        else:
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
        columns = [copy.deepcopy(self.y) + f"[t-{i + 1}]" for i in range(len(fi))]
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

    # Prediction / Transformation Methods.

    def predict(
        self,
        vdf: Optional[SQLRelation] = None,
        ts: Optional[str] = None,
        y: Optional[str] = None,
        start: Optional[int] = None,
        npredictions: int = 10,
        output_standard_errors: bool = False,
        output_index: bool = False,
        output_estimated_ts: bool = False,
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

              - No provided timeseries-column:
                    start must be an integer greater or equal
                    to 0, where zero indicates to start prediction
                    at the end of the in-sample data. If start is a
                    positive value, the function predicts the values
                    between the end of the in-sample data and the
                    start index, and then uses the predicted values
                    as time series inputs for the subsequent
                    npredictions.
              - timeseries-column provided:
                    start must be an integer greater or equal to 1
                    and identifies the index (row) of the timeseries
                    -column at which to begin prediction. If the start
                    index is greater than the number of rows, N, in the
                    input data, the function predicts the values between
                    N and start and uses the predicted values as time
                    series inputs for the subsequent npredictions.

            Default:

              - No provided timeseries-column:
                    prediction begins from the end of the in-sample
                    data.
              - timeseries-column provided:
                    prediction begins from the end of the provided
                    input data.
        npredictions: int, optional
            Integer greater or equal to 1, the number of predicted
            timesteps.
        output_standard_errors: bool, optional
            Boolean,  whether to return estimates  of the standard
            error of each prediction.
        output_index: bool, optional
            Boolean, whether to return the index of each prediction.
        output_estimated_ts: bool, optional
            Boolean, whether to return the estimated abscissa of
            each prediction. The real one is hard to obtain due to
            interval computations.

        Returns
        -------
        vDataFrame
            a new object.
        """
        ar_ma = False
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
            ar_ma = True
        sql = "SELECT " + self.deploySQL(
            ts=ts,
            y=y,
            start=start,
            npredictions=npredictions,
            output_standard_errors=(
                output_standard_errors or output_index or output_estimated_ts
            ),
            output_index=output_index,
        )
        no_relation = True
        if not (isinstance(vdf, NoneType)):
            sql += f" FROM {vdf}"
            no_relation = False
        if output_index or output_estimated_ts:
            j = self.n_
            if no_relation:
                if not (isinstance(start, NoneType)):
                    j = j + start
            elif not (isinstance(start, NoneType)):
                j = start
            if (output_standard_errors or output_estimated_ts) and not (ar_ma):
                if not (output_standard_errors):
                    stde_out = ""
                else:
                    stde_out = ", std_err"
                output_standard_errors = ", std_err"
            else:
                output_standard_errors = ""
                stde_out = ""
            if ar_ma:
                order_by = ""
            else:
                order_by = 'ORDER BY "std_err"'
            sql = f"""
                SELECT 
                    ROW_NUMBER() OVER ({order_by}) + {j} - 1 AS idx,
                    prediction{output_standard_errors}
                FROM ({sql}) VERTICAPY_SUBTABLE"""
        if output_estimated_ts:
            min_value = f"(SELECT MIN({self.ts}) FROM {self.input_relation})"
            delta = f"""
                (SELECT 
                    AVG(delta) 
                 FROM (SELECT 
                        {self.ts} - LAG({self.ts}) OVER (ORDER BY {self.ts}) AS delta 
                       FROM {self.input_relation}) VERTICAPY_SUBTABLE)"""
            sql = f"""
                SELECT
                    idx * {delta} + {min_value} AS {self.ts},
                    prediction{stde_out}
                FROM ({sql}) VERTICAPY_SUBTABLE"""
        return vDataFrame(clean_query(sql))

    # Model Evaluation Methods.

    def _evaluation_relation(
        self,
        start: Optional[int] = None,
        npredictions: Optional[int] = None,
    ):
        """
        Returns the relation needed to evaluate the
        model.
        """
        if hasattr(self, "test_relation"):
            test_relation = self.test_relation
        elif hasattr(self, "input_relation"):
            test_relation = self.input_relation
        else:
            raise AttributeError(
                "No attributes found. The model is probably not yet fitted."
            )
        parameters = self.get_params()
        if isinstance(start, NoneType):
            start = self.n_ / 4
        if isinstance(npredictions, NoneType):
            npredictions = self.n_ - start
        prediction = self.predict(
            vdf=test_relation,
            ts=self.ts,
            y=self.y,
            start=start,
            npredictions=npredictions,
            output_index=True,
        )
        sql = f"""
            (SELECT
                true_values.y_true,
                prediction_relation.prediction AS y_pred
            FROM 
                (
                    SELECT
                        ROW_NUMBER() OVER (ORDER BY {self.ts}) AS idx,
                        {self.y} AS y_true
                    FROM {test_relation}
                ) AS true_values
                NATURAL JOIN
                (SELECT * FROM {prediction}) AS prediction_relation) VERTICAPY_SUBTABLE"""
        return clean_query(sql)

    def regression_report(
        self,
        metrics: Union[
            str,
            Literal[None, "anova", "details"],
            list[Literal[tuple(mt.FUNCTIONS_REGRESSION_DICTIONNARY)]],
        ] = None,
        start: Optional[int] = None,
        npredictions: Optional[int] = None,
    ) -> Union[float, TableSample]:
        """
        Computes a regression report using multiple metrics to
        evaluate the model (r2, mse, max error...).

        Parameters
        ----------
        metrics: str, optional
            The metrics used to compute the regression report.
                None    : Computes the model different metrics.
                anova   : Computes the model ANOVA table.
                details : Computes the model details.
            You can also provide a list of different metrics,
            including the following:
                aic    : Akaike’s Information Criterion
                bic    : Bayesian Information Criterion
                max    : Max Error
                mae    : Mean Absolute Error
                median : Median Absolute Error
                mse    : Mean Squared Error
                msle   : Mean Squared Log Error
                qe     : quantile  error,  the quantile must be
                         included in the name. Example:
                         qe50.1% will return the quantile error
                         using q=0.501.
                r2     : R squared coefficient
                r2a    : R2 adjusted
                rmse   : Root Mean Squared Error
                var    : Explained Variance
        start: int, optional
            The behavior of the start parameter and its
            range of accepted values depends on whether
            you provide a timeseries-column (ts):

              - No provided timeseries-column:
                    start must be an integer greater or equal
                    to 0, where zero indicates to start prediction
                    at the end of the in-sample data. If start is a
                    positive value, the function predicts the values
                    between the end of the in-sample data and the
                    start index, and then uses the predicted values
                    as time series inputs for the subsequent
                    npredictions.
              - timeseries-column provided:
                    start must be an integer greater or equal to 1
                    and identifies the index (row) of the timeseries
                    -column at which to begin prediction. If the start
                    index is greater than the number of rows, N, in the
                    input data, the function predicts the values between
                    N and start and uses the predicted values as time
                    series inputs for the subsequent npredictions.

            Default:

              - No provided timeseries-column:
                    prediction begins from the end of the in-sample
                    data.
              - timeseries-column provided:
                    prediction begins from the end of the provided
                    input data.
        npredictions: int, optional
            Integer greater or equal to 1, the number of predicted
            timesteps.

        Returns
        -------
        TableSample
            report.
        """
        return mt.regression_report(
            "y_true",
            "y_pred",
            self._evaluation_relation(start=start, npredictions=npredictions),
            metrics=metrics,
            k=1,
        )

    report = regression_report

    def score(
        self,
        metric: Literal[
            tuple(mt.FUNCTIONS_REGRESSION_DICTIONNARY)
            + ("r2a", "r2_adj", "rsquared_adj", "r2adj", "r2adjusted", "rmse")
        ] = "r2",
        start: Optional[int] = None,
        npredictions: Optional[int] = None,
    ) -> float:
        """
        Computes the model score.

        Parameters
        ----------
        metric: str, optional
            The metric used to compute the score.
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
        start: int, optional
            The behavior of the start parameter and its
            range of accepted values depends on whether
            you provide a timeseries-column (ts):

              - No provided timeseries-column:
                    start must be an integer greater or equal
                    to 0, where zero indicates to start prediction
                    at the end of the in-sample data. If start is a
                    positive value, the function predicts the values
                    between the end of the in-sample data and the
                    start index, and then uses the predicted values
                    as time series inputs for the subsequent
                    npredictions.
              - timeseries-column provided:
                    start must be an integer greater or equal to 1
                    and identifies the index (row) of the timeseries
                    -column at which to begin prediction. If the start
                    index is greater than the number of rows, N, in the
                    input data, the function predicts the values between
                    N and start and uses the predicted values as time
                    series inputs for the subsequent npredictions.

            Default:

              - No provided timeseries-column:
                    prediction begins from the end of the in-sample
                    data.
              - timeseries-column provided:
                    prediction begins from the end of the provided
                    input data.
        npredictions: int, optional
            Integer greater or equal to 1, the number of predicted
            timesteps.

        Returns
        -------
        float
            score.
        """
        # Initialization
        metric = str(metric).lower()
        if metric in ["r2adj", "r2adjusted"]:
            metric = "r2a"
        adj, root = False, False
        if metric in ("r2a", "r2adj", "r2adjusted", "r2_adj", "rsquared_adj"):
            metric, adj = "r2", True
        elif metric == "rmse":
            metric, root = "mse", True
        fun = mt.FUNCTIONS_REGRESSION_DICTIONNARY[metric]

        # Scoring
        arg = [
            "y_true",
            "y_pred",
            self._evaluation_relation(start=start, npredictions=npredictions),
        ]
        if metric in ("aic", "bic") or adj:
            arg += [1]
        if root or adj:
            arg += [True]
        return fun(*arg)

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

              - No provided timeseries-column:
                    start must be an integer greater or equal
                    to 0, where zero indicates to start prediction
                    at the end of the in-sample data. If start is a
                    positive value, the function predicts the values
                    between the end of the in-sample data and the
                    start index, and then uses the predicted values
                    as time series inputs for the subsequent
                    npredictions.
              - timeseries-column provided:
                    start must be an integer greater or equal to 1
                    and identifies the index (row) of the timeseries
                    -column at which to begin prediction. If the start
                    index is greater than the number of rows, N, in the
                    input data, the function predicts the values between
                    N and start and uses the predicted values as time
                    series inputs for the subsequent npredictions.

            Default:

              - No provided timeseries-column:
                    prediction begins from the end of the in-sample
                    data.
              - timeseries-column provided:
                    prediction begins from the end of the provided
                    input data.
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

    .. versionadded:: 23.3.0

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

        - 'zero':
            Coefficients are initialized to zero.
        - 'hr':
            Coefficients are initialized using the
            Hannan-Rissanen algorithm.

    missing: str, optional
        Method for handling missing values, one of the
        following strings:

        - 'drop':
            Missing values are ignored.
        - 'raise':
            Missing values raise an error.
        - 'zero':
            Missing values are set to zero.
        - 'linear_interpolation':
            Missing values are replaced by a linearly
            interpolated value based on the nearest
            valid entries before and after the missing
            value. In cases where the first or last
            values in a dataset are missing, the function
            errors.

    Examples
    ---------

    The following examples provide a basic understanding of usage.
    For more detailed examples, please refer to the
    :ref:`user_guide.machine_learning` or the
    `Examples <https://www.vertica.com/python/examples/>`_
    section on the website.

    Initialization
    ^^^^^^^^^^^^^^^

    We import ``verticapy``:

    .. ipython:: python

        import verticapy as vp

    .. hint::

        By assigning an alias to ``verticapy``, we mitigate the risk
        of code collisions with other libraries. This precaution is
        necessary because verticapy uses commonly known function names
        like "average" and "median", which can potentially lead to naming
        conflicts. The use of an alias ensures that the functions from
        verticapy are used as intended without interfering with functions
        from other libraries.

    For this example, we will use the airline passengers dataset.

    .. code-block:: python

        import verticapy.datasets as vpd

        data = vpd.load_airline_passengers()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_airline_passengers.html

    .. note::

        VerticaPy offers a wide range of sample datasets that are
        ideal for training and testing purposes. You can explore
        the full list of available datasets in the :ref:`api.datasets`,
        which provides detailed information on each dataset
        and how to use them effectively. These datasets are invaluable
        resources for honing your data analysis and machine learning
        skills within the VerticaPy environment.

    .. ipython:: python
        :suppress:

        import verticapy.datasets as vpd
        data = vpd.load_airline_passengers()

    Model Initialization
    ^^^^^^^^^^^^^^^^^^^^^

    First we import the ``ARIMA`` model:

    .. ipython:: python

        from verticapy.machine_learning.vertica.tsa import ARIMA

    Then we can create the model:

    .. ipython:: python
        :okwarning:

        model = ARIMA(order = (12, 2, 2))

    .. hint::

        In ``verticapy`` 1.0.x and higher, you do not need to specify the
        model name, as the name is automatically assigned. If you need to
        re-use the model, you can fetch the model name from the model's
        attributes.

    .. important::

        The model name is crucial for the model management system and
        versioning. It's highly recommended to provide a name if you
        plan to reuse the model later.

    Model Training
    ^^^^^^^^^^^^^^^

    We can now fit the model:

    .. ipython:: python
        :okwarning:

        model.fit(data, "date", "passengers")

    .. important::

        To train a model, you can directly use the ``vDataFrame`` or the
        name of the relation stored in the database. The test set is optional
        and is only used to compute the test metrics. In ``verticapy``, we
        don't work using ``X`` matrices and ``y`` vectors. Instead, we work
        directly with lists of predictors and the response name.

    Features Importance
    ^^^^^^^^^^^^^^^^^^^^

    We can conveniently get the features importance:

    .. ipython:: python
        :okwarning:

        model.features_importance()

    .. ipython:: python
        :suppress:
        :okwarning:

        vp.set_option("plotting_lib", "plotly")
        fig = model.features_importance()
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arima_features.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arima_features.html

    .. important::

        Feature importance is determined by using the coefficients of the
        auto-regressive (AR) process and normalizing them. This method
        tends to be precise when your time series primarily consists of an
        auto-regressive component. However, its accuracy may be a topic of
        discussion if the time series contains other components as well.

    Metrics
    ^^^^^^^^

    We can get the entire report using:

    .. code-block:: python

        model.report()

    .. ipython:: python
        :suppress:
        :okwarning:

        result = model.report()
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arima_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arima_report.html

    You can also choose the number of predictions and where to start the forecast.
    For example, the following code will allow you to generate a report with 30
    predictions, starting the forecasting process at index 40.

    .. code-block:: python

        model.report(start = 40, npredictions = 30)

    .. ipython:: python
        :suppress:
        :okwarning:

        result = model.report(start = 40, npredictions = 30)
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arima_report_pred_2.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arima_report_pred_2.html

    .. important::

        Most metrics are computed using a single SQL query, but some of them might
        require multiple SQL queries. Selecting only the necessary metrics in the
        report can help optimize performance.
        E.g. ``model.report(metrics = ["mse", "r2"])``.

    You can utilize the
    :py:mod:`verticapy.machine_learning.vertica.tsa.ARIMA.score`
    function to calculate various regression metrics, with the explained
    variance being the default.

    .. ipython:: python
        :okwarning:

        model.score()

    The same applies to the score. You can choose where to start and
    the number of predictions to use.

    .. ipython:: python
        :okwarning:

        model.score(start = 40, npredictions = 30)

    .. important::

        If you do not specify a starting point and the number of
        predictions, the forecast will begin at one-fourth of the
        dataset, which can result in an inaccurate score, especially
        for large datasets. It's important to choose these parameters
        carefully.

    Prediction
    ^^^^^^^^^^^

    Prediction is straight-forward:

    .. code-block:: python

        model.predict()

    .. ipython:: python
        :suppress:
        :okwarning:

        result = model.predict()
        html_file = open("figures/machine_learning_vertica_tsa_arima_prediction.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arima_prediction.html

    .. hint::

        You can control the number of prediction steps by changing
        the ``npredictions`` parameter:
        ``model.predict(npredictions = 30)``.

    .. note::

        Predictions can be made automatically by using the training set,
        in which case you don't need to specify the predictors. Alternatively, you
        can pass only the ``vDataFrame`` to the
        :py:mod:`verticapy.machine_learning.vertica.tsa.ARIMA.predict`
        function, but in this case, it's essential that the column names of
        the ``vDataFrame`` match the predictors and response name in the
        model.

    If you would like to have the 'time-stamps' (ts) in the output then
    you can switch the ``output_estimated_ts`` the parameter. And if you
    also would like to see the standard error then you can switch the
    ``output_standard_errors``parameter:

    .. code-block:: python

        model.predict(output_estimated_ts = True, output_standard_errors = True)

    .. ipython:: python
        :suppress:
        :okwarning:

        result = model.predict(output_estimated_ts = True, output_standard_errors = True)
        html_file = open("figures/machine_learning_vertica_tsa_arima_prediction_2.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arima_prediction_2.html

    .. important::

        The ``output_estimated_ts`` parameter provides an estimation of
        'ts' assuming that 'ts' is regularly spaced.

    If you don't provide any input, the function will begin forecasting
    after the last known value. If you want to forecast starting from a
    specific value within the input dataset or another dataset, you can
    use the following syntax.

    .. code-block:: python

        model.predict(
            data,
            "date",
            "passengers",
            start = 40,
            npredictions = 20,
            output_estimated_ts = True,
            output_standard_errors = True,
        )

    .. ipython:: python
        :suppress:
        :okwarning:

        result = model.predict(data, "date", "passengers", start = 40, npredictions = 20, output_estimated_ts = True, output_standard_errors = True)
        html_file = open("figures/machine_learning_vertica_tsa_arima_prediction_3.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arima_prediction_3.html

    Plots
    ^^^^^^

    We can conveniently plot the predictions on a line plot
    to observe the efficacy of our model:

    .. code-block:: python

        model.plot(data, "date", "passengers", npredictions = 80, start=120)

    .. ipython:: python
        :suppress:
        :okwarning:

        vp.set_option("plotting_lib", "plotly")
        fig = model.plot(data, "date", "passengers", npredictions = 80, start = 120, width = 650)
        fig.write_html("figures/machine_learning_vertica_tsa_arima_plot_1.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arima_plot_1.html

    .. note::

        You can control the number of prediction steps by changing
        the ``npredictions`` parameter:
        ``model.plot(npredictions = 30)``.

    Please refer to  :ref:`chart_gallery.tsa` for more examples.

    Model Register
    ^^^^^^^^^^^^^^

    In order to register the model for tracking and versioning:

    .. code-block:: python

        model.register("model_v1")

    Please refer to :ref:`notebooks/ml/model_tracking_versioning/index.html`
    for more details on model tracking and versioning.
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

    .. versionadded:: 12.0.3

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

        - 'zero':
            Coefficients are initialized to zero.
        - 'hr':
            Coefficients are initialized using the
            Hannan-Rissanen algorithm.

    missing: str, optional
        Method for handling missing values, one of the
        following strings:

        - 'drop':
            Missing values are ignored.
        - 'raise':
            Missing values raise an error.
        - 'zero':
            Missing values are set to zero.
        - 'linear_interpolation':
            Missing values are replaced by a linearly
            interpolated value based on the nearest
            valid entries before and after the missing
            value. In cases where the first or last
            values in a dataset are missing, the function
            errors.

    Examples
    ---------

    The following examples provide a basic understanding of usage.
    For more detailed examples, please refer to the
    :ref:`user_guide.machine_learning` or the
    `Examples <https://www.vertica.com/python/examples/>`_
    section on the website.

    Initialization
    ^^^^^^^^^^^^^^^

    We import ``verticapy``:

    .. ipython:: python

        import verticapy as vp

    .. hint::

        By assigning an alias to ``verticapy``, we mitigate the risk
        of code collisions with other libraries. This precaution is
        necessary because verticapy uses commonly known function names
        like "average" and "median", which can potentially lead to naming
        conflicts. The use of an alias ensures that the functions from
        verticapy are used as intended without interfering with functions
        from other libraries.

    For this example, we will use the airline passengers dataset.

    .. code-block:: python

        import verticapy.datasets as vpd

        data = vpd.load_airline_passengers()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_airline_passengers.html

    .. note::

        VerticaPy offers a wide range of sample datasets that are
        ideal for training and testing purposes. You can explore
        the full list of available datasets in the :ref:`api.datasets`,
        which provides detailed information on each dataset
        and how to use them effectively. These datasets are invaluable
        resources for honing your data analysis and machine learning
        skills within the VerticaPy environment.

    .. ipython:: python
        :suppress:

        import verticapy.datasets as vpd
        data = vpd.load_airline_passengers()

    Model Initialization
    ^^^^^^^^^^^^^^^^^^^^^

    First we import the ``ARMA`` model:

    .. ipython:: python

        from verticapy.machine_learning.vertica.tsa import ARMA

    Then we can create the model:

    .. ipython:: python
        :okwarning:

        model = ARMA(order = (12, 2))

    .. hint::

        In ``verticapy`` 1.0.x and higher, you do not need to specify the
        model name, as the name is automatically assigned. If you need to
        re-use the model, you can fetch the model name from the model's
        attributes.

    .. important::

        The model name is crucial for the model management system and
        versioning. It's highly recommended to provide a name if you
        plan to reuse the model later.

    Model Training
    ^^^^^^^^^^^^^^^

    We can now fit the model:

    .. ipython:: python
        :okwarning:

        model.fit(data, "date", "passengers")

    .. important::

        To train a model, you can directly use the ``vDataFrame`` or the
        name of the relation stored in the database. The test set is optional
        and is only used to compute the test metrics. In ``verticapy``, we
        don't work using ``X`` matrices and ``y`` vectors. Instead, we work
        directly with lists of predictors and the response name.

    Features Importance
    ^^^^^^^^^^^^^^^^^^^^

    We can conveniently get the features importance:

    .. ipython:: python
        :okwarning:

        model.features_importance()

    .. ipython:: python
        :suppress:
        :okwarning:

        vp.set_option("plotting_lib", "plotly")
        fig = model.features_importance()
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arma_features.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arma_features.html

    .. important::

        Feature importance is determined by using the coefficients of the
        auto-regressive (AR) process and normalizing them. This method
        tends to be precise when your time series primarily consists of an
        auto-regressive component. However, its accuracy may be a topic of
        discussion if the time series contains other components as well.

    Metrics
    ^^^^^^^^

    We can get the entire report using:

    .. code-block:: python

        model.report()

    .. ipython:: python
        :suppress:
        :okwarning:

        result = model.report()
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arma_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arma_report.html

    You can also choose the number of predictions and where to start the forecast.
    For example, the following code will allow you to generate a report with 30
    predictions, starting the forecasting process at index 40.

    .. code-block:: python

        model.report(start = 40, npredictions = 30)

    .. ipython:: python
        :suppress:
        :okwarning:

        result = model.report(start = 40, npredictions = 30)
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arma_report_pred_2.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arma_report_pred_2.html

    .. important::

        Most metrics are computed using a single SQL query, but some of them might
        require multiple SQL queries. Selecting only the necessary metrics in the
        report can help optimize performance.
        E.g. ``model.report(metrics = ["mse", "r2"])``.

    You can utilize the
    :py:mod:`verticapy.machine_learning.vertica.tsa.ARMA.score`
    function to calculate various regression metrics, with the explained
    variance being the default.

    .. ipython:: python
        :okwarning:

        model.score()

    The same applies to the score. You can choose where to start and
    the number of predictions to use.

    .. ipython:: python
        :okwarning:

        model.score(start = 40, npredictions = 30)

    .. important::

        If you do not specify a starting point and the number of
        predictions, the forecast will begin at one-fourth of the
        dataset, which can result in an inaccurate score, especially
        for large datasets. It's important to choose these parameters
        carefully.

    Prediction
    ^^^^^^^^^^^

    Prediction is straight-forward:

    .. code-block:: python

        model.predict()

    .. ipython:: python
        :suppress:
        :okwarning:

        result = model.predict()
        html_file = open("figures/machine_learning_vertica_tsa_arma_prediction.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arma_prediction.html

    .. hint::

        You can control the number of prediction steps by changing
        the ``npredictions`` parameter:
        ``model.predict(npredictions = 30)``.

    .. note::

        Predictions can be made automatically by using the training set,
        in which case you don't need to specify the predictors. Alternatively, you
        can pass only the ``vDataFrame`` to the
        :py:mod:`verticapy.machine_learning.vertica.tsa.ARMA.predict`
        function, but in this case, it's essential that the column names of
        the ``vDataFrame`` match the predictors and response name in the
        model.

    If you would like to have the 'time-stamps' (ts) in the output then
    you can switch the ``output_estimated_ts`` the parameter. And if you
    also would like to see the standard error then you can switch the
    ``output_standard_errors``parameter:

    .. code-block:: python

        model.predict(output_estimated_ts = True, output_standard_errors = True)

    .. ipython:: python
        :suppress:
        :okwarning:

        result = model.predict(output_estimated_ts = True, output_standard_errors = True)
        html_file = open("figures/machine_learning_vertica_tsa_arma_prediction_2.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arma_prediction_2.html

    .. important::

        The ``output_estimated_ts`` parameter provides an estimation of
        'ts' assuming that 'ts' is regularly spaced.

    If you don't provide any input, the function will begin forecasting
    after the last known value. If you want to forecast starting from a
    specific value within the input dataset or another dataset, you can
    use the following syntax.

    .. code-block:: python

        model.predict(
            data,
            "date",
            "passengers",
            start = 40,
            npredictions = 20,
            output_estimated_ts = True,
            output_standard_errors = True,
        )

    .. ipython:: python
        :suppress:
        :okwarning:

        result = model.predict(data, "date", "passengers", start = 40, npredictions = 20, output_estimated_ts = True, output_standard_errors = True)
        html_file = open("figures/machine_learning_vertica_tsa_arma_prediction_3.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arma_prediction_3.html

    Plots
    ^^^^^^

    We can conveniently plot the predictions on a line plot
    to observe the efficacy of our model:

    .. code-block:: python

        model.plot(data, "date", "passengers", npredictions = 80, start=120)

    .. ipython:: python
        :suppress:
        :okwarning:

        vp.set_option("plotting_lib", "plotly")
        fig = model.plot(data, "date", "passengers", npredictions = 80, start = 120, width = 650)
        fig.write_html("figures/machine_learning_vertica_tsa_arma_plot_1.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arma_plot_1.html

    .. note::

        You can control the number of prediction steps by changing
        the ``npredictions`` parameter:
        ``model.plot(npredictions = 30)``.

    Please refer to  :ref:`chart_gallery.tsa` for more examples.

    Model Register
    ^^^^^^^^^^^^^^

    In order to register the model for tracking and versioning:

    .. code-block:: python

        model.register("model_v1")

    Please refer to :ref:`notebooks/ml/model_tracking_versioning/index.html`
    for more details on model tracking and versioning.
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

    .. versionadded:: 11.0.0

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

        - 'drop':
            Missing values are ignored.
        - 'raise':
            Missing values raise an error.
        - 'zero':
            Missing values are set to zero.
        - 'linear_interpolation':
            Missing values are replaced by a linearly
            interpolated value based on the nearest
            valid entries before and after the missing
            value. In cases where the first or last
            values in a dataset are missing, the function
            errors.

    Examples
    ---------

    The following examples provide a basic understanding of usage.
    For more detailed examples, please refer to the
    :ref:`user_guide.machine_learning` or the
    `Examples <https://www.vertica.com/python/examples/>`_
    section on the website.

    Initialization
    ^^^^^^^^^^^^^^^

    We import ``verticapy``:

    .. ipython:: python

        import verticapy as vp

    .. hint::

        By assigning an alias to ``verticapy``, we mitigate the risk
        of code collisions with other libraries. This precaution is
        necessary because verticapy uses commonly known function names
        like "average" and "median", which can potentially lead to naming
        conflicts. The use of an alias ensures that the functions from
        verticapy are used as intended without interfering with functions
        from other libraries.

    For this example, we will generate a dummy time-series
    dataset.

    .. ipython:: python

        data = vp.vDataFrame(
            {
                "month": [i for i in range(1, 11)],
                "GB": [5, 10, 20, 35, 55, 80, 110, 145, 185, 230],
            }
        )

    .. ipython:: python
        :suppress:

        html_file = open("figures/machine_learning_vertica_tsa_ar_data.html", "w")
        html_file.write(data._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_ar_data.html

    .. note::

        VerticaPy offers a wide range of sample datasets that are
        ideal for training and testing purposes. You can explore
        the full list of available datasets in the :ref:`api.datasets`,
        which provides detailed information on each dataset
        and how to use them effectively. These datasets are invaluable
        resources for honing your data analysis and machine learning
        skills within the VerticaPy environment.

    Model Initialization
    ^^^^^^^^^^^^^^^^^^^^^

    First we import the ``AR`` model:

    .. ipython:: python

        from verticapy.machine_learning.vertica.tsa import AR

    Then we can create the model:

    .. ipython:: python
        :okwarning:

        model = AR(p = 2)

    .. hint::

        In ``verticapy`` 1.0.x and higher, you do not need to specify the
        model name, as the name is automatically assigned. If you need to
        re-use the model, you can fetch the model name from the model's
        attributes.

    .. important::

        The model name is crucial for the model management system and
        versioning. It's highly recommended to provide a name if you
        plan to reuse the model later.

    Model Training
    ^^^^^^^^^^^^^^^

    We can now fit the model:

    .. ipython:: python
        :okwarning:

        model.fit(data, "month", "GB")

    .. important::

        To train a model, you can directly use the ``vDataFrame`` or the
        name of the relation stored in the database. The test set is optional
        and is only used to compute the test metrics. In ``verticapy``, we
        don't work using ``X`` matrices and ``y`` vectors. Instead, we work
        directly with lists of predictors and the response name.

    Features Importance
    ^^^^^^^^^^^^^^^^^^^^

    We can conveniently get the features importance:

    .. ipython:: python
        :okwarning:

        model.features_importance()

    .. ipython:: python
        :suppress:
        :okwarning:

        vp.set_option("plotting_lib", "plotly")
        fig = model.features_importance()
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_ar_features.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_ar_features.html

    Metrics
    ^^^^^^^^

    We can get the entire report using:

    .. code-block:: python

        model.report()

    .. ipython:: python
        :suppress:
        :okwarning:

        result = model.report()
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_ar_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_ar_report.html

    You can also choose the number of predictions and where to start the forecast.
    For example, the following code will allow you to generate a report with 30
    predictions, starting the forecasting process at index 40.

    .. code-block:: python

        model.report(start = 40, npredictions = 30)

    .. ipython:: python
        :suppress:
        :okwarning:

        result = model.report(start = 40, npredictions = 30)
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_ar_report_pred_2.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_ar_report_pred_2.html

    .. important::

        Most metrics are computed using a single SQL query, but some of them might
        require multiple SQL queries. Selecting only the necessary metrics in the
        report can help optimize performance.
        E.g. ``model.report(metrics = ["mse", "r2"])``.

    You can utilize the
    :py:mod:`verticapy.machine_learning.vertica.tsa.AR.score`
    function to calculate various regression metrics, with the explained
    variance being the default.

    .. ipython:: python
        :okwarning:

        model.score()

    The same applies to the score. You can choose where to start and
    the number of predictions to use.

    .. ipython:: python
        :okwarning:

        model.score(start = 40, npredictions = 30)

    .. important::

        If you do not specify a starting point and the number of
        predictions, the forecast will begin at one-fourth of the
        dataset, which can result in an inaccurate score, especially
        for large datasets. It's important to choose these parameters
        carefully.

    Prediction
    ^^^^^^^^^^^

    Prediction is straight-forward:

    .. code-block:: python

        model.predict()

    .. ipython:: python
        :suppress:
        :okwarning:

        result = model.predict()
        html_file = open("figures/machine_learning_vertica_tsa_ar_prediction.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_ar_prediction.html

    .. hint::

        You can control the number of prediction steps by changing
        the ``npredictions`` parameter:
        ``model.predict(npredictions = 30)``.

    .. note::

        Predictions can be made automatically by using the training set,
        in which case you don't need to specify the predictors. Alternatively, you
        can pass only the ``vDataFrame`` to the
        :py:mod:`verticapy.machine_learning.vertica.tsa.AR.predict`
        function, but in this case, it's essential that the column names of
        the ``vDataFrame`` match the predictors and response name in the
        model.

    If you would like to have the 'time-stamps' (ts) in the output then
    you can switch the ``output_estimated_ts`` the parameter.

    .. code-block:: python

        model.predict(output_estimated_ts = True)

    .. ipython:: python
        :suppress:
        :okwarning:

        result = model.predict(output_estimated_ts = True)
        html_file = open("figures/machine_learning_vertica_tsa_ar_prediction_2.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_ar_prediction_2.html

    .. important::

        The ``output_estimated_ts`` parameter provides an estimation of
        'ts' assuming that 'ts' is regularly spaced.

    If you don't provide any input, the function will begin forecasting
    after the last known value. If you want to forecast starting from a
    specific value within the input dataset or another dataset, you can
    use the following syntax.

    .. code-block:: python

        model.predict(
            data,
            "month",
            "GB",
            start = 40,
            npredictions = 20,
            output_estimated_ts = True,
        )

    .. ipython:: python
        :suppress:
        :okwarning:

        result = model.predict(data, "month", "GB", start = 40, npredictions = 20, output_estimated_ts = True)
        html_file = open("figures/machine_learning_vertica_tsa_ar_prediction_3.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_ar_prediction_3.html

    Plots
    ^^^^^^

    We can conveniently plot the predictions on a line plot
    to observe the efficacy of our model:

    .. code-block:: python

        model.plot(data, "month", "GB", npredictions = 80, start=120)

    .. ipython:: python
        :suppress:
        :okwarning:

        vp.set_option("plotting_lib", "plotly")
        fig = model.plot(data, "month", "GB", npredictions = 80, start = 120, width = 650)
        fig.write_html("figures/machine_learning_vertica_tsa_ar_plot_1.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_ar_plot_1.html

    .. note::

        You can control the number of prediction steps by changing
        the ``npredictions`` parameter:
        ``model.plot(npredictions = 30)``.

    Please refer to  :ref:`chart_gallery.tsa` for more examples.

    Model Register
    ^^^^^^^^^^^^^^

    In order to register the model for tracking and versioning:

    .. code-block:: python

        model.register("model_v1")

    Please refer to :ref:`notebooks/ml/model_tracking_versioning/index.html`
    for more details on model tracking and versioning.
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

        - 'drop':
            Missing values are ignored.
        - 'raise':
            Missing values raise an error.
        - 'zero':
            Missing values are set to zero.
        - 'linear_interpolation':
            Missing values are replaced by a linearly
            interpolated value based on the nearest
            valid entries before and after the missing
            value. In cases where the first or last
            values in a dataset are missing, the function
            errors.

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
