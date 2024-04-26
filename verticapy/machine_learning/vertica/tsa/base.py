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
from abc import abstractmethod
import copy
import datetime
from dateutil.relativedelta import relativedelta
from typing import Literal, Optional, Union

import numpy as np

import verticapy._config.config as conf
from verticapy._typing import (
    PlottingObject,
    NoneType,
    SQLColumns,
    SQLRelation,
)
from verticapy._utils._gen import gen_tmp_name
from verticapy._utils._sql._format import (
    clean_query,
    format_type,
    quote_ident,
    schema_relation,
)
from verticapy._utils._sql._sys import _executeSQL

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
        if self._model_type in ("ARMA", "ARIMA"):
            return [
                "phi_",
                "theta_",
                "mean_",
                "features_importance_",
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

    def _ismultivar(self) -> bool:
        """
        Returns ``True`` if the model
        is multivariate.
        """
        try:
            return (
                self.get_vertica_attributes("num_predictors")["num_predictors"][0] > 1
            )
        except:
            return False

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        if self._ismultivar():
            p = self.parameters["p"]
            self.intercept_ = np.array(self.get_vertica_attributes("mean")["value"])
            minmax = (
                vDataFrame(self.input_relation)[self.y]
                .aggregate(["min", "max"])
                .to_numpy()
            )
            self.min_ = minmax[:, 0]
            self.max_ = minmax[:, 1]
            self.phi_ = []
            for i in range(1, p + 1):
                phi = np.array(self.get_vertica_attributes(f"phi_(t-{i})").to_numpy())
                self.phi_ += [phi[:, 1:].astype(float)]
            self.phi_ = np.array(self.phi_)
        else:
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
        y: SQLColumns,
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
            TS (Time Series) :py:class`vDataColumn`
            used to order the data. The
            :py:class`vDataColumn` type must be
            ``date`` (``date``, ``datetime``,
            ``timestamp``...) or numerical.
        y: SQLColumns
            Response column.

            In the case of multivariate analysis,
            it represents a ``list`` of all the
            predictors.
        test_relation: SQLRelation, optional
            Relation used to test the model.
        return_report: bool, optional
            [For native models]
            When set to ``True``, the model
            summary will be returned. Otherwise,
            it will be printed.

        Returns
        -------
        str
            model's summary.

        Examples
        --------
        We import :py:mod:`verticapy`:

        .. ipython:: python

            import verticapy as vp

        For this example, we will use
        the airline passengers dataset.

        .. code-block:: python

            import verticapy.datasets as vpd

            data = vpd.load_airline_passengers()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_airline_passengers.html

        .. ipython:: python
            :suppress:

            import verticapy.datasets as vpd
            data = vpd.load_airline_passengers()

        First we import the model:

        .. ipython:: python

            from verticapy.machine_learning.vertica.tsa import ARIMA

        Then we can create the model:

        .. ipython:: python
            :okwarning:

            model = ARIMA(order = (12, 1, 2))

        We can now fit the model:

        .. ipython:: python
            :okwarning:

            model.fit(data, "date", "passengers")

        .. important::

            For this example, a specific model is
            utilized, and it may not correspond
            exactly to the model you are working
            with. To see a comprehensive example
            specific to your class of interest,
            please refer to that particular class.

            Examples:
            :py:class:`~verticapy.machine_learning.vertica.tsa.ARIMA`;
            :py:class:`~verticapy.machine_learning.vertica.tsa.ARMA`;
            :py:class:`~verticapy.machine_learning.vertica.tsa.AR`;
            :py:class:`~verticapy.machine_learning.vertica.tsa.MA`;
        """

        # Initialization
        if self.overwrite_model:
            self.drop()
        else:
            self._is_already_stored(raise_error=True)
        self.ts = quote_ident(ts)
        y = format_type(y, dtype=list)
        if len(y) > 1:
            self.y = quote_ident(y)
            y_str = ", ".join(self.y)
        else:
            self.y = quote_ident(y[0])
            y_str = self.y
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
                            {self.ts}, {y_str}
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
                     '{y_str}',
                     '{self.ts}' 
                     USING PARAMETERS 
                     {', '.join([f"{p} = {parameters[p]}" for p in parameters])})"""
            try:
                _executeSQL(query, title="Fitting the model.")
                self._compute_attributes()
            finally:
                if tmp_view:
                    drop(relation, method="view")
            report = self.summarize()
            if return_report:
                return report
            if conf.get_option("print_info"):
                print(report)
        else:
            self._compute_attributes()
        return None

    # I/O Methods.

    def deploySQL(
        self,
        ts: Optional[str] = None,
        y: Optional[SQLColumns] = None,
        start: Optional[int] = None,
        npredictions: int = 10,
        output_standard_errors: bool = False,
        output_index: bool = False,
        use_index_as_suffix: bool = False,
    ) -> str:
        """
        Returns the SQL code
        needed to deploy the
        model.

        Parameters
        ----------
        ts: str
            TS (Time Series) :py:class`vDataColumn`
            used to order the data. The
            :py:class`vDataColumn` type must be
            ``date`` (``date``, ``datetime``,
            ``timestamp``...) or numerical.
        y: SQLColumns, optional
            Response column.

            In the case of multivariate analysis,
            it represents a ``list`` of all the
            predictors.
        start: int, optional
            The behavior of the start parameter and its
            range of accepted values depends on whether
            you provide a timeseries-column (``ts``):

            - No provided timeseries-column:
                ``start`` must be an integer
                greater or equal to 0, where
                zero indicates to start
                prediction at the end of the
                in-sample data. If ``start``
                is a positive value, the function
                predicts the values between the
                end of the in-sample data and
                the start index, and then uses
                the predicted values as time
                series inputs for the subsequent
                ``npredictions``.
            - timeseries-column provided:
                ``start`` must be an ``integer``
                greater or equal to ``1`` and
                identifies the index (row) of
                the timeseries-column at which
                to begin prediction. If the
                ``start`` index is greater than
                the number of rows, ``N``, in the
                input data, the function predicts
                the values between ``N`` and
                ``start`` and uses the predicted
                values as time series inputs for
                the subsequent npredictions.

            Default:

            - No provided timeseries-column:
                prediction begins from the
                end of the in-sample data.
            - timeseries-column provided:
                prediction begins from the
                end of the provided input
                data.

        npredictions: int, optional
            ``integer`` greater or equal to ``1``,
            the number of predicted timesteps.
        output_standard_errors: bool, optional
            ``boolean``, whether to return
            estimates  of the standard error
            of each prediction.
        output_index: bool, optional
            ``boolean``, whether to return
            the index of each position.
        use_index_as_suffix: bool, optional
            [Only used for multivariates models]
            If set to ``True``, indexes are used as
            suffix instead of predictors names.

        Returns
        -------
        str
            the SQL code needed
            to deploy the model.

        Examples
        --------
        We import :py:mod:`verticapy`:

        .. ipython:: python

            import verticapy as vp

        For this example, we will use
        the airline passengers dataset.

        .. code-block:: python

            import verticapy.datasets as vpd

            data = vpd.load_airline_passengers()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_airline_passengers.html

        .. ipython:: python
            :suppress:

            import verticapy.datasets as vpd
            data = vpd.load_airline_passengers()

        First we import the model:

        .. ipython:: python

            from verticapy.machine_learning.vertica.tsa import ARIMA

        Then we can create the model:

        .. ipython:: python
            :okwarning:

            model = ARIMA(order = (12, 1, 2))

        We can now fit the model:

        .. ipython:: python
            :okwarning:

            model.fit(data, "date", "passengers")

        To get the SQL query which uses
        Vertica functions use below:

        .. ipython:: python

            model.deploySQL()

        .. important::

            For this example, a specific model is
            utilized, and it may not correspond
            exactly to the model you are working
            with. To see a comprehensive example
            specific to your class of interest,
            please refer to that particular class.

            Examples:
            :py:class:`~verticapy.machine_learning.vertica.tsa.ARIMA`;
            :py:class:`~verticapy.machine_learning.vertica.tsa.ARMA`;
            :py:class:`~verticapy.machine_learning.vertica.tsa.AR`;
            :py:class:`~verticapy.machine_learning.vertica.tsa.MA`;
        """
        if self._vertica_predict_sql:
            # Initialization
            if isinstance(ts, NoneType):
                ts = ""
            else:
                ts = "ORDER BY " + quote_ident(ts)
            if isinstance(y, NoneType):
                y_str = ""
            elif isinstance(y, list):
                y_str = ", ".join(quote_ident(y))
            else:
                y_str = quote_ident(y)
            if isinstance(start, NoneType):
                start = ""
            else:
                start = f"start = {start},"
            if output_standard_errors or output_index:
                output_standard_errors = f", output_standard_errors = true"
            else:
                output_standard_errors = ""
            if self._ismultivar():
                if use_index_as_suffix:
                    alias = ", ".join([f"prediction{i}" for i in range(len(self.y))])
                else:
                    alias = ", ".join([f"prediction_{col[1:-1]}" for col in self.y])
                alias = f" AS (index, {alias})"
            else:
                alias = ""
            # Deployment
            sql = f"""
                {self._vertica_predict_sql}({y_str}
                                            USING PARAMETERS 
                                            model_name = '{self.model_name}',
                                            add_mean = True,
                                            {start}
                                            npredictions = {npredictions}
                                            {output_standard_errors}) 
                                            OVER ({ts}){alias}"""
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
        elif self._ismultivar():
            res = []
            n, k = len(self.y), self.parameters["p"]
            for i in range(n):
                fi_k = []
                for m in range(k):
                    fi_k += [self.phi_[m][i, :] * (self.max_ - self.min_)]
                v = np.concatenate(fi_k)
                res += [100.0 * v / sum(abs(v))]
            self.features_importance_ = res
        else:
            self.features_importance_ = 100.0 * self.phi_ / sum(abs(self.phi_))

    def _get_features_importance(self) -> np.ndarray:
        """
        Returns the features' importance.
        """
        if not hasattr(self, "features_importance_"):
            self._compute_features_importance()
        return copy.deepcopy(self.features_importance_)

    def features_importance(
        self,
        idx: int = 0,
        show: bool = True,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Computes the model's
        features importance.

        Parameters
        ----------
        idx: int, optional
            It represents the index of the
            predictor for which we want to
            compute the feature importance.
        show: bool, optional
            If set to ``True``, draw the
            feature's importance.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to pass
            to the Plotting functions.

        Returns
        -------
        obj
            features importance.

        Examples
        --------
        We import :py:mod:`verticapy`:

        .. ipython:: python

            import verticapy as vp

        For this example, we will use
        the airline passengers dataset.

        .. code-block:: python

            import verticapy.datasets as vpd

            data = vpd.load_airline_passengers()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_airline_passengers.html

        .. ipython:: python
            :suppress:

            import verticapy.datasets as vpd
            data = vpd.load_airline_passengers()

        First we import the model:

        .. ipython:: python

            from verticapy.machine_learning.vertica.tsa import ARIMA

        Then we can create the model:

        .. ipython:: python
            :okwarning:

            model = ARIMA(order = (12, 1, 2))

        We can now fit the model:

        .. ipython:: python
            :okwarning:

            model.fit(data, "date", "passengers")

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

            For this example, a specific model is
            utilized, and it may not correspond
            exactly to the model you are working
            with. To see a comprehensive example
            specific to your class of interest,
            please refer to that particular class.

            Examples:
            :py:class:`~verticapy.machine_learning.vertica.tsa.ARIMA`;
            :py:class:`~verticapy.machine_learning.vertica.tsa.ARMA`;
            :py:class:`~verticapy.machine_learning.vertica.tsa.AR`;
            :py:class:`~verticapy.machine_learning.vertica.tsa.MA`;
        """
        n, k = (
            len(self.y),
            self.parameters["p"]
            if "p" in self.parameters
            else self.parameters["order"][0],
        )
        fi = self._get_features_importance()
        if self._ismultivar() and not (0 <= idx < n):
            raise ValueError(
                "Parameter 'idx' represents the index of the predictor for "
                "which we want to compute the feature importance. It should "
                "be between 0 and the total number of predictors minus one"
                f" ({len(self.y) - 1})"
            )
        if self._ismultivar():
            fi = fi[idx]
            columns = []
            for m in range(k):
                for i in range(n):
                    columns += [f"{self.y[i]}[t-{m + 1}]"]
            title = f"Importance [{self.y[idx]}] (%)"
        else:
            columns = [copy.deepcopy(self.y) + f"[t-{i + 1}]" for i in range(len(fi))]
            title = "Importance (%)"
        if show:
            data = {
                "importance": fi,
            }
            layout = {"columns": columns, "x_label": title}
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
        y: Optional[SQLColumns] = None,
        start: Optional[int] = None,
        npredictions: int = 10,
        output_standard_errors: bool = False,
        output_index: bool = False,
        output_estimated_ts: bool = False,
        freq: Literal[None, "m", "months", "y", "year", "infer"] = "infer",
        filter_step: Optional[int] = None,
        method: Literal["auto", "forecast"] = "auto",
        use_index_as_suffix: bool = False,
    ) -> vDataFrame:
        """
        Predicts using the input relation.

        Parameters
        ----------
        vdf: SQLRelation
            Object used to run the prediction.
            You can also specify a customized
            relation, but you must enclose it
            with an alias. For example,
            ``(SELECT 1) x`` is valid, whereas
            ``(SELECT 1)`` and ``SELECT 1``
            are invalid.
        ts: str
            TS (Time Series) :py:class`vDataColumn`
            used to order the data. The
            :py:class`vDataColumn` type must be
            ``date`` (``date``, ``datetime``,
            ``timestamp``...) or numerical.
        y: SQLColumns, optional
            Response column.

            In the case of multivariate analysis,
            it represents a ``list`` of all the
            predictors.
        start: int, optional
            The behavior of the start parameter and its
            range of accepted values depends on whether
            you provide a timeseries-column (``ts``):

            - No provided timeseries-column:
                ``start`` must be an integer
                greater or equal to 0, where
                zero indicates to start
                prediction at the end of the
                in-sample data. If ``start``
                is a positive value, the function
                predicts the values between the
                end of the in-sample data and
                the start index, and then uses
                the predicted values as time
                series inputs for the subsequent
                ``npredictions``.
            - timeseries-column provided:
                ``start`` must be an ``integer``
                greater or equal to ``1`` and
                identifies the index (row) of
                the timeseries-column at which
                to begin prediction. If the
                ``start`` index is greater than
                the number of rows, ``N``, in the
                input data, the function predicts
                the values between ``N`` and
                ``start`` and uses the predicted
                values as time series inputs for
                the subsequent npredictions.

            Default:

            - No provided timeseries-column:
                prediction begins from the
                end of the in-sample data.
            - timeseries-column provided:
                prediction begins from the
                end of the provided input
                data.

        npredictions: int, optional
            ``integer`` greater or equal to ``1``,
            the number of predicted timesteps.
        output_standard_errors: bool, optional
            ``boolean``, whether to return
            estimates  of the standard error
            of each prediction.
        output_index: bool, optional
            ``boolean``, whether to return
            the index of each position.
        output_estimated_ts: bool, optional
            Boolean, whether to return the
            estimated abscissa of each
            prediction. The real one is
            hard to obtain due to interval
            computations.
        freq: str, optional
            How to compute the delta.

            - m/month:
                We assume that the data
                is organized on a monthly
                basis.
            - y/year:
                We assume that the data
                is organized on a yearly
                basis.
            - infer:
                When making inferences, the
                system will attempt to identify
                the best option, which may
                involve more computational
                resources.
            - None:
                The inference is based on the
                average of the difference
                between ``ts`` and its lag.

        filter_step: int, optional
            Integer parameter that determines
            the frequency of predictions. You
            can adjust it according to your
            specific requirements, such as
            setting it to ``3`` for predictions
            every third step.

            .. note::

                It is only utilized when
                ``output_estimated_ts=True``.
        method: str, optional
            Forecasting method. One of the following:

            - auto:
                the model initially utilizes the true
                values at each step for forecasting.
                However, when it reaches a point where
                it can no longer rely on true values,
                it transitions to using its own
                predictions for further forecasting.
                This method is often referred to as
                "one step ahead" forecasting.
            - forecast:
                the model initiates forecasting from
                an initial value and entirely disregards
                any subsequent true values. This approach
                involves forecasting based solely on the
                model's own predictions and does not
                consider actual observations after the
                start point.
        use_index_as_suffix: bool, optional
            [Only used for multivariates models]
            If set to ``True``, indexes are used as
            suffix instead of predictors names.

        Returns
        -------
        vDataFrame
            a new object.

        Examples
        --------
        We import :py:mod:`verticapy`:

        .. ipython:: python

            import verticapy as vp

        For this example, we will use
        the airline passengers dataset.

        .. code-block:: python

            import verticapy.datasets as vpd

            data = vpd.load_airline_passengers()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_airline_passengers.html

        .. ipython:: python
            :suppress:

            import verticapy.datasets as vpd
            data = vpd.load_airline_passengers()

        First we import the model:

        .. ipython:: python

            from verticapy.machine_learning.vertica.tsa import ARIMA

        Then we can create the model:

        .. ipython:: python
            :okwarning:

            model = ARIMA(order = (12, 1, 2))

        We can now fit the model:

        .. ipython:: python
            :okwarning:

            model.fit(data, "date", "passengers")

        Prediction is straight-forward:

        .. code-block:: python

            model.predict()

        .. ipython:: python
            :suppress:
            :okwarning:

            result = model.predict()
            html_file = open("SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arima_prediction.html", "w")
            html_file.write(result._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arima_prediction.html

        .. important::

            For this example, a specific model is
            utilized, and it may not correspond
            exactly to the model you are working
            with. To see a comprehensive example
            specific to your class of interest,
            please refer to that particular class.

            Examples:
            :py:class:`~verticapy.machine_learning.vertica.tsa.ARIMA`;
            :py:class:`~verticapy.machine_learning.vertica.tsa.ARMA`;
            :py:class:`~verticapy.machine_learning.vertica.tsa.AR`;
            :py:class:`~verticapy.machine_learning.vertica.tsa.MA`;
        """
        ar_ma = False
        if self._model_type in (
            "VAR",
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
        if isinstance(start, (int, float)):
            start_predict = int(start + 1)
        elif not (isinstance(start, NoneType)):
            start_predict = int(start)
        else:
            start_predict = None
        where = ""
        if isinstance(filter_step, NoneType):
            filter_step = 1
        elif filter_step < 1:
            raise ValueError("Parameter 'filter_step' must be greater or equal to 1.")
        else:
            where = f" WHERE MOD(idx, {filter_step}) = 0"
        sql = "SELECT " + self.deploySQL(
            ts=ts,
            y=y,
            start=start_predict,
            npredictions=npredictions,
            output_standard_errors=(
                output_standard_errors or output_index or output_estimated_ts
            ),
            output_index=output_index,
            use_index_as_suffix=use_index_as_suffix,
        )
        no_relation = True
        if self._ismultivar():
            if use_index_as_suffix:
                prediction = ", ".join([f"prediction{i}" for i in range(len(self.y))])
            else:
                prediction = ", ".join([f"prediction_{col[1:-1]}" for col in self.y])
        else:
            prediction = "prediction"
        if not (isinstance(vdf, NoneType)):
            relation = vdf
            if not (isinstance(start, NoneType)) and str(method).lower() == "forecast":
                relation = f"(SELECT * FROM {vdf} ORDER BY {ts} LIMIT {start}) VERTICAPY_SUBTABLE"
            sql += f" FROM {relation}"
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
            sql = f"""
                SELECT 
                    ROW_NUMBER() OVER () + {j} - 1 AS idx,
                    {prediction}{output_standard_errors}
                FROM ({sql}) VERTICAPY_SUBTABLE"""
        if output_estimated_ts:
            if isinstance(freq, str):
                freq = freq.lower()
            if freq == "infer":
                infer_sql = f"""
                    SELECT 
                        {self.ts} 
                    FROM {self.input_relation} 
                    WHERE {self.ts} IS NOT NULL
                    ORDER BY 1 
                    LIMIT 100"""
                res = _executeSQL(
                    infer_sql, title="Finding the right delta.", method="fetchall"
                )
                res = [l[0] for l in res]
                n = len(res)
                for i in range(1, n):
                    if not (isinstance(res[i], datetime.date)):
                        freq = None
                        break
                    dm = ((res[i] - res[i - 1]) / 28).days
                    dy = ((res[i] - res[i - 1]) / 365).days
                    if res[i - 1] + relativedelta(months=dm) == res[i] and freq != "y":
                        freq = "m"
                    elif res[i - 1] + relativedelta(years=dy) == res[i] and freq != "m":
                        freq = "y"
                    else:
                        freq = None
                        break
            min_value = f"(SELECT MIN({self.ts}) FROM {self.input_relation})"
            if freq in ("m", "months", "y", "year"):
                delta_ts = f"MONTHS_BETWEEN({self.ts}, LAG({self.ts}) OVER (ORDER BY {self.ts})) AS delta"
            else:
                delta_ts = (
                    f"{self.ts} - LAG({self.ts}) OVER (ORDER BY {self.ts}) AS delta"
                )
            delta = f"""
                (SELECT 
                    AVG(delta)
                 FROM (SELECT 
                        {delta_ts} 
                       FROM {self.input_relation}) VERTICAPY_SUBTABLE)"""
            if freq in ("m", "months"):
                estimation = f"TIMESTAMPADD(MONTH, (idx * {delta})::int, {min_value})::date AS {self.ts}"
            elif freq in ("y", "year"):
                estimation = f"TIMESTAMPADD(YEAR, (idx * {delta} / 12)::int, {min_value})::date AS {self.ts}"
            else:
                estimation = f"idx * {delta} + {min_value} AS {self.ts}"
            sql = f"""
                SELECT
                    {estimation},
                    {prediction}{stde_out}
                FROM ({sql}) VERTICAPY_SUBTABLE{where}"""
        return vDataFrame(clean_query(sql))

    # Model Evaluation Methods.

    def _evaluation_relation(
        self,
        start: Optional[int] = None,
        npredictions: Optional[int] = None,
        method: Literal["auto", "forecast"] = "auto",
    ) -> str:
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
            method=method,
            use_index_as_suffix=True,
        )
        if self._ismultivar():
            y_str = ", ".join([f"{col} AS y_true{i}" for i, col in enumerate(self.y)])
            prediction_str = ", ".join(
                [
                    f"prediction_relation.prediction{i} AS y_pred{i}"
                    for i, col in enumerate(self.y)
                ]
            )
            y_true_str = ", ".join(
                [f"true_values.y_true{i}" for i, col in enumerate(self.y)]
            )
        else:
            y_str = f"{self.y} AS y_true"
            prediction_str = "prediction_relation.prediction AS y_pred"
            y_true_str = "true_values.y_true"
        sql = f"""
            (SELECT
                {y_true_str},
                {prediction_str}
            FROM 
                (
                    SELECT
                        ROW_NUMBER() OVER (ORDER BY {self.ts}) - 1 AS idx,
                        {y_str}
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
        method: Literal["auto", "forecast"] = "auto",
    ) -> Union[float, TableSample]:
        """
        Computes a regression report
        using multiple metrics to
        evaluate the model (``r2``,
        ``mse``, ``max error``...).

        Parameters
        ----------
        metrics: str | list, optional
            The metrics used to compute
            the regression report.

             - None:
                Computes the model different metrics.
             - anova:
                Computes the model ANOVA table.
             - details:
                Computes the model details.

            It can also be a ``list`` of the
            metrics used to compute the final
            report.

            - aic:
                Akaike's Information Criterion

                .. math::

                    AIC = 2k - 2\ln(\hat{L})

            - bic:
                Bayesian Information Criterion

                .. math::

                    BIC = -2\ln(\hat{L}) + k \ln(n)

            - max:
                Max Error.

                .. math::

                    ME = \max_{i=1}^{n} \left| y_i - \hat{y}_i \\right|

            - mae:
                Mean Absolute Error.

                .. math::

                    MAE = \\frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \\right|

            - median:
                Median Absolute Error.

                .. math::

                    MedAE = \\text{median}_{i=1}^{n} \left| y_i - \hat{y}_i \\right|

            - mse:
                Mean Squared Error.

                .. math::

                    MsE = \\frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \\right)^2

            - msle:
                Mean Squared Log Error.

                .. math::

                    MSLE = \\frac{1}{n} \sum_{i=1}^{n} (\log(1 + y_i) - \log(1 + \hat{y}_i))^2

            - r2:
                R squared coefficient.

                .. math::

                    R^2 = 1 - \\frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \\bar{y})^2}

            - r2a:
                R2 adjusted

                .. math::

                    \\text{Adjusted } R^2 = 1 - \\frac{(1 - R^2)(n - 1)}{n - k - 1}

            - qe:
                quantile error, the quantile must be
                included in the name. Example:
                qe50.1% will  return the quantile
                error using q=0.501.

            - rmse:
                Root-mean-squared error

                .. math::

                    RMSE = \sqrt{\\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}

            - var:
                Explained Variance

                .. math::

                    \\text{Explained Variance}   = 1 - \\frac{Var(y - \hat{y})}{Var(y)}
        start: int, optional
            The behavior of the start parameter and its
            range of accepted values depends on whether
            you provide a timeseries-column (``ts``):

            - No provided timeseries-column:
                ``start`` must be an integer
                greater or equal to 0, where
                zero indicates to start
                prediction at the end of the
                in-sample data. If ``start``
                is a positive value, the function
                predicts the values between the
                end of the in-sample data and
                the start index, and then uses
                the predicted values as time
                series inputs for the subsequent
                ``npredictions``.
            - timeseries-column provided:
                ``start`` must be an ``integer``
                greater or equal to ``1`` and
                identifies the index (row) of
                the timeseries-column at which
                to begin prediction. If the
                ``start`` index is greater than
                the number of rows, ``N``, in the
                input data, the function predicts
                the values between ``N`` and
                ``start`` and uses the predicted
                values as time series inputs for
                the subsequent npredictions.

            Default:

            - No provided timeseries-column:
                prediction begins from the
                end of the in-sample data.
            - timeseries-column provided:
                prediction begins from the
                end of the provided input
                data.

        npredictions: int, optional
            ``integer`` greater or equal to ``1``,
            the number of predicted timesteps.
        method: str, optional
            Forecasting method. One of the following:

            - auto:
                the model initially utilizes the true
                values at each step for forecasting.
                However, when it reaches a point where
                it can no longer rely on true values,
                it transitions to using its own
                predictions for further forecasting.
                This method is often referred to as
                "one step ahead" forecasting.
             - forecast:
                the model initiates forecasting from
                an initial value and entirely disregards
                any subsequent true values. This approach
                involves forecasting based solely on the
                model's own predictions and does not
                consider actual observations after the
                start point.

        Returns
        -------
        TableSample
            report.

        Examples
        --------
        We import :py:mod:`verticapy`:

        .. ipython:: python

            import verticapy as vp

        For this example, we will use
        the airline passengers dataset.

        .. code-block:: python

            import verticapy.datasets as vpd

            data = vpd.load_airline_passengers()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_airline_passengers.html

        .. ipython:: python
            :suppress:

            import verticapy.datasets as vpd
            data = vpd.load_airline_passengers()

        First we import the model:

        .. ipython:: python

            from verticapy.machine_learning.vertica.tsa import ARIMA

        Then we can create the model:

        .. ipython:: python
            :okwarning:

            model = ARIMA(order = (12, 1, 2))

        We can now fit the model:

        .. ipython:: python
            :okwarning:

            model.fit(data, "date", "passengers")

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

        .. important::

            For this example, a specific model is
            utilized, and it may not correspond
            exactly to the model you are working
            with. To see a comprehensive example
            specific to your class of interest,
            please refer to that particular class.

            Examples:
            :py:class:`~verticapy.machine_learning.vertica.tsa.ARIMA`;
            :py:class:`~verticapy.machine_learning.vertica.tsa.ARMA`;
            :py:class:`~verticapy.machine_learning.vertica.tsa.AR`;
            :py:class:`~verticapy.machine_learning.vertica.tsa.MA`;
        """
        if self._ismultivar():
            for i in range(len(self.y)):
                tmp_res = mt.regression_report(
                    f"y_true{i}",
                    f"y_pred{i}",
                    self._evaluation_relation(
                        start=start, npredictions=npredictions, method=method
                    ),
                    metrics=metrics,
                    k=1,
                )
                if i == 0:
                    res = {"index": tmp_res["index"]}
                res[self.y[i]] = tmp_res["value"]
            return TableSample(res)
        else:
            return mt.regression_report(
                "y_true",
                "y_pred",
                self._evaluation_relation(
                    start=start, npredictions=npredictions, method=method
                ),
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
        method: Literal["auto", "forecast"] = "auto",
    ) -> Union[float, TableSample]:
        """
        Computes the model score.

        Parameters
        ----------
        metric: str, optional
            The metric used to compute the score.

            - aic:
                Akaike's Information Criterion

                .. math::

                    AIC = 2k - 2\ln(\hat{L})

            - bic:
                Bayesian Information Criterion

                .. math::

                    BIC = -2\ln(\hat{L}) + k \ln(n)

            - max:
                Max Error.

                .. math::

                    ME = \max_{i=1}^{n} \left| y_i - \hat{y}_i \\right|

            - mae:
                Mean Absolute Error.

                .. math::

                    MAE = \\frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \\right|

            - median:
                Median Absolute Error.

                .. math::

                    MedAE = \\text{median}_{i=1}^{n} \left| y_i - \hat{y}_i \\right|

            - mse:
                Mean Squared Error.

                .. math::

                    MsE = \\frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \\right)^2

            - msle:
                Mean Squared Log Error.

                .. math::

                    MSLE = \\frac{1}{n} \sum_{i=1}^{n} (\log(1 + y_i) - \log(1 + \hat{y}_i))^2

            - r2:
                R squared coefficient.

                .. math::

                    R^2 = 1 - \\frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \\bar{y})^2}

            - r2a:
                R2 adjusted

                .. math::

                    \\text{Adjusted } R^2 = 1 - \\frac{(1 - R^2)(n - 1)}{n - k - 1}

            - qe:
                quantile error, the quantile must be
                included in the name. Example:
                qe50.1% will  return the quantile
                error using q=0.501.

            - rmse:
                Root-mean-squared error

                .. math::

                    RMSE = \sqrt{\\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}

            - var:
                Explained Variance

                .. math::

                    \\text{Explained Variance}   = 1 - \\frac{Var(y - \hat{y})}{Var(y)}
        start: int, optional
            The behavior of the start parameter and its
            range of accepted values depends on whether
            you provide a timeseries-column (``ts``):

            - No provided timeseries-column:
                ``start`` must be an integer
                greater or equal to 0, where
                zero indicates to start
                prediction at the end of the
                in-sample data. If ``start``
                is a positive value, the function
                predicts the values between the
                end of the in-sample data and
                the start index, and then uses
                the predicted values as time
                series inputs for the subsequent
                ``npredictions``.
            - timeseries-column provided:
                ``start`` must be an ``integer``
                greater or equal to ``1`` and
                identifies the index (row) of
                the timeseries-column at which
                to begin prediction. If the
                ``start`` index is greater than
                the number of rows, ``N``, in the
                input data, the function predicts
                the values between ``N`` and
                ``start`` and uses the predicted
                values as time series inputs for
                the subsequent npredictions.

            Default:

            - No provided timeseries-column:
                prediction begins from the
                end of the in-sample data.
            - timeseries-column provided:
                prediction begins from the
                end of the provided input
                data.

        npredictions: int, optional
            ``integer`` greater or equal to ``1``,
            the number of predicted timesteps.
        method: str, optional
            Forecasting method. One of the following:

            - auto:
                the model initially utilizes the true
                values at each step for forecasting.
                However, when it reaches a point where
                it can no longer rely on true values,
                it transitions to using its own
                predictions for further forecasting.
                This method is often referred to as
                "one step ahead" forecasting.
            - forecast:
                the model initiates forecasting from
                an initial value and entirely disregards
                any subsequent true values. This approach
                involves forecasting based solely on the
                model's own predictions and does not
                consider actual observations after the
                start point.

        Returns
        -------
        float
            score.

        Examples
        --------
        We import :py:mod:`verticapy`:

        .. ipython:: python

            import verticapy as vp

        For this example, we will use
        the airline passengers dataset.

        .. code-block:: python

            import verticapy.datasets as vpd

            data = vpd.load_airline_passengers()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_airline_passengers.html

        .. ipython:: python
            :suppress:

            import verticapy.datasets as vpd
            data = vpd.load_airline_passengers()

        First we import the model:

        .. ipython:: python

            from verticapy.machine_learning.vertica.tsa import ARIMA

        Then we can create the model:

        .. ipython:: python
            :okwarning:

            model = ARIMA(order = (12, 1, 2))

        We can now fit the model:

        .. ipython:: python
            :okwarning:

            model.fit(data, "date", "passengers")

        Let's compute the model's score.

        .. ipython:: python
            :okwarning:

            model.score(start = 40, npredictions = 30, method = "forecast")

        .. important::

            For this example, a specific model is
            utilized, and it may not correspond
            exactly to the model you are working
            with. To see a comprehensive example
            specific to your class of interest,
            please refer to that particular class.

            Examples:
            :py:class:`~verticapy.machine_learning.vertica.tsa.ARIMA`;
            :py:class:`~verticapy.machine_learning.vertica.tsa.ARMA`;
            :py:class:`~verticapy.machine_learning.vertica.tsa.AR`;
            :py:class:`~verticapy.machine_learning.vertica.tsa.MA`;
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
            self._evaluation_relation(
                start=start,
                npredictions=npredictions,
                method=method,
            ),
        ]
        if metric in ("aic", "bic") or adj:
            arg += [1]
        if root or adj:
            arg += [True]
        if self._ismultivar():
            res = {"index": [metric]}
            for i in range(len(self.y)):
                arg[0] = f"y_true{i}"
                arg[1] = f"y_pred{i}"
                res[self.y[i]] = [fun(*arg)]
            return TableSample(res)
        else:
            return fun(*arg)

    # Plotting Methods.

    def plot(
        self,
        vdf: Optional[SQLRelation] = None,
        ts: Optional[str] = None,
        y: Optional[SQLColumns] = None,
        start: Optional[int] = None,
        npredictions: int = 10,
        method: Literal["auto", "forecast"] = "auto",
        idx: int = 0,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the model.

        Parameters
        ----------
        vdf: SQLRelation, optional
            Object used to run the prediction.
            You can also specify a customized
            relation, but you must enclose it
            with an alias. For example,
            ``(SELECT 1) x`` is valid, whereas
            ``(SELECT 1)`` and ``SELECT 1``
            are invalid.
        ts: str, optional
            TS (Time Series) :py:class`vDataColumn`
            used to order the data. The
            :py:class`vDataColumn` type must be
            ``date`` (``date``, ``datetime``,
            ``timestamp``...) or numerical.
        y: SQLColumns, optional
            Response column.

            In the case of multivariate analysis,
            it represents a ``list`` of all the
            predictors.
        start: int, optional
            The behavior of the start parameter and its
            range of accepted values depends on whether
            you provide a timeseries-column (``ts``):

            - No provided timeseries-column:
                ``start`` must be an integer
                greater or equal to 0, where
                zero indicates to start
                prediction at the end of the
                in-sample data. If ``start``
                is a positive value, the function
                predicts the values between the
                end of the in-sample data and
                the start index, and then uses
                the predicted values as time
                series inputs for the subsequent
                ``npredictions``.
            - timeseries-column provided:
                ``start`` must be an ``integer``
                greater or equal to ``1`` and
                identifies the index (row) of
                the timeseries-column at which
                to begin prediction. If the
                ``start`` index is greater than
                the number of rows, ``N``, in the
                input data, the function predicts
                the values between ``N`` and
                ``start`` and uses the predicted
                values as time series inputs for
                the subsequent npredictions.

            Default:

            - No provided timeseries-column:
                prediction begins from the
                end of the in-sample data.
            - timeseries-column provided:
                prediction begins from the
                end of the provided input
                data.

        npredictions: int, optional
            ``integer`` greater or equal to ``1``,
            the number of predicted timesteps.
        method: str, optional
            Forecasting method. One of the following:

            - auto:
                the model initially utilizes the true
                values at each step for forecasting.
                However, when it reaches a point where
                it can no longer rely on true values,
                it transitions to using its own
                predictions for further forecasting.
                This method is often referred to as
                "one step ahead" forecasting.
            - forecast:
                the model initiates forecasting from
                an initial value and entirely disregards
                any subsequent true values. This approach
                involves forecasting based solely on the
                model's own predictions and does not
                consider actual observations after the
                start point.
        idx: int, optional
            It represents the index of the
            predictor for which we want to
            draw the TS plot.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any optional parameter to
            pass to the Plotting functions.

        Returns
        -------
        object
            Plotting Object.

        Examples
        --------
        We import :py:mod:`verticapy`:

        .. ipython:: python

            import verticapy as vp

        For this example, we will use
        the airline passengers dataset.

        .. code-block:: python

            import verticapy.datasets as vpd

            data = vpd.load_airline_passengers()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_airline_passengers.html

        .. ipython:: python
            :suppress:

            import verticapy.datasets as vpd
            data = vpd.load_airline_passengers()

        First we import the model:

        .. ipython:: python

            from verticapy.machine_learning.vertica.tsa import ARIMA

        Then we can create the model:

        .. ipython:: python
            :okwarning:

            model = ARIMA(order = (12, 1, 2))

        We can now fit the model:

        .. ipython:: python
            :okwarning:

            model.fit(data, "date", "passengers")

        We can conveniently plot the
        predictions on a line plot
        to observe the efficacy of
        our model:

        .. code-block:: python

            model.plot(data, "date", "passengers", npredictions = 20, start = 140)

        .. ipython:: python
            :suppress:
            :okwarning:

            vp.set_option("plotting_lib", "plotly")
            fig = model.plot(data, "date", "passengers", npredictions = 20, start = 140, width = 650)
            fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arima_plot_1.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/machine_learning_vertica_tsa_arima_plot_1.html

        .. important::

            For this example, a specific model is
            utilized, and it may not correspond
            exactly to the model you are working
            with. To see a comprehensive example
            specific to your class of interest,
            please refer to that particular class.

            Examples:
            :py:class:`~verticapy.machine_learning.vertica.tsa.ARIMA`;
            :py:class:`~verticapy.machine_learning.vertica.tsa.ARMA`;
            :py:class:`~verticapy.machine_learning.vertica.tsa.AR`;
            :py:class:`~verticapy.machine_learning.vertica.tsa.MA`;
        """
        dataset_provided = not (isinstance(vdf, NoneType))
        y_str, n = self.y, len(self.y)
        prediction = self.predict(
            vdf=vdf,
            ts=ts,
            y=y,
            start=start,
            npredictions=npredictions,
            output_standard_errors=True,
            method=method,
            use_index_as_suffix=True,
        )
        if self._ismultivar() and not (0 <= idx < n):
            raise ValueError(
                "Parameter 'idx' represents the index of the predictor for "
                "which we want to compute the feature importance. It should "
                "be between 0 and the total number of predictors minus one"
                f" ({len(self.y) - 1})"
            )
        if self._ismultivar():
            y_str = self.y[idx]
            prediction = prediction[[f"prediction{idx}"]]
        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="TSPlot",
            chart=chart,
            style_kwargs=style_kwargs,
        )
        return vpy_plt.TSPlot(
            vdf=vDataFrame(self.input_relation),
            columns=y_str,
            order_by=self.ts,
            prediction=prediction,
            start=start,
            dataset_provided=dataset_provided,
            method=method,
        ).draw(**kwargs)
