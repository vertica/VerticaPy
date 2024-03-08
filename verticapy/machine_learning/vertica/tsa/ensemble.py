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
from typing import Literal, Optional, Union

import verticapy._config.config as conf
from verticapy._typing import NoneType, PlottingObject, SQLColumns, SQLRelation
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import extract_subquery, quote_ident, schema_relation
from verticapy._utils._sql._sys import _executeSQL

from verticapy.errors import ModelError
from verticapy.core.vdataframe.base import TableSample, vDataFrame
import verticapy.machine_learning.metrics as mt
from verticapy.machine_learning.vertica.tsa.arima import ARIMA
from verticapy.machine_learning.vertica.tsa.base import TimeSeriesModelBase

from verticapy.sql.drop import drop

"""
General Classes.
"""


class TimeSeriesByCategory(TimeSeriesModelBase):
    """
    This model is built based on multiple base models.
    You should look at the source models to see entire
    examples.

    .. important:: This is still Beta.


    Parameters
    ----------
    name: str, optional
        Name of the model. The  model is stored  in the
        database.
    overwrite_model: bool, optional
        If set to ``True``, training a
        model with the same name as an
        existing model overwrites the
        existing model.
    base_model: TimeSeriesModelBase
        The user should provide a base model which will
        be used for each category. It could be
        - :py:class:`~verticapy.machine_learning.vertica.tsa.ARIMA`
        - :py:class:`~verticapy.machine_learning.vertica.tsa.ARMA`
        - :py:class:`~verticapy.machine_learning.vertica.tsa.AR`
        - :py:class:`~verticapy.machine_learning.vertica.tsa.MA'

    Attributes
    ----------
    Many attributes are created
    during the fitting phase.

    distinct: list
        This provides a sequential list of the categories
        used to build the different models.

    ts: str
        The column name for time stamp.

    y: str
        The column name used for building the model.

    _is_already_stored: bool
        This tells us whether a model is stored in the Vertica
        database.

    _get_model_names: list
        This returns the list of names of the models created.


    Examples
    --------

    The following examples provide a
    basic understanding of usage.

    Initialization
    ^^^^^^^^^^^^^^

    For this example, we will use
    a subset of the amazon dataset.

    .. code-block:: python

        import verticapy.datasets as vpd

        amazon_full = vpd.load_amazon()

    .. raw:: html
        :file: /project/data/VerticaPy/docs/figures/datasets_loaders_load_amazon.html

    .. ipython:: python
        :suppress:

        from verticapy.datasets import load_amazon
        amazon_full = load_amazon()

    We can reduce the number of states for the sake
    of ease in this example:

    .. ipython:: python

        amazon = amazon_full[(amazon_full["state"] == "PERNAMBUCO") | (amazon_full["state"] == "SERGIPE")]

    Now we can setup a base model that will be
    created for each unique state inside the dataset.
    For this example, we use ARIMA.

    .. ipython:: python

        from verticapy.machine_learning.vertica.tsa import ARIMA

        base_model = ARIMA(order = (2, 1, 2))

    Finally we can now initiate our multiple models
    in one go:

    .. ipython:: python

        from verticapy.machine_learning.vertica.tsa.ensemble import TimeSeriesByCategory

        model = TimeSeriesByCategory(base_model = base_model)

    Model Fitting
    ^^^^^^^^^^^^^^^

    We can now fit the model:

    .. ipython:: python
        :okwarning:

        model.fit(amazon, ts = "date", y = "number", by = "state")


    .. important::

        To train a model, you can directly use the
        :py:class:`~vDataFrame` or the name of the
        relation stored in the database. The test
        set is optional and is only used to compute
        the test metrics. In :py:mod:`verticapy`, we
        don't work using ``X`` matrices and ``y``
        vectors. Instead, we work directly with lists
        of predictors and the response name.


    Plots
    ^^^^^^

    We can conveniently plot the
    predictions on a line plot to
    observe the efficacy of our
    model. We need to provide the
    ``idx`` which represents the model number.

    .. code-block:: python

        model.plot(idx = 0, npredictions = 5)

    .. ipython:: python
        :suppress:
        :okwarning:

        vp.set_option("plotting_lib", "plotly")
        fig = model.plot(idx = 0, npredictions = 5)
        fig.write_html("/project/data/VerticaPy/docs/figures/machine_learning_vertica_tsa_ensemble_timeseriesbycategory_1.html")

    .. raw:: html
        :file: /project/data/VerticaPy/docs/figures/machine_learning_vertica_tsa_ensemble_timeseriesbycategory_1.html

    .. note::

        You can find out the name of the category by
        the ``distinct`` attribute. The sequential list of
        categories correspond to ``idx = 0, 1 ...``.
        ``model.distinct``.
    """

    # Properties.

    @property
    def _vertica_fit_sql(self) -> str:
        return self.parameters["base_model"]._vertica_fit_sql

    @property
    def _vertica_predict_sql(self) -> str:
        return self.parameters["base_model"]._vertica_predict_sql

    @property
    def _model_subcategory(self) -> Literal["TIMESERIES"]:
        return "TIMESERIES"

    @property
    def _model_type(self) -> Literal["TimeSeriesByCategory"]:
        return "TimeSeriesByCategory"

    # System & Special Methods.

    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        base_model: Optional[TimeSeriesModelBase] = None,
    ) -> None:
        ...
        if isinstance(base_model, NoneType):
            base_model = ARIMA()
        super().__init__(name, overwrite_model)
        self.parameters = {
            "base_model": base_model,
        }
        self.models_ = []

    def _is_already_stored(
        self,
        **kwargs,
    ) -> bool:
        """
        Checks whether the
        model is stored in
        the Vertica database.

        Returns
        -------
        bool
            ``True`` if the model is
            stored in the Vertica
            database.
        """
        return len(self._get_model_names()) > 0

    def _get_model_names(self):
        """
        Returns the list of models
        used to build the final one.
        """
        name = str(self.model_name).replace("'", "''")
        sql = f"""
            SELECT 
                model_name 
            FROM models 
            WHERE model_name LIKE '{name}%' 
              AND REGEXP_LIKE(model_name, '(.)+\_([0-9])+\_tsbc');
        """
        res = _executeSQL(
            sql,
            title="Finding all the input models.",
            method="fetchall",
        )
        res = [quote_ident(l[0]) for l in res]
        return res

    def drop(self):
        """
        Drops the model from
        the Vertica database.
        """
        names = self._get_model_names()
        for name in names:
            drop(name, method="model")

    # Model Fitting Method.

    def fit(
        self,
        input_relation: SQLRelation,
        ts: str,
        y: str,
        by: str,
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
        y: str
            Response column.
        by: str
            Column used to represent the different
            categories. The number of categories
            will define the number of models.
            The ``by`` column must not have more
            than 50 categories.
        test_relation: SQLRelation, optional
            Relation used to test the model.
        return_report: bool, optional
            [For native models]
            When set to ``True``, the model
            summary will be returned. Otherwise,
            it will be printed.
            In case of ``TimeSeriesByCategory``,
            the report of all the models for each
            category are merged together.

        Returns
        -------
        str
            model's summary.

        Examples
        --------
        This model is built based on multiple base models.
        You should look at the source models to see entire
        examples.

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
        if not (isinstance(input_relation, vDataFrame)):
            input_relation = vDataFrame(input_relation)
        distinct = input_relation[by].distinct()
        if len(distinct) > 50:
            raise ValueError(
                "The number of distinct categories must be lesser than 50."
            )
        self.ts = quote_ident(ts)
        self.y = quote_ident(y)
        self.by = quote_ident(by)
        self.distinct = copy.deepcopy(distinct)
        report = ""
        # Fitting
        for idx, cat in enumerate(distinct):
            if cat is not None:
                category = str(cat).replace("'", "''")
                number = str(idx).zfill(3)
                schema, name = schema_relation(self.model_name)
                model = copy.deepcopy(self.parameters["base_model"])
                model.model_name = schema + "." + name[:-1] + f'_{number}_tsbc"'
                report += f"For category: {cat}\n"
                report += model.fit(
                    input_relation=input_relation.search(f"{self.by} = '{category}'"),
                    ts=ts,
                    y=y,
                    test_relation=None,
                    return_report=True,
                )
                report += "\n\n"
                self.models_ += [model]
        if return_report:
            return report
        if conf.get_option("print_info"):
            print(report)
        return None

    # I/O Methods.

    def deploySQL(
        self,
        vdf: Optional[SQLRelation] = None,
        ts: Optional[str] = None,
        y: Optional[str] = None,
        start: Optional[int] = None,
        npredictions: int = 10,
        freq: Literal[None, "m", "months", "y", "year", "infer"] = "infer",
        filter_step: Optional[int] = None,
        method: Literal["auto", "forecast"] = "auto",
    ) -> str:
        """
        Returns the SQL code
        needed to deploy the
        model.

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
        y: str, optional
            Response column.
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

        Returns
        -------
        str
            the SQL code needed
            to deploy the model.

        Examples
        --------
        This model is built based on multiple base models.
        You should look at the source models to see entire
        examples.

        :py:class:`~verticapy.machine_learning.vertica.tsa.ARIMA`;
        :py:class:`~verticapy.machine_learning.vertica.tsa.ARMA`;
        :py:class:`~verticapy.machine_learning.vertica.tsa.AR`;
        :py:class:`~verticapy.machine_learning.vertica.tsa.MA`;
        """
        n = len(self.models_)
        if n == 0:
            raise ModelError("No model is yet created.")
        all_predictions = []
        for idx, model in enumerate(self.models_):
            category = str(self.distinct[idx]).replace("'", "''")
            all_predictions += [
                extract_subquery(
                    model.predict(
                        vdf=(
                            None
                            if isinstance(vdf, NoneType)
                            else vdf.search(f"{self.by} = '{category}'")
                        ),
                        ts=ts,
                        y=y,
                        start=start,
                        npredictions=npredictions,
                        freq=freq,
                        output_estimated_ts=True,
                    ).current_relation()
                )
            ]
        all_predictions = [
            f"({table}) t{idx}" for idx, table in enumerate(all_predictions)
        ]
        all_predictions = [
            table if idx == 0 else f"{table} USING ({self.ts})"
            for idx, table in enumerate(all_predictions)
        ]
        sql = "SELECT t0.date, " + ", ".join(
            [f"t{i}.prediction AS prediction{i}" for i in range(n)]
        )
        sql += " FROM " + " INNER JOIN ".join(all_predictions)
        return sql

    # Features Importance Methods.

    def features_importance(
        self,
        idx: int = 0,
        show: bool = True,
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Computes the input submodel's
        features importance.

        Parameters
        ----------
        idx: int, optional
            As the ``TimeSeriesByCategory``
            model generates multiple models,
            the importance of features varies
            for each submodel. The ``idx``
            parameter corresponds to the
            submodel index.
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
        This model is built based on multiple base models.
        You should look at the source models to see entire
        examples.

        :py:class:`~verticapy.machine_learning.vertica.tsa.ARIMA`;
        :py:class:`~verticapy.machine_learning.vertica.tsa.ARMA`;
        :py:class:`~verticapy.machine_learning.vertica.tsa.AR`;
        :py:class:`~verticapy.machine_learning.vertica.tsa.MA`;
        """
        n = len(self.models_)
        if not (0 <= idx < n):
            raise IndexError(
                "Index out of range: You should use "
                f"an index value between 0 and {n-1}."
            )
        return self.models_[idx].features_importance(
            show=show, chart=chart, **style_kwargs
        )

    # Prediction / Transformation Methods.

    def predict(
        self,
        vdf: Optional[SQLRelation] = None,
        ts: Optional[str] = None,
        y: Optional[str] = None,
        start: Optional[int] = None,
        npredictions: int = 10,
        freq: Literal[None, "m", "months", "y", "year", "infer"] = "infer",
        filter_step: Optional[int] = None,
        method: Literal["auto", "forecast"] = "auto",
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
        y: str, optional
            Response column.
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

        Returns
        -------
        vDataFrame
            a new object.

        Examples
        --------
        This model is built based on multiple base models.
        You should look at the source models to see entire
        examples.

        :py:class:`~verticapy.machine_learning.vertica.tsa.ARIMA`;
        :py:class:`~verticapy.machine_learning.vertica.tsa.ARMA`;
        :py:class:`~verticapy.machine_learning.vertica.tsa.AR`;
        :py:class:`~verticapy.machine_learning.vertica.tsa.MA`;
        """
        sql = self.deploySQL(
            vdf=vdf,
            ts=ts,
            y=y,
            start=start,
            npredictions=npredictions,
            freq=freq,
            filter_step=filter_step,
            method=method,
        )
        return vDataFrame(sql)

    # Model Evaluation Methods.

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
        This model is built based on multiple base models.
        You should look at the source models to see entire
        examples.

        :py:class:`~verticapy.machine_learning.vertica.tsa.ARIMA`;
        :py:class:`~verticapy.machine_learning.vertica.tsa.ARMA`;
        :py:class:`~verticapy.machine_learning.vertica.tsa.AR`;
        :py:class:`~verticapy.machine_learning.vertica.tsa.MA`;
        """
        raise NotImplementedError

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
    ) -> float:
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
        This model is built based on multiple base models.
        You should look at the source models to see entire
        examples.

        :py:class:`~verticapy.machine_learning.vertica.tsa.ARIMA`;
        :py:class:`~verticapy.machine_learning.vertica.tsa.ARMA`;
        :py:class:`~verticapy.machine_learning.vertica.tsa.AR`;
        :py:class:`~verticapy.machine_learning.vertica.tsa.MA`;
        """
        raise NotImplementedError

    # Plotting Methods.

    def plot(
        self,
        idx: int = 0,
        vdf: Optional[SQLRelation] = None,
        ts: Optional[str] = None,
        y: Optional[str] = None,
        start: Optional[int] = None,
        npredictions: int = 10,
        method: Literal["auto", "forecast"] = "auto",
        chart: Optional[PlottingObject] = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the input submodel.

        Parameters
        ----------
        idx: int, optional
            As the ``TimeSeriesByCategory``
            model generates multiple models,
            the final plot varies for each
            submodel. The ``idx`` parameter
            corresponds to the submodel index.
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
        y: str, optional
            Response column.
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
        This model is built based on multiple base models.
        You should look at the source models to see entire
        examples.

        :py:class:`~verticapy.machine_learning.vertica.tsa.ARIMA`;
        :py:class:`~verticapy.machine_learning.vertica.tsa.ARMA`;
        :py:class:`~verticapy.machine_learning.vertica.tsa.AR`;
        :py:class:`~verticapy.machine_learning.vertica.tsa.MA`;
        """
        n = len(self.models_)
        if not (0 <= idx < n):
            raise IndexError(
                "Index out of range: You should use "
                f"an index value between 0 and {n-1}."
            )
        return self.models_[idx].plot(
            vdf=vdf,
            ts=ts,
            y=y,
            start=start,
            npredictions=npredictions,
            method=method,
            chart=chart,
            **style_kwargs,
        )
