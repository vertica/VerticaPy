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
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import extract_subquery, quote_ident, schema_relation
from verticapy._utils._sql._sys import _executeSQL

from verticapy.errors import ModelError
from verticapy.core.vdataframe.base import vDataFrame
from verticapy.machine_learning.vertica.tsa.arima import ARIMA
from verticapy.machine_learning.vertica.tsa.base import TimeSeriesModelBase

from verticapy.sql.drop import drop

"""
General Classes.
"""


class TimeSeriesByCategory(TimeSeriesModelBase):
    """ """

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
                        vdf=None
                        if isinstance(vdf, NoneType)
                        else vdf.search(f"{self.by} = '{category}'"),
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
        return vDataFrame(sql)
