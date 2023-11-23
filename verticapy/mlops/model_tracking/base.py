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
import warnings
import numpy as np
import uuid
from typing import Literal

from verticapy._typing import (
    SQLRelation,
    SQLColumns,
)

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame
from verticapy.machine_learning.vertica.base import VerticaModel
from verticapy.machine_learning.vertica.model_management import load_model
from verticapy.sql.create import create_table
from verticapy.sql.drop import drop
from verticapy.plotting._utils import PlottingUtils
from verticapy._typing import PlottingObject
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._sys import _executeSQL

from verticapy._utils._sql._format import quote_ident, schema_relation

from verticapy.machine_learning.metrics.classification import (
    roc_auc_score,
    prc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import verticapy.sql.sys as sys


class vExperiment(PlottingUtils):
    """
    Creates a vExperiment object that can be used for tracking
    native vertica models trained as part of an experiment.

    Parameters
    ----------
    experiment_name: str
        The name of the experiment
    test_relation: SQLRelation
        Relation to use to test models in the experiment.
        It would be ignored for experiments of type clustering.
    X: SQLColumns
        List of the predictors.
        It would be ignored for experiments of type clustering.
    y: str
        Response column.
        It would be ignored for experiments of type clustering.
    experiment_type: str, optional
        The experiment type.
            auto      : Automatically     detects     the
                        experiment type from test_relation.
            regressor : The regression models can be added
                        to the experiment.
            binary    : The binary classification models
                        can be added to the experiment.
            multi     : The multiclass classification models
                        can be added to the experiment.
            clustering: The clustering models can be added
                        to the experiment.
    experiment_table: SQLRelation, optional
        The name of table ([schema_name.]table_name) in
        the database to archive the experiment.
        When not specified, the experiment will not be
        backed up in the database.
        When specified, the table will be created if it
        doesn't exist. In case that the table already exists,
        the user must have SELECT, INSERT, and DELETE
        privileges on the table.

    Attributes
    ----------
    _model_name_list: list
        The list of model names added to the experiment.
    _model_id_list: list
        The list of model IDs added to the experiment.
    _model_type_list: list
        The list of model types added to the experiment.
    _parameters: list
        The list of dictionaries of parameters of each added model.
    _measured_metrics: list
        The list of list of measured metrics for each added model.
    _metrics: list
        The list of metrics to be used for evaluating each model.
        This list will be determined based on the value of
        experiment_type at the time of object creation.
        Each metric is paired with 1 or -1 where 1 indicates a
        positive correlationthe between the value of the metric
        positive correlationthe between the value of the metric
        and the quality of the model. In contrast, number -1
        indicates a negative correlation.
    _user_defined_metrics: list
        The list of dictionaries of user-defined metrics.
    """

    _experiment_columns = {
        "experiment_name": "varchar(128)",
        "experiment_type": "varchar(32)",
        "model_id": "int",
        "user_id": "int",
        "parameters": "varchar(2048)",
        "measured_metrics": "varchar(2048)",
        "user_defined_metrics": "varchar(2048)",
    }

    _regressor_metrics = [
        ("explained_variance", 1),
        ("max_error", -1),
        ("median_absolute_error", -1),
        ("mean_absolute_error", -1),
        ("mean_squared_error", -1),
        ("root_mean_squared_error", -1),
        ("r2", 1),
        ("r2_adj", 1),
        ("aic", -1),
        ("bic", -1),
    ]

    _binary_metrics = [
        ("auc", 1),
        ("prc_auc", 1),
        ("accuracy", 1),
        ("log_loss", -1),
        ("precision", 1),
        ("recall", 1),
        ("f1_score", 1),
        ("mcc", 1),
        ("informedness", 1),
        ("markedness", 1),
        ("csi", -1),
    ]

    # TODO: commented metrics should be returned back when the problem of roc and prc is resolved
    _multi_metrics = [
        # ("micro_roc_auc",1), ("micro_prc_auc",1),
        ("micro_accuracy", 1),
        ("micro_precision", 1),
        ("micro_recall", 1),
        ("micro_f1_score", 1),
        # ("macro_roc_auc",1), ("macro_prc_auc",1),
        ("macro_accuracy", 1),
        ("macro_precision", 1),
        ("macro_recall", 1),
        ("macro_f1_score", 1),
        # ("weighted_roc_auc",1), ("weighted_prc_auc",1),
        ("weighted_accuracy", 1),
        ("weighted_precision", 1),
        ("weighted_recall", 1),
        ("weighted_f1_score", 1),
    ]

    @save_verticapy_logs
    def __init__(
        self,
        experiment_name: str,
        test_relation: SQLRelation,
        X: SQLColumns,
        y: str,
        experiment_type: Literal[
            "auto", "regressor", "binary", "multi", "clustering"
        ] = "auto",
        experiment_table: str = "",
    ) -> None:
        self.experiment_name = experiment_name
        self.test_relation = test_relation
        self.X = X
        self.y = y
        self.experiment_type = experiment_type.lower()
        self.experiment_table = experiment_table

        self._model_name_list = []
        self._model_id_list = []
        self._model_type_list = []
        self._parameters = []
        self._measured_metrics = []
        self._metrics = []
        self._user_defined_metrics = []

        # if there is already a saved experiment in experiment_table,
        # it will determine experiment_type and its info will be loaded.
        if self.experiment_table:
            self._load_or_create_experiment_table()
        else:
            warning_message = (
                "The experiment will not be backed up in the database "
                "when experiment_table is not specified."
            )
            warnings.warn(warning_message, Warning)

        if self.experiment_type == "clustering" or (
            self.experiment_type == "auto"
            and (test_relation is None or X is None or y is None)
        ):
            self.experiment_type = "clustering"
            return

        if not (self.test_relation and X and y):
            raise ValueError(
                "test_relation, X, and y must be specified except for experiments of type clustering"
            )

        if not isinstance(test_relation, vDataFrame):
            self.test_relation = vDataFrame(test_relation)

        columns = self.test_relation.get_columns()
        if not (all(quote_ident(x) in columns for x in X)):
            raise ValueError("not all columns of X are available in test_relation")
        if not (quote_ident(y) in columns):
            raise ValueError("y is not available in test_relation")

        if self.experiment_type == "auto":
            # finding experiment_type from the content of test_relation
            if self._is_regressor():
                self.experiment_type = "regressor"
                self._metrics = self._regressor_metrics
            elif self._is_binary():
                self.experiment_type = "binary"
                self._metrics = self._binary_metrics
            else:
                self.experiment_type = "multi"
                self._metrics = self._multi_metrics

        elif self.experiment_type == "regressor":
            self._is_regressor(raise_error=True)
            self._metrics = self._regressor_metrics
        elif self.experiment_type == "binary":
            self._is_binary(raise_error=True)
            self._metrics = self._binary_metrics
        elif self.experiment_type == "multi":
            self._metrics = self._multi_metrics
        else:
            raise ValueError(
                f"Parameter 'experiment_type` must be in auto|binary|multi|regressor|clustering. "
                f"Found {self.experiment_type}."
            )

    def add_model(self, model: VerticaModel, metrics: dict = None) -> None:
        """
        Adds a model to the experiment.
        It will throw an error if the model type is not compatible with experiment_type

        Parameters
        ----------
        model: VerticaModel
            The model that is added in this experiment.
        metrics: dict
            The optional dictionary of metric names to their values used for evaluating the model.

        Returns
        -------
        None
        """
        # evaluating if model can be added
        if not model._is_native:
            raise AttributeError("Only native models can be added to an experiment.")

        experiment_category = self.experiment_type
        if experiment_category == "binary" or experiment_category == "multi":
            experiment_category = "classifier"
        model_subcategory = (model._model_subcategory).lower()
        if experiment_category != model_subcategory:
            raise ValueError(
                f"Only {experiment_category} models can be added to this experiment."
            )

        if metrics:
            # No user defined metric should be named the same as a standard one.
            # Besides, their keys must be string and their values numeric
            for ud_metric in metrics.keys():
                if (ud_metric, 1) in self._metrics or (ud_metric, -1) in self._metrics:
                    raise ValueError(
                        f"A user defined metric must not be named the same as "
                        f"a standard metric '{ud_metric}'."
                    )
                if not isinstance(ud_metric, str):
                    raise ValueError(
                        f"The name of a user defined metric must be string, but "
                        f"{ud_metric} is provided."
                    )
                if not type(metrics[ud_metric]) in [int, float]:
                    raise ValueError(
                        f"The value of a user defined metric must be numeric, but "
                        f"the value of{ud_metric} is {metrics[ud_metric]}."
                    )

        # finding model_id
        model_id = model._get_vertica_model_id()
        if model_id == 0:
            raise ValueError(
                f"model {model.model_name} does not exist in the database."
            )

        # collecting model parameters
        model_parameters = model.get_params()

        # measuring metrics
        measured_metrics = []
        if self.experiment_type != "clustering":
            model.test_relation = self.test_relation
            if self.experiment_type == "regressor" or self.experiment_type == "binary":
                report = model.report()
                measured_metrics = report.values["value"]
            else:  # self.experiment_type == "multi"
                average_methods = ["micro", "macro", "weighted"]
                # calculating y_score and store it in a temp table will help performance
                test_vdf = self.test_relation.copy()
                model.predict(vdf=test_vdf, X=self.X, name="y_score")
                prediction_vdf = test_vdf.select([self.y, "y_score"])
                temp_table_name = "temp" + str(uuid.uuid1()).replace("-", "")
                prediction_vdf.to_db(
                    name=temp_table_name, relation_type="local", inplace=True
                )

                for avg_method in average_methods:
                    # TODO: roc_auc_score and prc_auc_score are commented out until
                    # their problem is resolved
                    """
                    m_metric = roc_auc_score(y_true = self.y,
                                             y_score = "y_score",
                                             input_relation = prediction_vdf,
                                             average = avg_method,
                                             labels = model.classes_)
                    measured_metrics.append(m_metric)

                    m_metric = prc_auc_score(y_true = self.y,
                                             y_score = "y_score",
                                             input_relation = prediction_vdf,
                                             average = avg_method,
                                             labels = model.classes_)
                    measured_metrics.append(m_metric)
                    """
                    m_metric = accuracy_score(
                        y_true=self.y,
                        y_score="y_score",
                        input_relation=prediction_vdf,
                        average=avg_method,
                        labels=model.classes_,
                    )
                    measured_metrics.append(m_metric)

                    m_metric = precision_score(
                        y_true=self.y,
                        y_score="y_score",
                        input_relation=prediction_vdf,
                        average=avg_method,
                        labels=model.classes_,
                    )
                    measured_metrics.append(m_metric)

                    m_metric = recall_score(
                        y_true=self.y,
                        y_score="y_score",
                        input_relation=prediction_vdf,
                        average=avg_method,
                        labels=model.classes_,
                    )
                    measured_metrics.append(m_metric)

                    m_metric = f1_score(
                        y_true=self.y,
                        y_score="y_score",
                        input_relation=prediction_vdf,
                        average=avg_method,
                        labels=model.classes_,
                    )
                    measured_metrics.append(m_metric)

                drop(name=temp_table_name, method="table")

        # the model will not be added if any of the above steps fail
        self._model_name_list.append(model.model_name)
        self._model_id_list.append(model_id)
        self._model_type_list.append(model._model_type)
        self._parameters.append(model_parameters)
        self._measured_metrics.append(measured_metrics)
        self._user_defined_metrics.append(metrics)

        self._save_experiment(model_id, model_parameters, measured_metrics, metrics)

    def list_models(self) -> TableSample:
        values_table = {
            "model_name": self._model_name_list,
            "model_type": self._model_type_list,
            "model_parameters": self._parameters,
        }

        for index, metric in enumerate(self._metrics):
            metric_values = []
            for values_list in self._measured_metrics:
                metric_values.append(values_list[index])
            values_table[metric[0]] = metric_values

        values_table["user_defined_metrics"] = self._user_defined_metrics

        return TableSample(values_table)

    def load_best_model(self, metric: str) -> VerticaModel:
        if len(self._model_name_list) == 0:
            return None

        max_value = float("-inf")
        max_index = -1

        if (metric, 1) in self._metrics or (metric, -1) in self._metrics:
            # search the list of the standard metrics for the requested metric
            if (metric, 1) in self._metrics:
                metric_index = self._metrics.index((metric, 1))
                metric_sign = 1
            else:
                metric_index = self._metrics.index((metric, -1))
                metric_sign = -1

            for index, item in enumerate(self._measured_metrics):
                if (item[metric_index] * metric_sign) > max_value:
                    max_value = item[metric_index] * metric_sign
                    max_index = index
        else:
            # search the list of user defined metrics
            for index, item in enumerate(self._user_defined_metrics):
                if item is not None and metric in item.keys():
                    if item[metric] > max_value:
                        max_value = item[metric]
                        max_index = index

        if max_index == -1:
            raise ValueError(
                f"Cannot find a metric named {metric} for "
                f"this experiment of type {self.experiment_type}."
            )

        best_model = load_model(name=self._model_name_list[max_index])
        return best_model

    def plot(
        self,
        parameter: str,
        metric: str,
        chart: PlottingObject = None,
        **style_kwargs,
    ) -> PlottingObject:
        """
        Draws the scatter plot of a metric vs a parameter

        Parameters
        ----------
        parameter: str
            The name of parameter used by the models in the experiment
        metric: str
            The name of metric used in measuring the quality of the models
            in the experiment
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional  parameter  to pass to the  plotting functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        data_points = []

        if (metric, 1) in self._metrics or (metric, -1) in self._metrics:
            # it is a standard metric
            if (metric, 1) in self._metrics:
                metric_index = self._metrics.index((metric, 1))
            else:
                metric_index = self._metrics.index((metric, -1))

            for model_index, item in enumerate(self._measured_metrics):
                if parameter in self._parameters[model_index].keys():
                    data_points.append(
                        (self._parameters[model_index][parameter], item[metric_index])
                    )
        else:
            # it is a user defined metric
            for model_index, item in enumerate(self._user_defined_metrics):
                if (metric in item.keys()) and (
                    parameter in self._parameters[model_index].keys()
                ):
                    data_points.append(
                        (self._parameters[model_index][parameter], item[metric])
                    )

        if len(data_points) == 0:
            raise ValueError(
                "Could not find any datapoint for the provided pair of "
                "(parameter, metric) = ({parameter}, {metric})."
            )

        data = {"X": np.array(data_points), "s": None, "c": None}
        layout = {
            "columns": [f"parameter={parameter}", f"metric={metric}"],
            "size": None,
            "c": None,
            "has_category": False,
            "has_cmap": False,
            "has_size": False,
        }

        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="ScatterPlot",
            chart=chart,
            style_kwargs=style_kwargs,
        )

        return vpy_plt.ScatterPlot(data=data, layout=layout).draw(**kwargs)

    def drop(self, keeping_models: list = None) -> bool:
        """
        Drops all models of the experiment except those in the keeping_models list.
        It also clears the info of models saved in the attributes, and drops the
        experiment_table if it is specified.
        """
        for model in self._model_name_list:
            if (keeping_models is None) or (not model in keeping_models):
                drop(name=model, method="model")

        self._model_name_list.clear()
        self._model_id_list.clear()
        self._model_type_list.clear()
        self._parameters.clear()
        self._measured_metrics.clear()
        self._user_defined_metrics.clear()

        drop(name=self.experiment_table, method="table")

    def __str__(self) -> str:
        """
        Returns the model Representation.
        """
        return self.__repr__()

    def __repr__(self) -> str:
        """
        Returns the model Representation.
        """
        return f"<experiment_name: {self.experiment_name}, experiment_type: {self.experiment_type}>"

    def _load_or_create_experiment_table(self):
        """
        Loads previous experiment info from the experiment table if it exists;
        creates the table otherwise.
        """
        schema, relation = schema_relation(self.experiment_table, do_quote=False)
        if sys.does_table_exist(relation, schema):
            # does the relation qualify to be an experiment table
            self._evaluate_experiment_table(table_name=relation, schema=schema)

            # load previously saved experiment
            self._load_experiment_table(table_name=relation, schema=schema)
        else:
            # create the experiment table
            create_table(
                table_name=quote_ident(relation),
                schema=quote_ident(schema),
                dtype=self._experiment_columns,
                temporary_local_table=False,
                raise_error=True,
            )

    def _evaluate_experiment_table(self, table_name: str, schema: str) -> None:
        """
        Evaluates if schema.table_name meets the criteria of being an experiment table.
        """
        # does user has required privileges?
        sys.has_privileges(
            table_name, schema, ["SELECT", "INSERT", "DELETE"], raise_error=True
        )

        # does the table have the expected columns?
        vdf = vDataFrame(input_relation=table_name, schema=schema)
        table_columns = vdf.get_columns()
        for index, item in enumerate(table_columns):
            table_columns[index] = item.lower()

        for column in self._experiment_columns:
            if not quote_ident(column) in table_columns:
                raise ValueError(
                    f"Table {schema}.{table_name} does not have the "
                    f"{column} column required for an experiment table."
                )

    def _load_experiment_table(self, table_name: str, schema: str) -> None:
        """
        Loads schema.table_name as an experiment table from the database.
        """
        if self.experiment_type == "auto":
            # trying to find the type from the experiment table
            query_type = (
                f"SELECT experiment_type FROM {schema}.{table_name} "
                f"WHERE experiment_name='{self.experiment_name}' LIMIT 1;"
            )
            exp_type = _executeSQL(
                query_type, title="finding experiment type", method="fetchrow"
            )
            if not exp_type:
                # there is no history
                return
            self.experiment_type = exp_type[0]

        query_load = (
            f"SELECT model_id, parameters, measured_metrics, user_defined_metrics "
            f"FROM {schema}.{table_name} WHERE experiment_name='{self.experiment_name}' "
            f"AND experiment_type='{self.experiment_type}';"
        )
        ts = TableSample.read_sql(query_load, title="loading experiment table")
        if len(ts.values["model_id"]) == 0:
            # there is no history
            return

        for index, model_id in enumerate(ts.values["model_id"]):
            query_model = (
                f"SELECT schema_name, model_name FROM models WHERE model_id={model_id};"
            )
            res = _executeSQL(query_model, title="finding model", method="fetchrow")
            if res:
                try:
                    model_name = res[0] + "." + res[1]
                    model_object = load_model(name=model_name)
                    parameters = eval(ts.values["parameters"][index])
                    measured_metrics = eval(ts.values["measured_metrics"][index])
                    ud_metrics = eval(ts.values["user_defined_metrics"][index])
                    # the model will be ignored if any of the above eval operations fails
                    self._model_name_list.append(model_name)
                    self._model_id_list.append(model_id)
                    self._model_type_list.append(model_object._model_type)
                    self._parameters.append(parameters)
                    self._measured_metrics.append(measured_metrics)
                    self._user_defined_metrics.append(ud_metrics)
                except:
                    pass

    def _is_regressor(self, raise_error: bool = False) -> bool:
        """
        Checks whether test_relation is suitable for an experiment
        of type regressor.

        Parameters
        ----------
        raise_error: bool, optional
            If set to True and an error occurs, raises the error.

        Returns
        -------
        bool
            True if test_relation is suitable; False otherwise.
        """
        if not self.test_relation[self.y].isnum():
            if raise_error:
                raise ValueError(
                    "In an experiment of type regressor,"
                    " the y column must be numerical"
                )
            else:
                return False

        return True

    def _is_binary(self, raise_error: bool = False) -> bool:
        """
        Checks whether test_relation is suitable for an experiment
        of type binary.

        Parameters
        ----------
        raise_error: bool, optional
            If set to True and an error occurs, raises the error.

        Returns
        -------
        bool
            True if test_relation is suitable; False otherwise.
        """
        if sorted(self.test_relation[self.y].distinct()) != [0, 1]:
            if raise_error:
                raise ValueError(
                    "In an experiment of type binary,"
                    " the y column must have only two distinct values"
                )
            else:
                return False

        return True

    def _save_experiment(
        self,
        model_id: int,
        model_parameters: dict,
        measured_metrics: list,
        user_defined_metrics: dict,
    ) -> None:
        if not self.experiment_table:
            return

        user_id = _executeSQL(
            "SELECT user_id FROM users WHERE user_name=current_user();",
            title="finding user_id",
            method="fetchfirstelem",
        )

        experiment_table = self.experiment_table.replace("'", "''")
        experiment_name = self.experiment_name.replace("'", "''")
        experiment_type = self.experiment_type.replace("'", "''")
        model_parameters_str = (model_parameters.__str__()).replace("'", "''")
        measured_metrics_str = (measured_metrics.__str__()).replace("'", "''")
        user_defined_metrics_str = (user_defined_metrics.__str__()).replace("'", "''")

        inser_query = (
            f"INSERT INTO {experiment_table} VALUES ('{experiment_name}',"
            f"'{experiment_type}', {model_id}, {user_id},"
            f"'{model_parameters_str}', '{measured_metrics_str}',"
            f"'{user_defined_metrics_str}');"
        )
        _executeSQL(inser_query, title="inserting experiment info")
        _executeSQL("COMMIT;", title="commiting the insert of experiment info")
