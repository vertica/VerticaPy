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
import warnings
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

from verticapy.machine_learning.metrics.classification import (
    roc_auc_score,
    prc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

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
    model_name_list_: list
        The list of model names added to the experiment.
    model_id_list_: list
        The list of model IDs added to the experiment.
    model_type_list_: list
        The list of model types added to the experiment.
    parameters_: list
        The list of dictionaries of parameters of each added model.
    measured_metrics_: list
        The list of list of measured metrics for each added model.
    metrics_: list
        The list of metrics to be used for evaluating each model.
        This list will be determined based on the value of
        experiment_type at the time of object creation.
    user_defined_metrics_: list
        The list of dictionaries of user-defined metrics.
    """
    _experiment_columns = {"experiment_name" : "varchar(128)",
                           "experiment_type" : "varchar(32)",
                           "model_id" : "int",
                           "user_id" : "int",
                           "parameters" : "varchar(2048)",
                           "measured_metrics" : "varchar(2048)",
                           "user_defined_metrics" : "varchar(2048)"}

    _regressor_metrics = ["explained_variance", "max_error", "median_absolute_error",
                          "mean_absolute_error", "mean_squared_error", "root_mean_squared_error",
                          "r2", "r2_adj", "aic", "bic"]
    _binary_metrics = ["auc", "prc_auc", "accuracy", "log_loss", "precision", "recall", "f1_score",
                       "mcc", "informedness", "markedness", "csi"]
    _multi_metrics = ["micro_roc_auc", "micro_prc_auc", "micro_accuracy",
                      "micro_precision", "micro_recall", "micro_f1_score",
                      "macro_roc_auc", "macro_prc_auc", "macro_accuracy",
                      "macro_precision", "macro_recall", "macro_f1_score",
                      "weighted_roc_auc", "weighted_prc_auc", "weighted_accuracy",
                      "weighted_precision", "weighted_recall", "weighted_f1_score"]

    @save_verticapy_logs
    def __init__(
        self,
        experiment_name: str,
        test_relation: SQLRelation,
        X: SQLColumns,
        y: str,
        experiment_type: Literal["auto", "regressor", "binary", "multi", "clustering"] = "auto",
        experiment_table: str = ""
    ) -> None:
        self.experiment_name = experiment_name
        self.test_relation = test_relation
        self.X = X
        self.y = y
        self.experiment_type = experiment_type.lower()
        self.experiment_table = experiment_table

        self.model_name_list_ = []
        self.model_id_list_ = []
        self.model_type_list_ = []
        self.parameters_ = []
        self.measured_metrics_ = []
        self.metrics_ = []
        self.user_defined_metrics_ = []

        # if there is already a saved experiment in experiment_table,
        # it will determine experiment_type and its info will be loaded.
        if self.experiment_table:
            self._load_or_create_experiment_table()
        else:
            warning_message = "The experiment will not be backed up in the database "
            "when experiment_table is not specified."
            warnings.warn(warning_message, Warning)

        if (self.experiment_type == "clustering") or
           (self.experiment_type == "auto" and not (test_relation and X and y)):
            self.experiment_type == "clustering"
            return

        if not (self.test_relation and X and y):
            raise("test_relation, X, and y must be specified except for experiments of type clustering")

        if not isinstance(test_relation, vDataFrame):
            self.test_relation = vDataFrame(test_relation)

        columns = self.test_relation[y].get_columns()
        if not (all(x in columns for x in X)):
            raise("not all columns of X are available in test_relation")
        if not (y in columns):
            raise("y is not available in test_relation")

        if self.experiment_type == "auto":
            # finding experiment_type from the content of test_relation
            if self._is_regressor():
                self.experiment_type = "regressor"
                self.metrics_ = _regressor_metrics
            elif self._is_binary():
                self.experiment_type = "binary"
                self.metrics_ = _binary_metrics
            else:
                self.experiment_type = "multi"
                self.metrics_ = _multi_metrics

        elif self.experiment_type == "regressor":
            self._is_regressor(raise_error=True)
            self.metrics_ = _regressor_metrics
        elif self.experiment_type == "binary":
            self._is_binary(raise_error=True)
            self.metrics_ = _binary_metrics
        elif self.experiment_type == "multi":
            self.metrics_ = _multi_metrics
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
        if (not model._is_native):
            raise AttributeError("Only native models can be added to an experiment.")

        experiment_category = self.experiment_type
        if (experiment_category == "binary" or experiment_category == "multi"):
            experiment_category = "classifier"
        model_subcategory = (model._model_subcategory).lower()
        if (experiment_category != model_subcategory):
            raise ValueError(f"Only {experiment_category} models can be added to this experiment.")

        if metrics:
            # No user defined metric should be named the same as a standard one.
            # Besides, their keys must be string and their values numeric
            for ud_metric in metrics.key():
                if ud_metric in self.metrics_:
                    raise ValueError(f"A user defined metric must not be named the same as " \
                                     f"a standard metric '{ud_metric}'.")
                if not isinstance(ud_metric, str):
                    raise ValueError(f"The name of a user defined metric must be string, but " \
                                     f"{ud_metric} is provided.")
                if not type(metrics[ud_metric]) in [int, float]:
                    raise ValueError(f"The value of a user defined metric must be numeric, but " \
                                     f"the value of{ud_metric} is {metrics[ud_metric]}.")

        # finding model_id
        model_id = model._get_vertica_model_id()
        if model_id == 0:
            raise ValueError(f"model {model.model_name} does not exist in the database.")

        # collecting model parameters
        model_parameters = model.get_params()

        # measuring metrics
        measured_metrics = []
        if self.experiment_type != "clustering":
            model.test_relation = self.test_relation
            if self.experiment_type == "regressor" or self.experiment_type == "binary":
                report = model.report()
                measured_metrics = report.values["value"]
            else: # self.experiment_type == "multi"
                metric_fucntion = ["micro", "macro", "weighted"]
                # TODO: calculating y_score and store it in a temp table will help performance
                for avg_method in average_methods:
                    m_metric = roc_auc_score(y_true = self.y,
                                             y_score = model.deploySQL(),
                                             input_relation = self.test_relation,
                                             average = avg_method)
                    measured_metrics.append(m_metric)

                    m_metric = prc_auc_score(y_true = self.y,
                                             y_score = model.deploySQL(),
                                             input_relation = self.test_relation,
                                             average = avg_method)
                    measured_metrics.append(m_metric)

                    m_metric = accuracy_score(y_true = self.y,
                                              y_score = model.deploySQL(),
                                              input_relation = self.test_relation,
                                              average = avg_method)
                    measured_metrics.append(m_metric)

                    m_metric = precision_score(y_true = self.y,
                                               y_score = model.deploySQL(),
                                               input_relation = self.test_relation,
                                               average = avg_method)
                    measured_metrics.append(m_metric)

                    m_metric = recall_score(y_true = self.y,
                                            y_score = model.deploySQL(),
                                            input_relation = self.test_relation,
                                            average = avg_method)
                    measured_metrics.append(m_metric)

                    m_metric = f1_score(y_true = self.y,
                                        y_score = model.deploySQL(),
                                        input_relation = self.test_relation,
                                        average = avg_method)
                    measured_metrics.append(m_metric)

        # the model will not be added if any of the above steps fail
        self.model_name_list_.append(model.model_name)
        self.model_id_list_.append(model_id)
        self.model_type_list_.append(model._model_type)
        self.parameters_.append(model_parameters)
        self.measured_metrics_.append(measured_metrics)
        self.user_defined_metrics_.append(metrics)

        self._save_experiment(model_id, model_parameters, measured_metrics, metrics)

    def list_models(self) -> TableSample:
        values_table = {"model_name": self.model_name_list_,
                        "model_type": self.model_type_list_,
                        "model_parameters": self.parameters_}

        for index, metric in enumerate(self.metrics_):
            metric_values = []
            for values_list in self.measured_metrics_:
                metric_values.append(values_list[index])
            values_table[metric] = metric_values

        values_table["user_defined_metrics"] = self.user_defined_metrics_

        return TableSample(values_table)

    def load_best_model(self, metric: str) -> VerticaModel:
        if len(self.model_name_list_) == 0:
            return None

        max_value = float('-inf')
        max_index = -1

        if metric in self.metrics_:
            # search the list of the standard metrics for the requested metric
            metric_index = self.metrics_.index(metric)

            for index, item in enumerate(sef.measured_metrics_):
                if item[metric_index] > max_value:
                    max_value = item[metric_index]
                    max_index = index
        else:
            # search the list of user defined metrics
            for index, item in enumerate(self.user_defined_metrics_):
                if metric in item.keys():
                    if item[metric] > max_value:
                        max_value = item[metric]
                        max_index = index

        if max_index == -1:
            raise ValueError(f"Cannot find a metric named {metric}")

        best_model = load_model(name = self.model_name_list_[max_index])
        return best_model

    def plot(self,
             parameter: str,
             metric: str,
             show: bool = True,
             chart: Optional[PlottingObject] = None,
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
        show: bool, optional
            If set to True, the Plotting object will be returned.
        chart: PlottingObject, optional
            The chart object to plot on.
        **style_kwargs
            Any  optional  parameter  to pass to the  plotting functions.

        Returns
        -------
        obj
            Plotting Object.
        """
        parameter_list = []
        metric_list = []

        if metric in self.metrics_:
            # it is a standard metric
            metric_index = self.metrics_.index(metric)

            for model_index, item in enumerate(sef.measured_metrics_):
                if parameter in self.parameters_[model_index].key():
                    parameter_list.append(self.parameters_[model_index][parameter])
                    metric_list.append(item[metric_index])
        else:
            # it is a user defined metric
            for model_index, item in enumerate(self.user_defined_metrics_):
                if (metric in item.keys()) and (parameter in self.parameters_[model_index].key()):
                    parameter_list.append(self.parameters_[model_index][parameter])
                    metric_list.append(item[metric])

        if len(parameter_list) == 0 or len(metric_list) == 0:
            raise ValueError("Could not find any datapoint for the provided pair of " \
                             "(parameter, metric) = ({parameter}, {metric}).")

        vml = get_vertica_mllib()

        vpy_plt, kwargs = self.get_plotting_lib(
            class_name="ScatterPlot",
            chart=chart,
            style_kwargs=style_kwargs,
        )
        # TODO: I'm not sure how to work with vpy_plt
        """
        return vpy_plt.ScatterPlot(
            vdf=self,
            columns=columns,
            by=by,
            cmap_col=cmap_col,
            size=size,
            max_cardinality=max_cardinality,
            cat_priority=cat_priority,
            max_nb_points=max_nb_points,
        ).draw(**kwargs)
        """
    def drop(self, keeping_models: list = None) -> bool:
        """
        Drops all models of the experiment except those in the keeping_models list.
        It also clears the info of models saved in the attributes, and drops the
        experiment_table if it is specified.
        """
        for model in self.model_name_list_:
            if not model in keeping_models:
                drop(name = model, method = "model")

        self.model_name_list_.clear()
        self.model_id_list_.clear()
        self.parameters_.clear()
        self.measured_metrics_.clear()
        self.user_defined_metrics_.clear()

        drop(name = self.experiment_table, method = "table") 

    def __str__(self) -> str:
        """
        Returns the model Representation.
        """
        return self.__repr__()

    def __repr__(self) -> str:
        """
        Returns the model Representation.
        """
        return f"experiment_name: {self.experiment_name}, experiment_type: {self.experiment_type}"

    def _load_or_create_experiment_table(self):
        """
        Loads previous experiment info from the experiment table if it exists;
        creates the table otherwise.
        """
        schema, relation = schema_relation(self.experiment_table)
        if does_table_exist(relation, schema):
            # does the relation qualify to be an experiment table
            self._evaluate_experiment_table(table_name=relation, schema=schema)

            # load previously saved experiment
            self._load_experiment_table(input_relation=relation, schema=schema)
        else:
            # create the experiment table
            create_table(table_name=relation, schema=schema, dtype=_experiment_columns,
                         temporary_local_table=False, raise_error=True)

    def _evaluate_experiment_table(self, table_name: str, schema: str) -> None:
        """
        Evaluates if schema.table_name meets the criteria of being an experiment table.
        """
        # does user has required privileges?
        has_privileges(table_name, schema, ["SELECT", "INSERT", "DELETE"], raise_error=True)

        # does the table have the expected columns?
        vdf = vDataFrame(input_relation=table_name, schema=schema)
        table_columns = vdf.get_columns()
        for index, item in enumerate(table_columns):
            table_columns[index] = item.lower()

        if sorted(_experiment_columns) != sorted(table_columns) :
            raise ValueError(f"Table {schema}.{table_name} does not have " \
                             "the required set of columns to be an experiment table.")

    def _load_experiment_table(self, table_name: str, schema: str) -> None:
        """
        Loads schema.table_name as an experiment table from the database.
        """
        if self.experiment_type == "auto":
            # trying to find the type from the experiment table
            query_type = f"SELECT experiment_type FROM {schema}.{table_name} "
                         f"WHERE experiment_name='{self.experiment_name}' LIMIT 1;"
            exp_type = _executeSQL(query_type, title="finding experiment type", method="fetchrow")
            if not exp_type:
                # there is no history
                return
            self.experiment_type = exp_type[0]

        query_load = f"SELECT model_id, parameters, measured_metrics, user_defined_metrics "
                     f"FROM {schema}.{table_name} WHERE experiment_name='{self.experiment_name}' "
                     f"AND experiment_type='{self.experiment_type}';"
        ts = TableSample.read_sql(query_load, title="loading experiment table")
        if (len(tc.values["model_id"]) == 0):
            # there is no history
            return

        for index, model_id in enumerate(ts.values["model_id"]):
            query_model = f"SELECT schema_name, model_name FROM models WHERE model_id={model_id};"
            res = _executeSQL(query_model, title="finding model", method="fetchrow")
            if res:
                try:
                    parameters = eval(ts.values["parameters"][index])
                    measured_metrics = eval(ts.values["measured_metrics"][index])
                    ud_metrics = eval(ts.values["user_defined_metrics"][index])
                    # the model will be ignored if any of the above eval operations fails
                    self.model_name_list_.append(res[0] + "." + res[1])
                    self.model_id_list_.append(model_id)
                    self.parameters_.append(parameters)
                    self.measured_metrics_.append(measured_metrics)
                    self.user_defined_metrics_.append(ud_metrics)
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
                raise ValueError("In an experiment of type regressor,"
                                 " the y column must be numerical")
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
        if sorted(self.test_relation[y].distinct()) != [0, 1]:
            if raise_error:
                raise ValueError("In an experiment of type binary,"
                                 " the y column must have only two distinct values")
            else:
                return False

        return True

    def _save_experiment(self, model_id: int, model_parameters: dict,
                         measured_metrics: list, user_defined_metrics: dict) -> None:
        if not self.experiment_table:
            return

        user_id = _executeSQL("SELECT user_id FROM users WHERE user_name=current_user();",
                              title = "finding user_id", method="fetchfirstelem")

        inser_query = f"INSERT INTO {self.experiment_table} VALUES ('{self.experiment_name}',"
                      f"'{self.experiment_type}', {model_id}, {user_id},"
                      f"'{model_parameters.__str__()}', '{measured_metrics.__str__()}',"
                      f"'{user_defined_metrics.__str__()}');"
        _executeSQL(inser_query, title = "inserting experiment info")
        _executeSQL("COMMIT;", title = "commiting the insert of experiment info")
