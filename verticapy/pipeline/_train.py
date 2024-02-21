#!/usr/bin/env python3
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
"""
This script runs the Vertica Machine Learning Pipeline Training.
"""
from typing import Tuple

from verticapy._typing import SQLColumns

from verticapy.core.vdataframe.base import vDataFrame

from verticapy.errors import QueryError

from verticapy.machine_learning.vertica.base import VerticaModel
from verticapy.machine_learning.vertica.cluster import (
    BisectingKMeans,
    DBSCAN,
    KMeans,
    KPrototypes,
    NearestCentroid,
)
from verticapy.machine_learning.vertica.decomposition import MCA, PCA, SVD
from verticapy.machine_learning.vertica.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
    XGBClassifier,
    XGBRegressor,
)
from verticapy.machine_learning.vertica.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    PoissonRegressor,
    Ridge,
)
from verticapy.machine_learning.vertica.naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    GaussianNB,
    MultinomialNB,
    NaiveBayes,
)
from verticapy.machine_learning.vertica.neighbors import (
    KNeighborsClassifier,
    KernelDensity,
    KNeighborsRegressor,
    LocalOutlierFactor,
)
from verticapy.machine_learning.vertica.svm import LinearSVC, LinearSVR
from verticapy.machine_learning.vertica.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    DummyTreeClassifier,
    DummyTreeRegressor,
)
from verticapy.machine_learning.vertica.tsa import ARIMA, ARMA, AR, MA

from verticapy.pipeline._helper import execute_and_return

SUPPORTED_FUNCTIONS = [
    BisectingKMeans,
    DBSCAN,
    KMeans,
    KPrototypes,
    NearestCentroid,
    MCA,
    PCA,
    SVD,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
    XGBClassifier,
    XGBRegressor,
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    PoissonRegressor,
    Ridge,
    BernoulliNB,
    CategoricalNB,
    GaussianNB,
    MultinomialNB,
    NaiveBayes,
    KNeighborsClassifier,
    KernelDensity,
    KNeighborsRegressor,
    LocalOutlierFactor,
    LinearSVC,
    LinearSVR,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    DummyTreeClassifier,
    DummyTreeRegressor,
    ARIMA,
    ARMA,
    AR,
    MA,
]


def training(
    train: dict, vdf: vDataFrame, pipeline_name: str, cols: SQLColumns
) -> Tuple[str, VerticaModel, str]:
    """
    Run the training step
    of the pipeline.

    Parameters
    ----------
    train: dict
        YAML object which outlines
        the steps of the operation.
    vdf: vDataFrame
        The model trained in the
        training step.
    pipeline_name: str
        The prefix name of the
        intended pipeline to unify
        the creation of the objects.
    cols: SQLColumns
        ``list`` of the columns needed
        to deploy the model.

    Returns
    -------
    return ``meta_sql, model, model_sql``

    str
        The SQL to replicate the
        steps of the yaml file.
    VerticaModel
        The model created after
        training.
    str
        The SQL needed to retrain
        the model.

    Example
    -------
    Here you can use an existing relation.

    .. code-block:: python

        from verticapy.datasets import load_titanic
        load_titanic() # Loading the titanic dataset in Vertica

        import verticapy as vp
        vp.vDataFrame("public.titanic")

    If you want to make some transformations checkout
    :py:function:`~pipeline._transform.transformation`.

    Then you can train a model.
    This example demonstrates how to use the `training`
    function to run the training step of a pipeline.

    .. code-block:: python

        from verticapy.pipeline._train import training

        # Define the training steps in a YAML object
        train = {
            'method':
                'name': 'RandomForestClassifier',
                'target': 'survival',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                }
        }

        # Define the vDataFrame containing the training data
        vdf = vDataFrame("public.titanic")

        # Define the pipeline name
        pipeline_name = "my_pipeline"

        # Define the columns needed to deploy the model
        cols = ['family_size', 'fares', 'sexes', 'ages']

        # Call the training function
        meta_sql, model, model_sql = training(train, vdf, pipeline_name, cols)
    """
    meta_sql = ""
    if "train_test_split" in train:
        info = train["train_test_split"]
        test_size = 0.33 if "test_size" not in info else info["test_size"]
        train_set, test_set = vdf.train_test_split(test_size)
        meta_sql += execute_and_return(
            f"CREATE OR REPLACE VIEW {pipeline_name + '_TRAIN_VIEW'} AS SELECT * FROM "
            + train_set.current_relation()
            + ";"
        )
        meta_sql += execute_and_return(
            f"CREATE OR REPLACE VIEW {pipeline_name + '_TEST_VIEW'} AS SELECT * FROM "
            + test_set.current_relation()
            + ";"
        )

    else:
        meta_sql += execute_and_return(
            f"CREATE OR REPLACE VIEW {pipeline_name + '_TRAIN_VIEW'} AS SELECT * FROM "
            + vdf.current_relation()
            + ";"
        )
        meta_sql += execute_and_return(
            f"CREATE OR REPLACE VIEW {pipeline_name + '_TEST_VIEW'} AS SELECT * FROM "
            + vdf.current_relation()
            + ";"
        )

    methods = list(train.keys())
    methods = [method for method in methods if "method" in method]
    for method in methods:
        tm = train[method]
        name = tm["name"]
        target = tm["target"] if "target" in tm else ""
        params = tm["params"] if "params" in tm else []
        param_str = ""
        for param in params:
            temp = params[param]
            if isinstance(temp, str):
                param_str += f"{param} = '{params[param]}', "
            else:
                param_str += f"{param} = {params[param]}, "

        param_str = param_str[:-2]
        model = eval(
            f"{name}('{pipeline_name + '_MODEL'}', {param_str})",
            globals(),
        )
        predictors = cols  # ['col1', 'col2', 'col3', 'col4']
        if "include" in tm:
            predictors = tm["include"]

        if "exclude" in tm:
            predictors = list(set(predictors) - set(tm["exclude"]))

        if target == "":
            # UNSUPERVISED
            model.fit(pipeline_name + "_TRAIN_VIEW", predictors)
        else:
            # SUPERVISED
            model.fit(pipeline_name + "_TRAIN_VIEW", predictors, target)
        meta_sql += "\n"

        ts = model.get_vertica_attributes()  # TableSample
        if "call_string" in ts["attr_name"]:
            model_sql = model.get_vertica_attributes("call_string")["call_string"][0]
            if model_sql.split(" ")[0] != "SELECT":
                model_sql = "SELECT " + model_sql + ";"
        else:
            model_sql = (
                "SELECT "
                + model.get_vertica_attributes("metrics")["metrics"][0]
                .split("Call:")[1]
                .replace("\n", " ")
                + ";"
            )
        meta_sql += model_sql + "\n"
    return meta_sql, model, model_sql
