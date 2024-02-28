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
This script runs the Vertica Machine Learning Pipeline Test.
"""
import pytest

from typing import Tuple

from verticapy import vDataFrame

from verticapy._typing import SQLColumns

from verticapy.machine_learning.metrics.regression import (
    regression_report,
    FUNCTIONS_REGRESSION_SQL_DICTIONARY,
)
from verticapy.machine_learning.vertica.base import VerticaModel

from verticapy.pipeline._helper import execute_and_return, remove_comments


@pytest.mark.skip(reason="[MODULE FOR PIPELINE, NOT A TEST]")
def testing(
    test: dict, model: VerticaModel, pipeline_name: str, cols: SQLColumns
) -> Tuple[str, str]:
    """
    Run the testing step
    of the pipeline.

    Parameters
    ----------
    test: dict
        YAML object which outlines
        the steps of the operation.
    model: VerticaModel
        The model trained in the training
        step.
    pipeline_name: str
        The prefix name of the intended
        pipeline to unify the creation
        of the objects.
    cols: SQLColumns
        ``list`` of the columns used to
        deploy the model.

    Returns
    -------
    str
        The SQL to replicate
        the steps of the yaml
        file.
    str
        The SQL to replicate
        the metric table.

    Example
    -------
    .. code-block:: python

        from verticapy.datasets import load_titanic
        load_titanic() # Loading the titanic dataset in Vertica

        import verticapy as vp
        vp.vDataFrame("public.titanic")

    If you want to make some transformations checkout
    :py:function:`~pipeline._transform.transformation`.
    Then you can train a model, see
    :py:function:`~pipeline._train.train`.

    .. code-block:: python

        from verticapy.pipeline._train import training

        # Define the training steps in a YAML object
        train = {
            'method': {
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

    This example demonstrates how to use
    the `testing` function to run the
    testing step of a pipeline.

    Then you can train a model, see
    :py:function:`~pipeline._train.train`.

    .. code-block:: python

        from verticapy.pipeline._validate import testing

        # Define the training steps in a YAML object
        test = {
            'test': {
                'metric1': {
                    'name': 'accuracy_score',
                    'y_true': 'survival',
                    'y_score': 'prediction',
                },
                'metric2': {
                    'name': 'r2_score',
                    'y_true': 'survival',
                    'y_score': 'prediction',
                },
                'metric3': {
                    'name': 'max_error',
                    'y_true': 'survival',
                    'y_score': 'prediction',
                },
            }
        }

        # Define the vDataFrame containing the training data
        vdf = vDataFrame("public.titanic")

        # Define the pipeline name
        pipeline_name = "my_pipeline"

        # Define the columns needed to deploy the model
        cols = ['family_size', 'fares', 'sexes', 'ages']

        # Call the testing function
        meta_sql, table_sql = testing(test, model, pipeline_name, cols)
    """
    meta_sql = ""
    dummy = model.predict(
        pipeline_name + "_TEST_VIEW", X=cols, name="prediction", inplace=True
    )

    # CREATE A PREDICT VIEW
    meta_sql += execute_and_return(
        f"CREATE OR REPLACE VIEW {pipeline_name + '_PREDICT_VIEW'} AS SELECT * FROM "
        + dummy.current_relation()
        + ";"
    )

    # FOR multiple metrics
    table_sql = ""
    table_sql += f"DROP TABLE IF EXISTS {pipeline_name + '_METRIC_TABLE'};"
    table_sql += execute_and_return(
        f"CREATE TABLE {pipeline_name + '_METRIC_TABLE'}(metric_name Varchar(100), metric FLOAT);"
    )

    for nth_metric in test:
        metric = test[nth_metric]
        name = metric["name"]
        if name not in FUNCTIONS_REGRESSION_SQL_DICTIONARY.keys():
            raise KeyError(f"{name} is not in the set of allowed metrics.")

        y_true = metric["y_true"]
        y_score = metric["y_score"]

        metric_sql = eval(
            f"""regression_report('{y_true}', '{y_score}',
            '{pipeline_name + '_PREDICT_VIEW'}', '{name}', genSQL=True)""",
            globals(),
        )
        pure_metric_sql = remove_comments(metric_sql[:-1])

        col_name = vDataFrame(input_relation=metric_sql).get_columns()[0]
        if col_name is None:
            table_sql += execute_and_return(
                f"INSERT INTO {pipeline_name + '_METRIC_TABLE'} SELECT '{name}', "
                + metric_sql.split("*/")[1]
            )
        else:
            table_sql += execute_and_return(
                f"""INSERT INTO {pipeline_name + '_METRIC_TABLE'} 
                SELECT '{name}', subquery.{col_name} FROM ("""
                + pure_metric_sql
                + ") as subquery;"
            )

    table_sql += execute_and_return(
        "COMMIT;"
    )  # This is one transaction: Otherwise, Vertica discards all changes.
    meta_sql += table_sql
    return meta_sql, table_sql
