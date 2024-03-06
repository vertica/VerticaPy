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
import pytest

from verticapy import drop
from verticapy._utils._sql._sys import _executeSQL
from verticapy.datasets import load_winequality

from verticapy.pipeline._validate import testing
from verticapy.pipeline._train import training

from verticapy.tests_new.pipeline.conftest import pipeline_exists


class TestValidate:
    """
    Analytic Functions test
    -
    """

    @pytest.mark.parametrize(
        "name",
        [
            "aic",
            "bic",
            "max_error",
            "mean_absolute_error",
            "mean_squared_error",
            "mean_squared_log_error",
            "r2",
        ],
    )
    def test_regression(
        self,
        name,
    ):
        pipeline_name = "test_pipeline"
        _executeSQL(f"CALL drop_pipeline('public', '{pipeline_name}');")

        # Model Setup
        table = load_winequality()
        kwargs = {
            "method": {
                "name": "LinearRegression",
                "target": "quality",
                "params": {
                    "tol": 1e-6,
                    "max_iter": 100,
                    "solver": "newton",
                    "fit_intercept": True,
                },
            }
        }

        cols = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar"]

        metric = {
            "metric": {
                "name": name,
                "y_true": "quality",
                "y_score": "prediction",
            }
        }

        # Part 1: Train a Model
        _, model, _ = training(kwargs, table, pipeline_name, cols)

        # Part 2: Run the Metrics
        _, metric_sql = testing(metric, model, pipeline_name, cols)

        assert model
        assert pipeline_exists(pipeline_name, check_metric=True, model=model)

        # Check the main functions of the metric_table sql script are included
        assert "DROP TABLE" in metric_sql
        assert "CREATE TABLE" in metric_sql
        assert f"{pipeline_name}_PREDICT_VIEW" in metric_sql
        assert "COMMIT;" in metric_sql

        # Check the created metric_table with one created with metric_sql
        # (these should be identical)
        res = _executeSQL(
            f"SELECT * FROM {pipeline_name}_METRIC_TABLE;", method="fetchrow"
        )

        _executeSQL(metric_sql)
        res_from_sql = _executeSQL(
            f"SELECT * FROM {pipeline_name}_METRIC_TABLE;", method="fetchrow"
        )

        assert res == res_from_sql
        assert name in res

        # drop pipeline
        _executeSQL(f"CALL drop_pipeline('public', '{pipeline_name}');")
        drop("public.winequality")
