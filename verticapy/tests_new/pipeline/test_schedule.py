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
import itertools

from verticapy import drop
from verticapy._utils._sql._sys import _executeSQL
from verticapy.datasets import load_winequality
from verticapy.machine_learning.vertica.linear_model import LinearRegression

from verticapy.pipeline._validate import testing
from verticapy.pipeline._schedule import scheduler
from verticapy.pipeline._train import training

import verticapy.sql.sys as sys


class TestSchedule:
    def test_schedule_good(self):
        pipeline_name = "test_pipeline"
        _executeSQL("CALL drop_pipeline('public', 'test_pipeline');")

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
        test = {
            "metric": {
                "name": "r2",
                "y_true": "quality",
                "y_score": "prediction",
            }
        }
        schedule = "* * * * *"
        # Part 1: Train a Model
        meta_sql, model, model_sql = training(kwargs, table, "test_pipeline", cols)

        # Part 2: Run the Metrics
        test_sql, table_sql = testing(test, model, pipeline_name, cols)

        # Part 3: Run the Scheduler
        sql = scheduler(schedule, model_sql, table_sql, pipeline_name)
        assert model

        assert sys.does_view_exist("test_pipeline_TRAIN_VIEW", "public")
        assert sys.does_view_exist("test_pipeline_TEST_VIEW", "public")
        assert sys.does_table_exist("test_pipeline_METRIC_TABLE", "public")

        # drop pipeline
        _executeSQL("CALL drop_pipeline('public', 'test_pipeline');")
        drop("public.winequality")
