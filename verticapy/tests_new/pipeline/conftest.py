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

import verticapy.sql.sys as sys

from verticapy.pipeline import parser
from verticapy.pipeline._helper import setup


@pytest.fixture(scope="session", autouse=True)
def pipeline_setup():
    setup()


def pipeline_exists(pipeline_name: str, check_metric=False, model=None):
    """
    helper function to test if a pipeline
    is properly created.
    """
    assert sys.does_view_exist(f"{pipeline_name}_TRAIN_VIEW", "public")
    assert sys.does_view_exist(f"{pipeline_name}_TEST_VIEW", "public")
    assert not check_metric or sys.does_table_exist(
        f"{pipeline_name}_METRIC_TABLE", "public"
    )
    assert model == None or model.does_model_exists(f"public.{pipeline_name}_MODEL")
    return True


def pipeline_not_exists(pipeline_name: str, check_metric=False, model=None):
    """
    helper function to test if a pipeline
    no longer exists.
    """
    assert not sys.does_view_exist(f"{pipeline_name}_TRAIN_VIEW", "public")
    assert not sys.does_view_exist(f"{pipeline_name}_TEST_VIEW", "public")
    assert not check_metric or not sys.does_table_exist(
        f"{pipeline_name}_METRIC_TABLE", "public"
    )
    assert model == None or not model.does_model_exists(f"public.{pipeline_name}_MODEL")
    return True


def build_pipeline(pipeline_name: str):
    """
    helper function to build identical pipelines
    with different names.
    """
    steps = {
        "schema": "public",
        "pipeline": pipeline_name,
        "table": "public.winequality",
        "steps": {
            "transform": {
                "col1": {"sql": "fixed_acidity"},
                "col2": {
                    "sql": "volatile_acidity",
                },
                "col3": {
                    "sql": "citric_acid",
                },
            },
            "train": {
                "train_test_split": {"test_size": 0.34},
                "method": {
                    "name": "LinearRegression",
                    "target": "quality",
                },
                "schedule": "* * * * *",
            },
            "test": {
                "metric1": {"name": "r2", "y_true": "quality", "y_score": "prediction"}
            },
        },
    }
    parser.parse_yaml(steps)
