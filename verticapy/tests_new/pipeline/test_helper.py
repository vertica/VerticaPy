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
from collections import defaultdict

from verticapy._utils._sql._sys import _executeSQL

from verticapy.pipeline import parser, _helper

import verticapy.sql.sys as sys


def test_required_keywords():
    """
    test function required_keywords.
    """
    yaml = {"key1": 1, "key2": 2, "key3": 3}
    keywords = ["key1", "key2", "key3"]
    assert _helper.required_keywords(yaml, keywords) == True

    yaml = {"key1": 1, "key3": 3}
    keywords = ["key1", "key2", "key3"]
    with pytest.raises(KeyError) as error_info:
        _helper.required_keywords(yaml, keywords)
    assert "key2" in str(error_info.value)

    yaml = {}
    keywords = []
    assert _helper.required_keywords(yaml, keywords) == True

    yaml = {"key1": 1}
    keywords = []
    assert _helper.required_keywords(yaml, keywords) == True


def test_execute_and_return():
    """
    test function execute_and_return.
    """
    sql = "SELECT 1;"
    assert _helper.execute_and_return(sql) == "SELECT 1;\n"


def test_remove_comments():
    """
    test function remove_comments
    """
    input_string = "SELECT 1;"
    assert _helper.remove_comments(input_string) == input_string

    input_string = "SELECT 1; /* THIS IS A COMMENT */ "
    assert _helper.remove_comments(input_string) == "SELECT 1;  "

    input_string = "SELECT 1; /* THIS IS A COMMENT */ /* THIS IS ALSO A COMMENT */"
    assert _helper.remove_comments(input_string) == "SELECT 1;  "


def test_to_sql():
    """
    test function to_sql
    """
    dummy_sql = (
        """SELECT model('model_name', 'table_name', 'target_column', '"col1, col2"')"""
    )
    dummy_string = """SELECT model(' || QUOTE_LITERAL('model_name') || ', ' || QUOTE_LITERAL('table_name') || ', ' || QUOTE_LITERAL('target_column') || ', ' || QUOTE_LITERAL('"col1, col2"') || '';"""
    assert _helper.to_sql(dummy_sql) == dummy_string


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


def test_setup():
    """
    test function setup.
    """
    # Does function exist?
    assert _executeSQL(
        """SELECT EXISTS (
    SELECT 1
    FROM v_catalog.user_procedures
    WHERE procedure_name = 'drop_pipeline');
    """
    )
    assert _executeSQL("CALL drop_pipeline('public', 'pipeline');")

    # Does it do the expected thing?
    build_pipeline("test_pipeline")
    build_pipeline("test_pipeline2")
    build_pipeline("test_pipeline_2")

    assert sys.does_view_exist("test_pipeline_TRAIN_VIEW", "public")
    assert sys.does_view_exist("test_pipeline_TEST_VIEW", "public")
    assert sys.does_table_exist("test_pipeline_METRIC_TABLE", "public")

    assert sys.does_view_exist("test_pipeline2_TRAIN_VIEW", "public")
    assert sys.does_view_exist("test_pipeline2_TEST_VIEW", "public")
    assert sys.does_table_exist("test_pipeline2_METRIC_TABLE", "public")

    assert sys.does_view_exist("test_pipeline_2_TRAIN_VIEW", "public")
    assert sys.does_view_exist("test_pipeline_2_TEST_VIEW", "public")
    assert sys.does_table_exist("test_pipeline_2_METRIC_TABLE", "public")

    # Drop 'test_pipeline'
    _executeSQL("CALL drop_pipeline('public', 'test_pipeline')")

    assert not sys.does_view_exist("test_pipeline_TRAIN_VIEW", "public")
    assert not sys.does_view_exist("test_pipeline_TEST_VIEW", "public")
    assert not sys.does_table_exist("test_pipeline_METRIC_TABLE", "public")

    assert sys.does_view_exist("test_pipeline2_TRAIN_VIEW", "public")
    assert sys.does_view_exist("test_pipeline2_TEST_VIEW", "public")
    assert sys.does_table_exist("test_pipeline2_METRIC_TABLE", "public")

    assert sys.does_view_exist("test_pipeline_2_TRAIN_VIEW", "public")
    assert sys.does_view_exist("test_pipeline_2_TEST_VIEW", "public")
    assert sys.does_table_exist("test_pipeline_2_METRIC_TABLE", "public")

    _executeSQL("CALL drop_pipeline('public', 'test_pipeline2')")
    _executeSQL("CALL drop_pipeline('public', 'test_pipeline_2')")

    assert not sys.does_view_exist("test_pipeline2_TRAIN_VIEW", "public")
    assert not sys.does_view_exist("test_pipeline2_TEST_VIEW", "public")
    assert not sys.does_table_exist("test_pipeline2_METRIC_TABLE", "public")

    assert not sys.does_view_exist("test_pipeline_2_TRAIN_VIEW", "public")
    assert not sys.does_view_exist("test_pipeline_2_TEST_VIEW", "public")
    assert not sys.does_table_exist("test_pipeline_2_METRIC_TABLE", "public")
