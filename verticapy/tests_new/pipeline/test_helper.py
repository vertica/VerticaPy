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

from verticapy._utils._sql._sys import _executeSQL

from verticapy.pipeline import _helper

from verticapy.tests_new.pipeline.conftest import (
    build_pipeline,
    pipeline_exists,
    pipeline_not_exists,
)


@pytest.mark.parametrize(
    "yaml,keywords",
    [
        ({"key1": 1, "key2": 2, "key3": 3}, ["key1", "key2", "key3"]),
        ({}, []),
        ({"key1": 1}, []),
    ],
)
def test_required_keywords(yaml, keywords):
    """
    test function required_keywords.
    """
    assert _helper.required_keywords(yaml, keywords)


@pytest.mark.parametrize(
    "yaml,keywords,error_keyword",
    [
        ({"key1": 1, "key3": 3}, ["key1", "key2", "key3"], "key2"),
        ({}, ["key1", "key2", "key3"], "key1"),
        ({"key1": 1}, ["key1", "key2", "key3"], "key2"),
        ({"key1": 1, "key2": 2}, ["key1", "key2", "key3"], "key3"),
    ],
)
def test_required_keywords_negative(yaml, keywords, error_keyword):
    """
    test function required_keywords.
    """
    with pytest.raises(KeyError) as error_info:
        _helper.required_keywords(yaml, keywords)
    assert error_keyword in str(error_info.value)


@pytest.mark.parametrize(
    "delimiter",
    [
        ",",
        " ",
        "a",
    ],
)
def test_is_valid_delimiter(delimiter):
    """
    test function is_valid_delimiter
    """
    assert _helper.is_valid_delimiter(delimiter)


@pytest.mark.parametrize(
    "delimiter",
    [
        "ú",
        "ð",
    ],
)
def test_is_valid_delimiter_negative(delimiter):
    """
    test function is_valid_delimiter
    """
    assert not _helper.is_valid_delimiter(delimiter)


def test_execute_and_return():
    """
    test function execute_and_return.
    """
    sql = "SELECT 1;"
    assert _helper.execute_and_return(sql) == "SELECT 1;\n"


@pytest.mark.parametrize(
    "input_string,intended_string",
    [
        ("SELECT 1;", "SELECT 1;"),
        ("SELECT 1; /* THIS IS A COMMENT */ ", "SELECT 1;  "),
        (
            "SELECT 1; /* THIS IS A COMMENT */ /* THIS IS ALSO A COMMENT */",
            "SELECT 1;  ",
        ),
    ],
)
def test_remove_comments(input_string, intended_string):
    """
    test function remove_comments
    """
    assert _helper.remove_comments(input_string) == intended_string


def test_to_sql():
    """
    test function to_sql
    """
    dummy_sql = (
        """SELECT model('model_name', 'table_name', 'target_column', '"col1, col2"')"""
    )
    dummy_string = """SELECT model(' || QUOTE_LITERAL('model_name') || ', ' || QUOTE_LITERAL('table_name') || ', ' || QUOTE_LITERAL('target_column') || ', ' || QUOTE_LITERAL('"col1, col2"') || '';"""
    assert _helper.to_sql(dummy_sql) == dummy_string


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

    assert pipeline_exists("test_pipeline", check_metric=True)
    assert pipeline_exists("test_pipeline2", check_metric=True)
    assert pipeline_exists("test_pipeline_2", check_metric=True)

    # Drop 'test_pipeline'
    _executeSQL("CALL drop_pipeline('public', 'test_pipeline')")

    assert pipeline_not_exists("test_pipeline", check_metric=True)
    assert pipeline_exists("test_pipeline2", check_metric=True)
    assert pipeline_exists("test_pipeline_2", check_metric=True)

    _executeSQL("CALL drop_pipeline('public', 'test_pipeline2')")
    _executeSQL("CALL drop_pipeline('public', 'test_pipeline_2')")

    assert pipeline_not_exists("test_pipeline2", check_metric=True)
    assert pipeline_not_exists("test_pipeline_2", check_metric=True)
