"""
Copyright  (c)  2018-2025 Open Text  or  one  of its
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

# Pytest
import pytest

# VerticaPy
from verticapy._utils._sql._format import schema_relation
from verticapy._utils._sql._format import quote_ident
from verticapy.errors import ParsingError
import verticapy._config.config as conf

@pytest.fixture(scope="module", autouse=True)
def temp_schema_fixture():
    # Save the current value
    original_temp_schema = conf.get_option("temp_schema")
    
    # Set the temp schema for tests
    conf.set_option("temp_schema", "temp_schema")
    
    # Run the tests
    yield
    
    # Restore the original value after tests in this file
    conf.set_option("temp_schema", original_temp_schema)


@pytest.mark.parametrize("input_relation, expected_schema, expected_table", [
    # No schema: should use temp schema
    ("my_table", "temp_schema", "my_table"),
    # Basic schema.table
    ("my_schema.my_table", "my_schema", "my_table"),
    # Namespace.schema.table
    ("namespace.schema.table", "namespace.schema", "table"),
    # Quoted relation
    ('"my_schema"."my_table"', "my_schema", "my_table"),
    # Quoted table only
    ('"my_table"', "temp_schema", "my_table"),
])
def test_schema_relation_basic(input_relation, expected_schema, expected_table):
    result_schema, result_table = schema_relation(input_relation, do_quote=False)
    assert result_schema == expected_schema
    assert result_table == expected_table

@pytest.mark.parametrize("input_relation, expected_schema, expected_table", [
    ("my_schema.my_table", quote_ident("my_schema"), quote_ident("my_table")),
    ("namespace.schema.table", quote_ident("namespace.schema"), quote_ident("table")),
])
def test_schema_relation_with_quotes(input_relation, expected_schema, expected_table):
    result_schema, result_table = schema_relation(input_relation, do_quote=True)
    assert result_schema == expected_schema
    assert result_table == expected_table

@pytest.mark.parametrize("invalid_input", [
    "too.many.dots.in.this",   # more than 3 parts
    "missing.dot",             # should be fine, temp_schema used
    '"weirdquote.table',       # unbalanced quote
])
def test_schema_relation_invalid_inputs(invalid_input):
    if invalid_input.count('.') >= 3 or '"' in invalid_input and invalid_input.count('"') % 2 != 0:
        with pytest.raises(ParsingError):
            schema_relation(invalid_input)

def test_schema_relation_non_string_input():
    # Input is not a string, should return temp_schema + empty string
    schema, table = schema_relation(12345, do_quote=False)
    assert schema == "temp_schema"
    assert table == ""