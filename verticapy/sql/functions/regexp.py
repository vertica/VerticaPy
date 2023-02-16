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

#
#
# Modules
#
# VerticaPy Modules
from verticapy.core.str_sql import str_sql
from verticapy.sql._utils._format import format_magic


def regexp_count(
    expr, pattern, position: int = 1,
):
    """
Returns the number times a regular expression matches a string.

Parameters
----------
expr: object
    Expression.
pattern: object
    The regular expression to search for within string.
position: int, optional
    The number of characters from the start of the string where the function 
    should start searching for matches.

Returns
-------
str_sql
    SQL expression.
    """
    expr = format_magic(expr)
    pattern = format_magic(pattern)
    return str_sql(f"REGEXP_COUNT({expr}, {pattern}, {position})", "int")


def regexp_ilike(expr, pattern):
    """
Returns true if the string contains a match for the regular expression.

Parameters
----------
expr: object
    Expression.
pattern: object
    A string containing the regular expression to match against the string.

Returns
-------
str_sql
    SQL expression.
    """
    expr = format_magic(expr)
    pattern = format_magic(pattern)
    return str_sql(f"REGEXP_ILIKE({expr}, {pattern})")


def regexp_instr(
    expr, pattern, position: int = 1, occurrence: int = 1, return_position: int = 0
):
    """
Returns the starting or ending position in a string where a regular 
expression matches.

Parameters
----------
expr: object
    Expression.
pattern: object
    The regular expression to search for within the string.
position: int, optional
    The number of characters from the start of the string where the function 
    should start searching for matches.
occurrence: int, optional
    Controls which occurrence of a pattern match in the string to return.
return_position: int, optional
    Sets the position within the string to return.

Returns
-------
str_sql
    SQL expression.
    """
    expr = format_magic(expr)
    pattern = format_magic(pattern)
    return str_sql(
        f"REGEXP_INSTR({expr}, {pattern}, {position}, {occurrence}, {return_position})"
    )


def regexp_like(expr, pattern):
    """
Returns true if the string matches the regular expression.

Parameters
----------
expr: object
    Expression.
pattern: object
    A string containing the regular expression to match against the string.

Returns
-------
str_sql
    SQL expression.
    """
    expr = format_magic(expr)
    pattern = format_magic(pattern)
    return str_sql(f"REGEXP_LIKE({expr}, {pattern})")


def regexp_replace(expr, target, replacement, position: int = 1, occurrence: int = 1):
    """
Replace all occurrences of a substring that match a regular expression 
with another substring.

Parameters
----------
expr: object
    Expression.
target: object
    The regular expression to search for within the string.
replacement: object
    The string to replace matched substrings.
position: int, optional
    The number of characters from the start of the string where the function 
    should start searching for matches.
occurrence: int, optional
    Controls which occurrence of a pattern match in the string to return.

Returns
-------
str_sql
    SQL expression.
    """
    expr = format_magic(expr)
    target = format_magic(target)
    replacement = format_magic(replacement)
    return str_sql(
        f"REGEXP_REPLACE({expr}, {target}, {replacement}, {position}, {occurrence})"
    )


def regexp_substr(expr, pattern, position: int = 1, occurrence: int = 1):
    """
Returns the substring that matches a regular expression within a string.

Parameters
----------
expr: object
    Expression.
pattern: object
    The regular expression to find a substring to extract.
position: int, optional
    The number of characters from the start of the string where the function 
    should start searching for matches.
occurrence: int, optional
    Controls which occurrence of a pattern match in the string to return.

Returns
-------
str_sql
    SQL expression.
    """
    expr = format_magic(expr)
    pattern = format_magic(pattern)
    return str_sql(f"REGEXP_SUBSTR({expr}, {pattern}, {position}, {occurrence})")
