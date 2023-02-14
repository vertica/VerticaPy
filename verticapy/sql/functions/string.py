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
from verticapy._version import check_minimum_version
from verticapy.core.str_sql import str_sql
from verticapy.sql._utils._format import format_magic

# Regular str functions


def length(expr):
    """
Returns the length of a string.

Parameters
----------
expr: object
    Expression.

Returns
-------
str_sql
    SQL expression.
    """
    expr = format_magic(expr)
    return str_sql(f"LENGTH({expr})", "int")


def lower(expr):
    """
Returns a VARCHAR value containing the argument converted to 
lowercase letters. 

Parameters
----------
expr: object
    Expression.

Returns
-------
str_sql
    SQL expression.
    """
    expr = format_magic(expr)
    return str_sql(f"LOWER({expr})", "text")


def substr(expr, position: int, extent: int = None):
    """
Returns VARCHAR or VARBINARY value representing a substring of a specified 
string.

Parameters
----------
expr: object
    Expression.
position: int
    Starting position of the substring.
extent: int, optional
    Length of the substring to extract.

Returns
-------
str_sql
    SQL expression.
    """
    expr = format_magic(expr)
    if extent:
        position = f"{position}, {extent}"
    return str_sql(f"SUBSTR({expr}, {position})", "text")


def upper(expr):
    """
Returns a VARCHAR value containing the argument converted to uppercase 
letters. 

Parameters
----------
expr: object
    Expression.

Returns
-------
str_sql
    SQL expression.
    """
    expr = format_magic(expr)
    return str_sql(f"UPPER({expr})", "text")


# Edit Distance & Soundex


@check_minimum_version
def edit_distance(
    expr1, expr2,
):
    """
Calculates and returns the Levenshtein distance between the two strings.

Parameters
----------
expr1: object
    Expression.
expr2: object
    Expression.

Returns
-------
str_sql
    SQL expression.
    """
    expr1 = format_magic(expr1)
    expr2 = format_magic(expr2)
    return str_sql(f"EDIT_DISTANCE({expr1}, {expr2})", "int")


levenshtein = edit_distance


@check_minimum_version
def soundex(expr):
    """
Returns Soundex encoding of a varchar strings as a four -character string.

Parameters
----------
expr: object
    Expression.

Returns
-------
str_sql
    SQL expression.
    """
    expr = format_magic(expr)
    return str_sql(f"SOUNDEX({expr})", "varchar")


@check_minimum_version
def soundex_matches(
    expr1, expr2,
):
    """
Generates and compares Soundex encodings of two strings, and returns a count 
of the matching characters (ranging from 0 for no match to 4 for an exact 
match).

Parameters
----------
expr1: object
    Expression.
expr2: object
    Expression.

Returns
-------
str_sql
    SQL expression.
    """
    expr1 = format_magic(expr1)
    expr2 = format_magic(expr2)
    return str_sql(f"SOUNDEX_MATCHES({expr1}, {expr2})", "int")


# Jaro & Jaro Winkler


@check_minimum_version
def jaro_distance(
    expr1, expr2,
):
    """
Calculates and returns the Jaro distance between two strings.

Parameters
----------
expr1: object
    Expression.
expr2: object
    Expression.

Returns
-------
str_sql
    SQL expression.
    """
    expr1 = format_magic(expr1)
    expr2 = format_magic(expr2)
    return str_sql(f"JARO_DISTANCE({expr1}, {expr2})", "float")


@check_minimum_version
def jaro_winkler_distance(
    expr1, expr2,
):
    """
Calculates and returns the Jaro-Winkler distance between two strings.

Parameters
----------
expr1: object
    Expression.
expr2: object
    Expression.

Returns
-------
str_sql
    SQL expression.
    """
    expr1 = format_magic(expr1)
    expr2 = format_magic(expr2)
    return str_sql(f"JARO_WINKLER_DISTANCE({expr1}, {expr2})", "float")
