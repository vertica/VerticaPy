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
from typing import Optional

from verticapy._typing import SQLExpression
from verticapy._utils._sql._format import format_magic
from verticapy._utils._sql._vertica_version import check_minimum_version

from verticapy.core.string_sql.base import StringSQL


def length(expr: SQLExpression) -> StringSQL:
    """
    Returns the length of a string.

    Parameters
    ----------
    expr: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(f"LENGTH({expr})", "int")


def lower(expr: SQLExpression) -> StringSQL:
    """
    Returns  a VARCHAR value containing  the
    argument converted to lowercase letters.

    Parameters
    ----------
    expr: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(f"LOWER({expr})", "text")


def substr(
    expr: SQLExpression, position: int, extent: Optional[int] = None
) -> StringSQL:
    """
    Returns   VARCHAR  or  VARBINARY  value
    representing a substring of a specified
    string.

    Parameters
    ----------
    expr: SQLExpression
        Expression.
    position: int
        Starting position of the substring.
    extent: int, optional
        Length of the substring to extract.

    Returns
    -------
    StringSQL
        SQL string.
    """
    expr = format_magic(expr)
    if extent:
        position = f"{position}, {extent}"
    return StringSQL(f"SUBSTR({expr}, {position})", "text")


def upper(expr: SQLExpression) -> StringSQL:
    """
    Returns  a VARCHAR value containing  the
    argument converted to uppercase letters.

    Parameters
    ----------
    expr: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(f"UPPER({expr})", "text")


# Edit Distance & Soundex


@check_minimum_version
def edit_distance(
    expr1: SQLExpression,
    expr2: SQLExpression,
) -> StringSQL:
    """
    Calculates and returns the Levenshtein
    distance  between two  strings.

    Parameters
    ----------
    expr1: SQLExpression
        Expression.
    expr2: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.
    """
    expr1 = format_magic(expr1)
    expr2 = format_magic(expr2)
    return StringSQL(f"EDIT_DISTANCE({expr1}, {expr2})", "int")


levenshtein = edit_distance


@check_minimum_version
def soundex(expr: SQLExpression) -> StringSQL:
    """
    Returns Soundex encoding of a varchar
    strings  as a four character  string.

    Parameters
    ----------
    expr: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(f"SOUNDEX({expr})", "varchar")


@check_minimum_version
def soundex_matches(
    expr1: SQLExpression,
    expr2: SQLExpression,
) -> StringSQL:
    """
    Generates and compares Soundex encodings of
    two  strings,  and  returns a count of  the
    matching characters  (ranging from 0 for no
    match to 4 for an exact match).

    Parameters
    ----------
    expr1: SQLExpression
        Expression.
    expr2: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.
    """
    expr1 = format_magic(expr1)
    expr2 = format_magic(expr2)
    return StringSQL(f"SOUNDEX_MATCHES({expr1}, {expr2})", "int")


# Jaro & Jaro Winkler


@check_minimum_version
def jaro_distance(
    expr1: SQLExpression,
    expr2: SQLExpression,
) -> StringSQL:
    """
    Calculates and returns the Jaro distance
    between two strings.

    Parameters
    ----------
    expr1: SQLExpression
        Expression.
    expr2: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.
    """
    expr1 = format_magic(expr1)
    expr2 = format_magic(expr2)
    return StringSQL(f"JARO_DISTANCE({expr1}, {expr2})", "float")


@check_minimum_version
def jaro_winkler_distance(
    expr1: SQLExpression,
    expr2: SQLExpression,
) -> StringSQL:
    """
    Calculates and returns the Jaro-Winkler
    distance between two strings.

    Parameters
    ----------
    expr1: SQLExpression
        Expression.
    expr2: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.
    """
    expr1 = format_magic(expr1)
    expr2 = format_magic(expr2)
    return StringSQL(f"JARO_WINKLER_DISTANCE({expr1}, {expr2})", "float")
