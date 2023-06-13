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
from verticapy._typing import SQLExpression
from verticapy._utils._sql._format import format_magic

from verticapy.core.string_sql.base import StringSQL


def regexp_count(
    expr: SQLExpression,
    pattern: SQLExpression,
    position: int = 1,
) -> StringSQL:
    """
    Returns the number of times a regular expression
    matches a string.

    Parameters
    ----------
    expr: SQLExpression
        Expression.
    pattern: SQLExpression
        The regular expression to search for within
        the string.
    position: int, optional
        The number of characters from the start  of
        the  string where the function should start
        searching for matches.

    Returns
    -------
    StringSQL
        SQL string.
    """
    expr = format_magic(expr)
    pattern = format_magic(pattern)
    return StringSQL(f"REGEXP_COUNT({expr}, {pattern}, {position})", "int")


def regexp_ilike(expr: SQLExpression, pattern: SQLExpression) -> StringSQL:
    """
    Returns true if the string contains a match for
    the regular expression.

    Parameters
    ----------
    expr: SQLExpression
        Expression.
    pattern: SQLExpression
        A  string containing the regular expression
        to match against the string.

    Returns
    -------
    StringSQL
        SQL string.
    """
    expr = format_magic(expr)
    pattern = format_magic(pattern)
    return StringSQL(f"REGEXP_ILIKE({expr}, {pattern})")


def regexp_instr(
    expr: SQLExpression,
    pattern: SQLExpression,
    position: int = 1,
    occurrence: int = 1,
    return_position: int = 0,
) -> StringSQL:
    """
    Returns the  starting or  ending position  in a
    string  where  a  regular  expression  matches.

    Parameters
    ----------
    expr: SQLExpression
        Expression.
    pattern: SQLExpression
        The regular  expression to search for within
        the string.
    position: int, optional
        The number  of characters from the start  of
        the string where the  function should  start
        searching for matches.
    occurrence: int, optional
        Controls which occurrence of a pattern match
        in the string to return.
    return_position: int, optional
        Sets  the  position  within  the  string  to
        return.

    Returns
    -------
    StringSQL
        SQL string.
    """
    expr = format_magic(expr)
    pattern = format_magic(pattern)
    return StringSQL(
        f"REGEXP_INSTR({expr}, {pattern}, {position}, {occurrence}, {return_position})"
    )


def regexp_like(expr: SQLExpression, pattern: SQLExpression) -> StringSQL:
    """
    Returns true if the string matches the regular
    expression.

    Parameters
    ----------
    expr: SQLExpression
        Expression.
    pattern: SQLExpression
        A string containing the regular expression
        to match against the string.

    Returns
    -------
    StringSQL
        SQL string.
    """
    expr = format_magic(expr)
    pattern = format_magic(pattern)
    return StringSQL(f"REGEXP_LIKE({expr}, {pattern})")


def regexp_replace(
    expr: SQLExpression,
    target: SQLExpression,
    replacement: SQLExpression,
    position: int = 1,
    occurrence: int = 1,
) -> StringSQL:
    """
    Replace all occurrences of a substring that  match
    a  regular   expression  with  another  substring.

    Parameters
    ----------
    expr: SQLExpression
        Expression.
    target: SQLExpression
        The  regular  expression to search for  within
        the string.
    replacement: SQLExpression
        The string to replace matched substrings.
    position: int, optional
        The number of characters from the start of the
        string   where  the   function  should   start
        searching for matches.
    occurrence: int, optional
        Controls  which occurrence of a pattern  match
        in the string to return.

    Returns
    -------
    StringSQL
        SQL string.
    """
    expr = format_magic(expr)
    target = format_magic(target)
    replacement = format_magic(replacement)
    return StringSQL(
        f"REGEXP_REPLACE({expr}, {target}, {replacement}, {position}, {occurrence})"
    )


def regexp_substr(
    expr: SQLExpression, pattern: SQLExpression, position: int = 1, occurrence: int = 1
) -> StringSQL:
    """
    Returns the  substring  that matches a regular
    expression within a string.

    Parameters
    ----------
    expr: SQLExpression
        Expression.
    pattern: SQLExpression
        The regular expression to find a substring
        to extract.
    position: int, optional
        The number of characters from the start of
        the string where the function should start
        searching for matches.
    occurrence: int, optional
        Controls  which  occurrence  of a  pattern
        match in the string to return.

    Returns
    -------
    StringSQL
        SQL string.
    """
    expr = format_magic(expr)
    pattern = format_magic(pattern)
    return StringSQL(f"REGEXP_SUBSTR({expr}, {pattern}, {position}, {occurrence})")
