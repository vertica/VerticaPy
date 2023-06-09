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
from verticapy._utils._sql._cast import to_dtype_category
from verticapy._utils._sql._format import format_magic

from verticapy.core.string_sql.base import StringSQL


def coalesce(expr: SQLExpression, *args) -> StringSQL:
    """
    Returns the value of the first non-null
    expression in the list.

    Parameters
    ----------
    expr: SQLExpression
        Expression.
    args: SQLExpression
        A number of expressions.

    Returns
    -------
    StringSQL
        SQL string.
    """
    category = to_dtype_category(expr)
    expr = [format_magic(expr)]
    for arg in args:
        expr += [format_magic(arg)]
    expr = ", ".join([str(elem) for elem in expr])
    return StringSQL(f"COALESCE({expr})", category)


def nullifzero(expr: SQLExpression) -> StringSQL:
    """
    Evaluates to NULL if the value in the
    expression is 0.

    Parameters
    ----------
    expr: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.
    """
    expr, cat = format_magic(expr, True)
    return StringSQL(f"NULLIFZERO({expr})", cat)


def zeroifnull(expr: SQLExpression) -> StringSQL:
    """
    Evaluates to 0 if the expression is NULL.

    Parameters
    ----------
    expr: SQLExpression
        Expression.

    Returns
    -------
    StringSQL
        SQL string.
    """
    expr, cat = format_magic(expr, True)
    return StringSQL(f"ZEROIFNULL({expr})", cat)
