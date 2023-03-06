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
from verticapy._utils._sql._cast import to_dtype_category
from verticapy._utils._sql._format import format_magic
from verticapy._typing import SQLExpression
from verticapy.errors import ParameterError

from verticapy.core.string_sql.base import StringSQL


def case_when(*argv) -> StringSQL:
    """
    Returns the conditional statement of the input 
    arguments.

    Parameters
    ----------
    argv: SQLExpression
        Infinite Number of Expressions.
        The expression generated will look like:
        even: CASE ... WHEN argv[2 * i] 
                       THEN argv[2 * i + 1] ... END
        odd : CASE ... WHEN argv[2 * i] 
                       THEN argv[2 * i + 1] ... 
                       ELSE argv[n] END

    Returns
    -------
    StringSQL
        SQL string.
    """
    n = len(argv)
    if n < 2:
        raise ParameterError(
            "The number of arguments of the 'case_when' function must be strictly greater than 1."
        )
    category = to_dtype_category(argv[1])
    i = 0
    expr = "CASE"
    while i < n:
        if i + 1 == n:
            expr += " ELSE " + str(format_magic(argv[i]))
            i += 1
        else:
            expr += (
                " WHEN "
                + str(format_magic(argv[i]))
                + " THEN "
                + str(format_magic(argv[i + 1]))
            )
            i += 2
    expr += " END"
    return StringSQL(expr, category)


def decode(expr: SQLExpression, *argv) -> StringSQL:
    """
    Compares expression to each search value one 
    by one.

    Parameters
    ----------
    expr: SQLExpression
        Expression.
    argv: SQLExpression
        Infinite Number of Expressions.
        The expression generated will look like:
        even: CASE ... WHEN expr = argv[2 * i] 
                       THEN argv[2 * i + 1] ... END
        odd : CASE ... WHEN expr = argv[2 * i] 
                       THEN argv[2 * i + 1] ... 
                       ELSE argv[n] END

    Returns
    -------
    StringSQL
        SQL string.
    """
    n = len(argv)
    if n < 2:
        raise ParameterError(
            "The number of arguments of the 'decode' function must be greater than 3."
        )
    category = to_dtype_category(argv[1])
    expr = (
        "DECODE("
        + str(format_magic(expr))
        + ", "
        + ", ".join([str(format_magic(elem)) for elem in argv])
        + ")"
    )
    return StringSQL(expr, category)
