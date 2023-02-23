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

from verticapy.core.str_sql.base import str_sql


def coalesce(expr, *argv):
    """
Returns the value of the first non-null expression in the list.

Parameters
----------
expr: object
    Expression.
argv: object
    Infinite Number of Expressions.

Returns
-------
str_sql
    SQL expression.
    """
    category = to_dtype_category(expr)
    expr = [format_magic(expr)]
    for arg in argv:
        expr += [format_magic(arg)]
    expr = ", ".join([str(elem) for elem in expr])
    return str_sql(f"COALESCE({expr})", category)


def nullifzero(expr):
    """
Evaluates to NULL if the value in the expression is 0.

Parameters
----------
expr: object
    Expression.

Returns
-------
str_sql
    SQL expression.
    """
    expr, cat = format_magic(expr, True)
    return str_sql(f"NULLIFZERO({expr})", cat)


def zeroifnull(expr):
    """
Evaluates to 0 if the expression is NULL.

Parameters
----------
expr: object
    Expression.

Returns
-------
str_sql
    SQL expression.
    """
    expr, cat = format_magic(expr, True)
    return str_sql(f"ZEROIFNULL({expr})", cat)
