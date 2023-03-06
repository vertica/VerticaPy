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


def avg(expr):
    """
Computes the average (arithmetic mean) of an expression over a group of rows.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(f"AVG({expr})", "float")


mean = avg


def bool_and(expr):
    """
Processes Boolean values and returns a Boolean value result. If all input 
values are true, BOOL_AND returns True. Otherwise it returns False.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(f"BOOL_AND({expr})", "int")


def bool_or(expr):
    """
Processes Boolean values and returns a Boolean value result. If at least one 
input value is true, BOOL_OR returns True. Otherwise, it returns False.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(f"BOOL_OR({expr})", "int")


def bool_xor(expr):
    """
Processes Boolean values and returns a Boolean value result. If specifically 
only one input value is true, BOOL_XOR returns True. Otherwise, it returns 
False.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(f"BOOL_XOR({expr})", "int")


def conditional_change_event(expr):
    """
Assigns an event window number to each row, starting from 0, and increments 
by 1 when the result of evaluating the argument expression on the current 
row differs from that on the previous row.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(f"CONDITIONAL_CHANGE_EVENT({expr})", "int")


def conditional_true_event(expr):
    """
Assigns an event window number to each row, starting from 0, and increments 
the number by 1 when the result of the boolean argument expression evaluates 
true.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(f"CONDITIONAL_TRUE_EVENT({expr})", "int")


def count(expr):
    """
Returns as a BIGINT the number of rows in each group where the expression is 
not NULL.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(f"COUNT({expr})", "int")


def lag(expr, offset: int = 1):
    """
Returns the value of the input expression at the given offset before the 
current row within a window. 

Parameters
----------
expr: object
    Expression.
offset: int
    Indicates how great is the lag.

Returns
-------
StringSQL
    SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(f"LAG({expr}, {offset})")


def lead(expr, offset: int = 1):
    """
Returns values from the row after the current row within a window, letting 
you access more than one row in a table at the same time. 

Parameters
----------
expr: object
    Expression.
offset: int
    Indicates how great is the lead.

Returns
-------
StringSQL
    SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(f"LEAD({expr}, {offset})")


def max(expr):
    """
Returns the greatest value of an expression over a group of rows.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(f"MAX({expr})", "float")


def median(expr):
    """
Computes the approximate median of an expression over a group of rows.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(f"APPROXIMATE_MEDIAN({expr})", "float")


def min(expr):
    """
Returns the smallest value of an expression over a group of rows.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(f"MIN({expr})", "float")


def nth_value(expr, row_number: int):
    """
Returns the value evaluated at the row that is the nth row of the window 
(counting from 1).

Parameters
----------
expr: object
    Expression.
row_number: int
    Specifies the row to evaluate.

Returns
-------
StringSQL
    SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(f"NTH_VALUE({expr}, {row_number})", "int")


def quantile(expr, number: float):
    """
Computes the approximate percentile of an expression over a group of rows.

Parameters
----------
expr: object
    Expression.
number: float
    Percentile value, which must be a FLOAT constant ranging from 0 to 1 
    (inclusive).

Returns
-------
StringSQL
    SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(
        f"APPROXIMATE_PERCENTILE({expr} USING PARAMETERS percentile = {number})",
        "float",
    )


def rank():
    """
Within each window partition, ranks all rows in the query results set 
according to the order specified by the window's ORDER BY clause.

Returns
-------
StringSQL
    SQL string.
    """
    return StringSQL("RANK()", "int")


def row_number():
    """
Assigns a sequence of unique numbers, starting from 1, to each row in a 
window partition.

Returns
-------
StringSQL
    SQL string.
    """
    return StringSQL("ROW_NUMBER()", "int")


def std(expr):
    """
Evaluates the statistical sample standard deviation for each member of the 
group.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(f"STDDEV({expr})", "float")


stddev = std


def sum(expr):
    """
Computes the sum of an expression over a group of rows.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(f"SUM({expr})", "float")


def var(expr):
    """
Evaluates the sample variance for each row of the group.

Parameters
----------
expr: object
    Expression.

Returns
-------
StringSQL
    SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(f"VARIANCE({expr})", "float")


variance = var
