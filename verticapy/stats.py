# (c) Copyright [2018-2020] Micro Focus or one of its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# |_     |~) _  _| _  /~\    _ |.
# |_)\/  |_)(_|(_||   \_/|_|(_|||
#    /
#              ____________       ______
#             / __        `\     /     /
#            |  \/         /    /     /
#            |______      /    /     /
#                   |____/    /     /
#          _____________     /     /
#          \           /    /     /
#           \         /    /     /
#            \_______/    /     /
#             ______     /     /
#             \    /    /     /
#              \  /    /     /
#               \/    /     /
#                    /     /
#                   /     /
#                   \    /
#                    \  /
#                     \/
#                    _
# \  / _  __|_. _ _ |_)
#  \/ (/_|  | |(_(_|| \/
#                     /
# VerticaPy is a Python library with scikit-like functionality to use to conduct
# data science projects on data stored in Vertica, taking advantage Vertica’s
# speed and built-in analytics and machine learning features. It supports the
# entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize
# data transformation operations, and offers beautiful graphical options.
#
# VerticaPy aims to solve all of these problems. The idea is simple: instead
# of moving data around for processing, VerticaPy brings the logic to the data.
#
#
# Modules
#
# Standard Python Modules
import os, math, shutil, re, time, decimal, warnings, datetime

# VerticaPy Modules
import verticapy
import vertica_python
from verticapy.toolbox import *

#
# Variables
# ---#
pi = str_sql("PI()")
e = str_sql("EXP(1)")
tau = str_sql("2 * PI()")
inf = str_sql("'inf'::float")
nan = str_sql("'nan'::float")

# Aggregate & Analytical functions
# ---#
def avg(expr):
    """
---------------------------------------------------------------------------
Computes the average (arithmetic mean) of an expression over a group of rows.

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
    return str_sql("AVG({})".format(expr), "float")
mean=avg

# ---#
def bool_and(expr):
    """
---------------------------------------------------------------------------
Processes Boolean values and returns a Boolean value result. If all input 
values are true, BOOL_AND returns True. Otherwise it returns False.

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
    return str_sql("BOOL_AND({})".format(expr), "int")

# ---#
def bool_or(expr):
    """
---------------------------------------------------------------------------
Processes Boolean values and returns a Boolean value result. If at least one 
input value is true, BOOL_OR returns True. Otherwise, it returns False.

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
    return str_sql("BOOL_OR({})".format(expr), "int")


# ---#
def bool_xor(expr):
    """
---------------------------------------------------------------------------
Processes Boolean values and returns a Boolean value result. If specifically 
only one input value is true, BOOL_XOR returns True. Otherwise, it returns 
False.

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
    return str_sql("BOOL_XOR({})".format(expr), "int")


# ---#
def count(expr):
    """
---------------------------------------------------------------------------
Returns as a BIGINT the number of rows in each group where the expression is 
not NULL.

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
    return str_sql("COUNT({})".format(expr), "int")

# ---#
def max(expr):
    """
---------------------------------------------------------------------------
Returns the greatest value of an expression over a group of rows.

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
    return str_sql("MAX({})".format(expr), "float")


# ---#
def median(expr):
    """
---------------------------------------------------------------------------
Computes the approximate median of an expression over a group of rows.

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
    return str_sql("APPROXIMATE_MEDIAN({})".format(expr), "float")


# ---#
def min(expr):
    """
---------------------------------------------------------------------------
Returns the smallest value of an expression over a group of rows.

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
    return str_sql("MIN({})".format(expr), "float")


# ---#
def quantile(expr, number: float):
    """
---------------------------------------------------------------------------
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
str_sql
    SQL expression.
    """
    expr = format_magic(expr)
    return str_sql("APPROXIMATE_PERCENTILE({} USING PARAMETERS percentile = {})".format(expr, number), "float")


# ---#
def std(expr):
    """
---------------------------------------------------------------------------
Evaluates the statistical sample standard deviation for each member of the 
group.

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
    return str_sql("STDDEV({})".format(expr), "float")
stddev=std


# ---#
def sum(expr):
    """
---------------------------------------------------------------------------
Computes the sum of an expression over a group of rows.

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
    return str_sql("SUM({})".format(expr), "float")


# ---#
def var(expr):
    """
---------------------------------------------------------------------------
Evaluates the sample variance for each row of the group.

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
    return str_sql("VARIANCE({})".format(expr), "float")
variance=var


# Mathematical Functions
# ---#
def abs(expr):
    """
---------------------------------------------------------------------------
Absolute Value.

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
    return str_sql("ABS({})".format(expr), "float")


# ---#
def acos(expr):
    """
---------------------------------------------------------------------------
Trigonometric Inverse Cosine.

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
    return str_sql("ACOS({})".format(expr), "float")


# ---#
def asin(expr):
    """
---------------------------------------------------------------------------
Trigonometric Inverse Sine.

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
    return str_sql("ASIN({})".format(expr), "float")


# ---#
def atan(expr):
    """
---------------------------------------------------------------------------
Trigonometric Inverse Tangent.

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
    return str_sql("ATAN({})".format(expr), "float")


# ---#
def case_when(*argv):
    """
---------------------------------------------------------------------------
Returns the conditional statement of the input arguments.

Parameters
----------
argv: object
    Infinite Number of Expressions.
    The expression generated will look like:
    even: CASE ... WHEN argv[2 * i] THEN argv[2 * i + 1] ... END
    odd : CASE ... WHEN argv[2 * i] THEN argv[2 * i + 1] ... ELSE argv[n] END

Returns
-------
str_sql
    SQL expression.
    """
    n = len(argv)
    if n < 2:
        raise ParameterError(
            "The number of arguments of the 'case_when' function must be strictly greater than 1."
        )
    category = str_category(argv[1])
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
    return str_sql(expr, category)


# ---#
def cbrt(expr):
    """
---------------------------------------------------------------------------
Cube Root.

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
    return str_sql("CBRT({})".format(expr), "float")


# ---#
def ceil(expr):
    """
---------------------------------------------------------------------------
Ceiling Function.

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
    return str_sql("CEIL({})".format(expr), "float")


# ---#
def coalesce(expr, *argv):
    """
---------------------------------------------------------------------------
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
    category = str_category(expr)
    expr = [format_magic(expr)]
    for arg in argv:
        expr += [format_magic(arg)]
    expr = ", ".join([str(elem) for elem in expr])
    return str_sql("COALESCE({})".format(expr), category)


# ---#
def comb(n: int, k: int):
    """
---------------------------------------------------------------------------
Number of ways to choose k items from n items.

Parameters
----------
n : int
    items to choose from.
k : int
    items to choose.

Returns
-------
str_sql
    SQL expression.
    """
    return str_sql("({})! / (({})! * ({} - {})!)".format(n, k, n, k), "float")


# ---#
def cos(expr):
    """
---------------------------------------------------------------------------
Trigonometric Cosine.

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
    return str_sql("COS({})".format(expr), "float")


# ---#
def cosh(expr):
    """
---------------------------------------------------------------------------
Hyperbolic Cosine.

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
    return str_sql("COSH({})".format(expr), "float")


# ---#
def cot(expr):
    """
---------------------------------------------------------------------------
Trigonometric Cotangent.

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
    return str_sql("COT({})".format(expr), "float")


# ---#
def date(expr):
    """
---------------------------------------------------------------------------
Converts the input value to a DATE data type.

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
    return str_sql("DATE({})".format(expr), "date")


# ---#
def day(expr):
    """
---------------------------------------------------------------------------
Returns as an integer the day of the month from the input expression. 

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
    return str_sql("DAY({})".format(expr), "float")


# ---#
def dayofweek(expr):
    """
---------------------------------------------------------------------------
Returns the day of the week as an integer, where Sunday is day 1.

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
    return str_sql("DAYOFWEEK({})".format(expr), "float")


# ---#
def dayofyear(expr):
    """
---------------------------------------------------------------------------
Returns the day of the year as an integer, where January 1 is day 1.

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
    return str_sql("DAYOFYEAR({})".format(expr), "float")


# ---#
def decode(expr, *argv):
    """
---------------------------------------------------------------------------
Compares expression to each search value one by one.

Parameters
----------
expr: object
    Expression.
argv: object
    Infinite Number of Expressions.
    The expression generated will look like:
    even: CASE ... WHEN expr = argv[2 * i] THEN argv[2 * i + 1] ... END
    odd : CASE ... WHEN expr = argv[2 * i] THEN argv[2 * i + 1] ... ELSE argv[n] END

Returns
-------
str_sql
    SQL expression.
    """
    n = len(argv)
    if n < 2:
        raise ParameterError(
            "The number of arguments of the 'decode' function must be greater than 3."
        )
    category = str_category(argv[1])
    expr = (
        "DECODE("
        + str(format_magic(expr))
        + ", "
        + ", ".join([str(format_magic(elem)) for elem in argv])
        + ")"
    )
    return str_sql(expr, category)


# ---#
def degrees(expr):
    """
---------------------------------------------------------------------------
Converts Radians to Degrees.

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
    return str_sql("DEGREES({})".format(expr), "float")


# ---#
def distance(
    lat0: float, lon0: float, lat1: float, lon1: float, radius: float = 6371.009
):
    """
---------------------------------------------------------------------------
Returns the distance (in kilometers) between two points.

Parameters
----------
lat0: float
    Starting point latitude.
lon0: float
    Starting point longitude.
lat1: float
    Ending point latitude.
lon1: float
    Ending point longitude.
radius: float
    Specifies the radius of the curvature of the earth at the midpoint 
    between the starting and ending points.

Returns
-------
str_sql
    SQL expression.
    """
    return str_sql(
        "DISTANCE({}, {}, {}, {}, {})".format(lat0, lon0, lat1, lon1, radius), "float"
    )


# ---#
def exp(expr):
    """
---------------------------------------------------------------------------
Exponential.

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
    return str_sql("EXP({})".format(expr), "float")


# ---#
def extract(expr, field: str):
    """
---------------------------------------------------------------------------
Extracts a sub-field such as year or hour from a date/time expression.

Parameters
----------
expr: object
    Expression.
field: str
    The field to extract. It must be one of the following: 
 		CENTURY / DAY / DECADE / DOQ / DOW / DOY / EPOCH / HOUR / ISODOW / ISOWEEK /
 		ISOYEAR / MICROSECONDS / MILLENNIUM / MILLISECONDS / MINUTE / MONTH / QUARTER / 
 		SECOND / TIME ZONE / TIMEZONE_HOUR / TIMEZONE_MINUTE / WEEK / YEAR

Returns
-------
str_sql
    SQL expression.
    """
    expr = format_magic(expr)
    return str_sql("DATE_PART('{}', {})".format(field, expr), "int")


# ---#
def factorial(expr):
    """
---------------------------------------------------------------------------
Factorial.

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
    return str_sql("({})!".format(expr), "int")


# ---#
def floor(expr):
    """
---------------------------------------------------------------------------
Floor Function.

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
    return str_sql("FLOOR({})".format(expr), "int")


# ---#
def gamma(expr):
    """
---------------------------------------------------------------------------
Gamma Function.

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
    return str_sql("({} - 1)!".format(expr), "float")


# ---#
def getdate():
    """
---------------------------------------------------------------------------
Returns the current statement's start date and time as a TIMESTAMP value.

Returns
-------
str_sql
    SQL expression.
    """
    return str_sql("GETDATE()", "date")


# ---#
def getutcdate():
    """
---------------------------------------------------------------------------
Returns the current statement's start date and time at TIME ZONE 'UTC' 
as a TIMESTAMP value.

Returns
-------
str_sql
    SQL expression.
    """
    return str_sql("GETUTCDATE()", "date")


# ---#
def hash(*argv):
    """
---------------------------------------------------------------------------
Calculates a hash value over the function arguments.

Parameters
----------
argv: object
    Infinite Number of Expressions.

Returns
-------
str_sql
    SQL expression.
    """
    expr = []
    for arg in argv:
        expr += [format_magic(arg)]
    expr = ", ".join([str(elem) for elem in expr])
    return str_sql("HASH({})".format(expr), "float")


# ---#
def hour(expr):
    """
---------------------------------------------------------------------------
Returns the hour portion of the specified date as an integer, where 0 is 
00:00 to 00:59.

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
    return str_sql("HOUR({})".format(expr), "int")


# ---#
def interval(expr):
    """
---------------------------------------------------------------------------
Converts the input value to a INTERVAL data type.

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
    return str_sql("({})::interval".format(expr), "interval")


# ---#
def isfinite(expr):
    """
---------------------------------------------------------------------------
Returns True if the expression is finite.

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
    return str_sql(
        "(({}) = ({})) AND (ABS({}) < 'inf'::float)".format(expr, expr, expr), cat
    )


# ---#
def isinf(expr):
    """
---------------------------------------------------------------------------
Returns True if the expression is infinite.

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
    return str_sql("ABS({}) = 'inf'::float".format(expr), "float")


# ---#
def isnan(expr):
    """
---------------------------------------------------------------------------
Returns True if the expression is NaN.

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
    return str_sql("(({}) != ({}))".format(expr, expr), cat)


# ---#
def lgamma(expr):
    """
---------------------------------------------------------------------------
Natural Logarithm of the expression Gamma.

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
    return str_sql("LN(({} - 1)!)".format(expr), "float")


# ---#
def ln(expr):
    """
---------------------------------------------------------------------------
Natural Logarithm.

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
    return str_sql("LN({})".format(expr), "float")


# ---#
def log(expr, base: int = 10):
    """
---------------------------------------------------------------------------
Logarithm.

Parameters
----------
expr: object
    Expression.
base: int
    Specifies the base.

Returns
-------
str_sql
    SQL expression.
    """
    expr = format_magic(expr)
    return str_sql("LOG({}, {})".format(base, expr), "float")


# ---#
def minute(expr):
    """
---------------------------------------------------------------------------
Returns the minute portion of the specified date as an integer.

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
    return str_sql("MINUTE({})".format(expr), "int")


# ---#
def microsecond(expr):
    """
---------------------------------------------------------------------------
Returns the microsecond portion of the specified date as an integer.

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
    return str_sql("MICROSECOND({})".format(expr), "int")


# ---#
def month(expr):
    """
---------------------------------------------------------------------------
Returns the month portion of the specified date as an integer.

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
    return str_sql("MONTH({})".format(expr), "int")


# ---#
def nullifzero(expr):
    """
---------------------------------------------------------------------------
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
    return str_sql("NULLIFZERO({})".format(expr), cat)


# ---#
def overlaps(start0, end0, start1, end1):
    """
---------------------------------------------------------------------------
Evaluates two time periods and returns true when they overlap, false 
otherwise.

Parameters
----------
start0: object
    DATE, TIME, or TIMESTAMP/TIMESTAMPTZ value that specifies the beginning 
    of a time period.
end0: object
    DATE, TIME, or TIMESTAMP/TIMESTAMPTZ value that specifies the end of a 
    time period.
start1: object
    DATE, TIME, or TIMESTAMP/TIMESTAMPTZ value that specifies the beginning 
    of a time period.
end1: object
    DATE, TIME, or TIMESTAMP/TIMESTAMPTZ value that specifies the end of a 
    time period.

Returns
-------
str_sql
    SQL expression.
    """
    return str_sql(
        "({}, {}) OVERLAPS ({}, {})".format(
            format_magic(start0),
            format_magic(end0),
            format_magic(start1),
            format_magic(end1),
        ),
        "int",
    )


# ---#
def quarter(expr):
    """
---------------------------------------------------------------------------
Returns calendar quarter of the specified date as an integer, where the 
January-March quarter is 1.

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
    return str_sql("QUARTER({})".format(expr), "int")


# ---#
def radians(expr):
    """
---------------------------------------------------------------------------
Converts Degrees to Radians.

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
    return str_sql("RADIANS({})".format(expr), "float")


# ---#
def random():
    """
---------------------------------------------------------------------------
Returns a Random Number.

Returns
-------
str_sql
    SQL expression.
    """
    return str_sql("RANDOM()", "float")


# ---#
def randomint(n: int):
    """
---------------------------------------------------------------------------
Returns a Random Number from 0 through n – 1.

Parameters
----------
n: int
    Integer Value.

Returns
-------
str_sql
    SQL expression.
    """
    return str_sql("RANDOMINT({})".format(n), "int")


# ---#
def round(expr, places: int = 0):
    """
---------------------------------------------------------------------------
Rounds the expression.

Parameters
----------
expr: object
    Expression.
places: int
    Number used to round the expression.

Returns
-------
str_sql
    SQL expression.
    """
    expr = format_magic(expr)
    return str_sql("ROUND({}, {})".format(expr, places), "float")


# ---#
def round_date(expr, precision: str = "DD"):
    """
---------------------------------------------------------------------------
Rounds the specified date or time.

Parameters
----------
expr: object
    Expression.
precision: str, optional
    A string constant that specifies precision for the rounded value, 
    one of the following:
	    Century: CC | SCC
	    Year: SYYY | YYYY | YEAR | YYY | YY | Y
	    ISO Year: IYYY | IYY | IY | I
	    Quarter: Q
	    Month: MONTH | MON | MM | RM
	    Same weekday as first day of year: WW
	    Same weekday as first day of ISO year: IW
	    Same weekday as first day of month: W
	    Day (default): DDD | DD | J
	    First weekday: DAY | DY | D
	    Hour: HH | HH12 | HH24
	    Minute: MI
	    Second: SS

Returns
-------
str_sql
    SQL expression.
    """
    expr = format_magic(expr)
    return str_sql("ROUND({}, '{}')".format(expr, precision), "date")


# ---#
def second(expr):
    """
---------------------------------------------------------------------------
Returns the seconds portion of the specified date as an integer.

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
    return str_sql("SECOND({})".format(expr), "int")


# ---#
def seeded_random(random_state: int):
    """
---------------------------------------------------------------------------
Returns a Seeded Random Number using the input random state.

Parameters
----------
random_state: int
    Integer used to seed the randomness.

Returns
-------
str_sql
    SQL expression.
    """
    return str_sql("SEEDED_RANDOM({})".format(random_state), "float")


# ---#
def sign(expr):
    """
---------------------------------------------------------------------------
Sign of the expression.

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
    return str_sql("SIGN({})".format(expr), "int")


# ---#
def sin(expr):
    """
---------------------------------------------------------------------------
Trigonometric Sine.

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
    return str_sql("SIN({})".format(expr), "float")


# ---#
def sinh(expr):
    """
---------------------------------------------------------------------------
Hyperbolic Sine.

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
    return str_sql("SINH({})".format(expr), "float")


# ---#
def sqrt(expr):
    """
---------------------------------------------------------------------------
Arithmetic Square Root.

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
    return str_sql("SQRT({})".format(expr), "float")


# ---#
def tan(expr):
    """
---------------------------------------------------------------------------
Trigonometric Tangent.

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
    return str_sql("TAN({})".format(expr), "float")


# ---#
def tanh(expr):
    """
---------------------------------------------------------------------------
Hyperbolic Tangent.

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
    return str_sql("TANH({})".format(expr), "float")


# ---#
def timestamp(expr):
    """
---------------------------------------------------------------------------
Converts the input value to a TIMESTAMP data type.

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
    return str_sql("({})::timestamp".format(expr), "date")


# ---#
def trunc(expr, places: int = 0):
    """
---------------------------------------------------------------------------
Truncates the expression.

Parameters
----------
expr: object
    Expression.
places: int
    Number used to truncate the expression.

Returns
-------
str_sql
    SQL expression.
    """
    expr = format_magic(expr)
    return str_sql("TRUNC({}, {})".format(expr, places), "float")


# ---#
def week(expr):
    """
---------------------------------------------------------------------------
Returns the week of the year for the specified date as an integer, where the 
first week begins on the first Sunday on or preceding January 1.

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
    return str_sql("WEEK({})".format(expr), "int")


# ---#
def year(expr):
    """
---------------------------------------------------------------------------
Returns an integer that represents the year portion of the specified date.

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
    return str_sql("YEAR({})".format(expr), "int")


# ---#
def zeroifnull(expr):
    """
---------------------------------------------------------------------------
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
    return str_sql("ZEROIFNULL({})".format(expr), cat)
