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
import os, math, shutil, re, time, decimal, warnings

# VerticaPy Modules
import verticapy
import vertica_python
from verticapy.toolbox import *

#
# ---#
pi = str_sql("PI()")
e = str_sql("EXP(1)")
tau = str_sql("2 * PI()")
inf = str_sql("'inf'::float")
nan = str_sql("'nan'::float")

# ---#
def abs(expr):
    """
---------------------------------------------------------------------------
Absolute Value.

Parameters
----------
expr: object
    Expression.
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
    """
    expr = format_magic(expr)
    return str_sql("ATAN({})".format(expr), "float")


# ---#
def cbrt(expr):
    """
---------------------------------------------------------------------------
Cube Root.

Parameters
----------
expr: object
    Expression.
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
    """
    expr = format_magic(expr)
    return str_sql("CEIL({})".format(expr), "float")


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
    """
    expr = format_magic(expr)
    return str_sql("COT({})".format(expr), "float")


# ---#
def degrees(expr):
    """
---------------------------------------------------------------------------
Convert Radians to Degrees.

Parameters
----------
expr: object
    Expression.
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
    """
    expr = format_magic(expr)
    return str_sql("EXP({})".format(expr), "float")


# ---#
def factorial(expr):
    """
---------------------------------------------------------------------------
Factorial.

Parameters
----------
expr: object
    Expression.
    """
    expr = format_magic(expr)
    return str_sql("({})!".format(expr), "float")


# ---#
def floor(expr):
    """
---------------------------------------------------------------------------
Floor Function.

Parameters
----------
expr: object
    Expression.
    """
    expr = format_magic(expr)
    return str_sql("FLOOR({})".format(expr), "float")


# ---#
def gamma(expr):
    """
---------------------------------------------------------------------------
Gamma Function.

Parameters
----------
expr: object
    Expression.
    """
    expr = format_magic(expr)
    return str_sql("({} - 1)!".format(expr), "float")


# ---#
def hash(*argv):
    """
---------------------------------------------------------------------------
Calculates a hash value over the function arguments.

Parameters
----------
argv: object
    Infinite Number of Expressions.
    """
    expr = []
    for arg in argv:
        expr += [format_magic(arg)]
    expr = ", ".join(expr)
    return str_sql("HASH({})".format(expr), "float")


# ---#
def isfinite(expr):
    """
---------------------------------------------------------------------------
Returns True if the expression is finite.

Parameters
----------
expr: object
    Expression.
    """
    expr = format_magic(expr)
    return str_sql(
        "(({}) = ({})) AND (ABS({}) < 'inf'::float)".format(expr, expr, expr), "float"
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
    """
    expr = format_magic(expr)
    return str_sql("(({}) != ({}))".format(expr, expr), "float")


# ---#
def lgamma(expr):
    """
---------------------------------------------------------------------------
Natural Logarithm of the expression Gamma.

Parameters
----------
expr: object
    Expression.
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
    """
    expr = format_magic(expr)
    return str_sql("LOG({}, {})".format(base, expr), "float")


# ---#
def radians(expr):
    """
---------------------------------------------------------------------------
Convert Degrees to Radians.

Parameters
----------
expr: object
    Expression.
    """
    expr = format_magic(expr)
    return str_sql("RADIANS({})".format(expr), "float")


# ---#
def random():
    """
---------------------------------------------------------------------------
Returns a Random Number.
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
    """
    return str_sql("RANDOMINT({})".format(n), "float")


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
    """
    expr = format_magic(expr)
    return str_sql("ROUND({}, {})".format(expr, places), "float")


# ---#
def sign(expr):
    """
---------------------------------------------------------------------------
Sign of the expression.

Parameters
----------
expr: object
    Expression.
    """
    expr = format_magic(expr)
    return str_sql("SIGN({})".format(expr), "float")


# ---#
def sin(expr):
    """
---------------------------------------------------------------------------
Trigonometric Sine.

Parameters
----------
expr: object
    Expression.
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
    """
    expr = format_magic(expr)
    return str_sql("TANH({})".format(expr), "float")


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
    """
    expr = format_magic(expr)
    return str_sql("TRUNC({}, {})".format(expr, places), "float")
