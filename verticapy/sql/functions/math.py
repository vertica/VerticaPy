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
from verticapy.core.str_sql.base import str_sql
from verticapy._utils._sql._format import format_magic

PI = str_sql("PI()")
E = str_sql("EXP(1)")
TAU = str_sql("2 * PI()")
INF = str_sql("'inf'::float")
NAN = str_sql("'nan'::float")

# General Function


def apply(func: str, *args, **kwargs):
    """
Applies any Vertica function on the input expressions.
Please check-out Vertica Documentation to see the available functions:

https://www.vertica.com/docs/

Parameters
----------
func : str
    Vertica Function. In case of geospatial, you can write the function name
    without the prefix ST_ or STV_.
args : object, optional
    Expressions.
kwargs: object, optional
    Optional Parameters Expressions.

Returns
-------
str_sql
    SQL expression.
    """
    ST_f = [
        "Area",
        "AsBinary",
        "Boundary",
        "Buffer",
        "Centroid",
        "Contains",
        "ConvexHull",
        "Crosses",
        "Difference",
        "Disjoint",
        "Distance",
        "Envelope",
        "Equals",
        "GeographyFromText",
        "GeographyFromWKB",
        "GeoHash",
        "GeometryN",
        "GeometryType",
        "GeomFromGeoHash",
        "GeomFromText",
        "GeomFromWKB",
        "Intersection",
        "Intersects",
        "IsEmpty",
        "IsSimple",
        "IsValid",
        "Length",
        "NumGeometries",
        "NumPoints",
        "Overlaps",
        "PointFromGeoHash",
        "PointN",
        "Relate",
        "SRID",
        "SymDifference",
        "Touches",
        "Transform",
        "Union",
        "Within",
        "X",
        "XMax",
        "XMin",
        "YMax",
        "YMin",
        "Y",
    ]
    STV_f = [
        "AsGeoJSON",
        "Create_Index",
        "Describe_Index",
        "Drop_Index",
        "DWithin",
        "Export2Shapefile",
        "Extent",
        "ForceLHR",
        "Geography",
        "GeographyPoint",
        "Geometry",
        "GeometryPoint",
        "GetExportShapefileDirectory",
        "Intersect",
        "IsValidReason",
        "LineStringPoint",
        "MemSize",
        "NN",
        "PolygonPoint",
        "Reverse",
        "Rename_Index",
        "Refresh_Index",
        "SetExportShapefileDirectory",
        "ShpSource",
        "ShpParser",
        "ShpCreateTable",
    ]
    ST_f_lower = [elem.lower() for elem in ST_f]
    STV_f_lower = [elem.lower() for elem in STV_f]
    if func.lower() in ST_f_lower:
        func = "ST_" + func
    elif func.lower() in STV_f_lower:
        func = "STV_" + func
    if len(args) > 0:
        expr = ", ".join([str(format_magic(elem)) for elem in args])
    else:
        expr = ""
    if len(kwargs) > 0:
        param_expr = ", ".join(
            [str((elem + " = ") + str(format_magic(kwargs[elem]))) for elem in kwargs]
        )
    else:
        param_expr = ""
    if param_expr:
        param_expr = " USING PARAMETERS " + param_expr
    func = func.upper()
    return str_sql(f"{func}({expr}{param_expr})")


# Other Math Functions


def abs(expr):
    """
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
    return str_sql(f"ABS({expr})", "float")


def acos(expr):
    """
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
    return str_sql(f"ACOS({expr})", "float")


def asin(expr):
    """
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
    return str_sql(f"ASIN({expr})", "float")


def atan(expr):
    """
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
    return str_sql(f"ATAN({expr})", "float")


def atan2(quotient, divisor):
    """
Trigonometric Inverse Tangent of the arithmetic dividend of the arguments.

Parameters
----------
quotient: object
    Expression representing the quotient.
divisor: object
    Expression representing the divisor.

Returns
-------
str_sql
    SQL expression.
    """
    quotient, divisor = format_magic(quotient), format_magic(divisor)
    return str_sql(f"ATAN2({quotient}, {divisor})", "float")


def cbrt(expr):
    """
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
    return str_sql(f"CBRT({expr})", "float")


def ceil(expr):
    """
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
    return str_sql(f"CEIL({expr})", "float")


def comb(n: int, k: int):
    """
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
    return str_sql(f"({n})! / (({k})! * ({n} - {k})!)", "float")


def cos(expr):
    """
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
    return str_sql(f"COS({expr})", "float")


def cosh(expr):
    """
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
    return str_sql(f"COSH({expr})", "float")


def cot(expr):
    """
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
    return str_sql(f"COT({expr})", "float")


def degrees(expr):
    """
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
    return str_sql(f"DEGREES({expr})", "float")


def distance(
    lat0: float, lon0: float, lat1: float, lon1: float, radius: float = 6371.009
):
    """
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
    return str_sql(f"DISTANCE({lat0}, {lon0}, {lat1}, {lon1}, {radius})", "float")


def exp(expr):
    """
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
    return str_sql(f"EXP({expr})", "float")


def factorial(expr):
    """
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
    return str_sql(f"({expr})!", "int")


def floor(expr):
    """
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
    return str_sql(f"FLOOR({expr})", "int")


def gamma(expr):
    """
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
    return str_sql(f"({expr} - 1)!", "float")


def hash(*argv):
    """
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
    return str_sql(f"HASH({expr})", "float")


def isfinite(expr):
    """
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
    return str_sql(f"(({expr}) = ({expr})) AND (ABS({expr}) < 'inf'::float)", cat)


def isinf(expr):
    """
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
    return str_sql(f"ABS({expr}) = 'inf'::float", "float")


def isnan(expr):
    """
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
    return str_sql(f"(({expr}) != ({expr}))", cat)


def lgamma(expr):
    """
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
    return str_sql(f"LN(({expr} - 1)!)", "float")


def ln(expr):
    """
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
    return str_sql(f"LN({expr})", "float")


def log(expr, base: int = 10):
    """
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
    return str_sql(f"LOG({base}, {expr})", "float")


def radians(expr):
    """
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
    return str_sql(f"RADIANS({expr})", "float")


def round(expr, places: int = 0):
    """
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
    return str_sql(f"ROUND({expr}, {places})", "float")


def sign(expr):
    """
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
    return str_sql(f"SIGN({expr})", "int")


def sin(expr):
    """
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
    return str_sql(f"SIN({expr})", "float")


def sinh(expr):
    """
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
    return str_sql(f"SINH({expr})", "float")


def sqrt(expr):
    """
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
    return str_sql(f"SQRT({expr})", "float")


def tan(expr):
    """
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
    return str_sql(f"TAN({expr})", "float")


def tanh(expr):
    """
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
    return str_sql(f"TANH({expr})", "float")


def trunc(expr, places: int = 0):
    """
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
    return str_sql(f"TRUNC({expr}, {places})", "float")
