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

"""
Constants.
"""

E = StringSQL("EXP(1)")
INF = StringSQL("'inf'::float")
NAN = StringSQL("'nan'::float")
PI = StringSQL("PI()")
TAU = StringSQL("2 * PI()")

"""
General Function.
"""


def apply(func: SQLExpression, *args, **kwargs) -> StringSQL:
    """
    Applies any Vertica function on the input
    expressions.
    Please check-out the Vertica Documentation
    to see the available functions:
    https://www.vertica.com/docs/

    Parameters
    ----------
    func: SQLExpression
        Vertica Function. For geospatial
        functions, you can write  the function
        name without the ST_ or STV_ prefix.
    args: SQLExpression, optional
        Expressions.
    kwargs: SQLExpression, optional
        Optional Parameters Expressions.

    Returns
    -------
    StringSQL
        SQL string.
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
    return StringSQL(f"{func}({expr}{param_expr})")


"""
Mathematical Functions.
"""


def abs(expr: SQLExpression) -> StringSQL:
    """
    Absolute Value.

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
    return StringSQL(f"ABS({expr})", "float")


def acos(expr: SQLExpression) -> StringSQL:
    """
    Trigonometric Inverse Cosine.

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
    return StringSQL(f"ACOS({expr})", "float")


def asin(expr: SQLExpression) -> StringSQL:
    """
    Trigonometric Inverse Sine.

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
    return StringSQL(f"ASIN({expr})", "float")


def atan(expr: SQLExpression) -> StringSQL:
    """
    Trigonometric Inverse Tangent.

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
    return StringSQL(f"ATAN({expr})", "float")


def atan2(quotient: SQLExpression, divisor: SQLExpression) -> StringSQL:
    """
    Trigonometric Inverse Tangent of the arithmetic
    dividend of the arguments.

    Parameters
    ----------
    quotient: SQLExpression
        Expression representing the quotient.
    divisor: SQLExpression
        Expression representing the divisor.

    Returns
    -------
    StringSQL
        SQL string.
    """
    quotient, divisor = format_magic(quotient), format_magic(divisor)
    return StringSQL(f"ATAN2({quotient}, {divisor})", "float")


def cbrt(expr: SQLExpression) -> StringSQL:
    """
    Cube Root.

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
    return StringSQL(f"CBRT({expr})", "float")


def ceil(expr: SQLExpression) -> StringSQL:
    """
    Ceiling Function.

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
    return StringSQL(f"CEIL({expr})", "float")


def comb(n: int, k: int) -> StringSQL:
    """
    Number of ways to choose k items from n items.

    Parameters
    ----------
    n: int
        Items to choose from.
    k: int
        Items to choose.

    Returns
    -------
    StringSQL
        SQL string.
    """
    return StringSQL(f"({n})! / (({k})! * ({n} - {k})!)", "float")


def cos(expr: SQLExpression) -> StringSQL:
    """
    Trigonometric Cosine.

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
    return StringSQL(f"COS({expr})", "float")


def cosh(expr: SQLExpression) -> StringSQL:
    """
    Hyperbolic Cosine.

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
    return StringSQL(f"COSH({expr})", "float")


def cot(expr: SQLExpression) -> StringSQL:
    """
    Trigonometric Cotangent.

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
    return StringSQL(f"COT({expr})", "float")


def degrees(expr: SQLExpression) -> StringSQL:
    """
    Converts Radians to Degrees.

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
    return StringSQL(f"DEGREES({expr})", "float")


def distance(
    lat0: float, lon0: float, lat1: float, lon1: float, radius: float = 6371.009
) -> StringSQL:
    """
    Returns the distance (in kilometers) between two
    points.

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
        Specifies the radius of the curvature of the
        earth  at the midpoint between the  starting
        and ending points.

    Returns
    -------
    StringSQL
        SQL string.
    """
    return StringSQL(f"DISTANCE({lat0}, {lon0}, {lat1}, {lon1}, {radius})", "float")


def exp(expr: SQLExpression) -> StringSQL:
    """
    Exponential.

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
    return StringSQL(f"EXP({expr})", "float")


def factorial(expr: SQLExpression) -> StringSQL:
    """
    Factorial.

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
    return StringSQL(f"({expr})!", "int")


def floor(expr: SQLExpression) -> StringSQL:
    """
    Floor Function.

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
    return StringSQL(f"FLOOR({expr})", "int")


def gamma(expr: SQLExpression) -> StringSQL:
    """
    Gamma Function.

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
    return StringSQL(f"({expr} - 1)!", "float")


def hash(*args) -> StringSQL:
    """
    Calculates a hash value over the function
    arguments.

    Parameters
    ----------
    args: SQLExpression
        Infinite Number of Expressions.

    Returns
    -------
    StringSQL
        SQL string.
    """
    expr = []
    for arg in args:
        expr += [format_magic(arg)]
    expr = ", ".join([str(elem) for elem in expr])
    return StringSQL(f"HASH({expr})", "float")


def isfinite(expr: SQLExpression) -> StringSQL:
    """
    Returns True if the expression is finite.

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
    return StringSQL(f"(({expr}) = ({expr})) AND (ABS({expr}) < 'inf'::float)", cat)


def isinf(expr: SQLExpression) -> StringSQL:
    """
    Returns True if the expression is infinite.

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
    return StringSQL(f"ABS({expr}) = 'inf'::float", "float")


def isnan(expr: SQLExpression) -> StringSQL:
    """
    Returns True if the expression is NaN.

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
    return StringSQL(f"(({expr}) != ({expr}))", cat)


def lgamma(expr: SQLExpression) -> StringSQL:
    """
    Natural Logarithm of the expression Gamma.

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
    return StringSQL(f"LN(({expr} - 1)!)", "float")


def ln(expr: SQLExpression) -> StringSQL:
    """
    Natural Logarithm.

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
    return StringSQL(f"LN({expr})", "float")


def log(expr: SQLExpression, base: int = 10) -> StringSQL:
    """
    Logarithm.

    Parameters
    ----------
    expr: SQLExpression
        Expression.
    base: int
        Specifies the base.

    Returns
    -------
    StringSQL
        SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(f"LOG({base}, {expr})", "float")


def radians(expr: SQLExpression) -> StringSQL:
    """
    Converts Degrees to Radians.

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
    return StringSQL(f"RADIANS({expr})", "float")


def round(expr: SQLExpression, places: int = 0) -> StringSQL:
    """
    Rounds the expression.

    Parameters
    ----------
    expr: SQLExpression
        Expression.
    places: int
        Number used to round the expression.

    Returns
    -------
    StringSQL
        SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(f"ROUND({expr}, {places})", "float")


def sign(expr: SQLExpression) -> StringSQL:
    """
    Sign of the expression.

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
    return StringSQL(f"SIGN({expr})", "int")


def sin(expr: SQLExpression) -> StringSQL:
    """
    Trigonometric Sine.

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
    return StringSQL(f"SIN({expr})", "float")


def sinh(expr: SQLExpression) -> StringSQL:
    """
    Hyperbolic Sine.

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
    return StringSQL(f"SINH({expr})", "float")


def sqrt(expr: SQLExpression) -> StringSQL:
    """
    Arithmetic Square Root.

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
    return StringSQL(f"SQRT({expr})", "float")


def tan(expr: SQLExpression) -> StringSQL:
    """
    Trigonometric Tangent.

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
    return StringSQL(f"TAN({expr})", "float")


def tanh(expr: SQLExpression) -> StringSQL:
    """
    Hyperbolic Tangent.

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
    return StringSQL(f"TANH({expr})", "float")


def trunc(expr: SQLExpression, places: int = 0) -> StringSQL:
    """
    Truncates the expression.

    Parameters
    ----------
    expr: SQLExpression
        Expression.
    places: int
        Number used to truncate the expression.

    Returns
    -------
    StringSQL
        SQL string.
    """
    expr = format_magic(expr)
    return StringSQL(f"TRUNC({expr}, {places})", "float")
