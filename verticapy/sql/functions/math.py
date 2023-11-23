"""
Copyright  (c)  2018-2024 Open Text  or  one  of its
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [11.4, -2.5, 3.5, -4.2]})
        # apply the avg function, creating a "avg_x" column
        df["avg_x"] = vpf.apply("avg", df["x"])._over()
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [11.4, -2.5, 3.5, -4.2]})
        df["avg_x"] = vpf.apply("avg", df["x"])._over()
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_apply.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_apply.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [0, -1, -2, -3]})
        # apply the abs function to create a "abs_x" column
        df["abs_x"] = vpf.abs(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [0, -1, -2, -3]})
        df["abs_x"] = vpf.abs(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_abs.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_abs.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [0, -1, 0.7, 0.5]})
        # apply the acos function, creating a "acos_x" column
        df["acos_x"] = vpf.acos(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [0, -1, 0.7, 0.5]})
        df["acos_x"] = vpf.acos(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_acos.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_acos.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [0, -1, 0.7, 0.5]})
        # apply the asin function, creating a "asin_x" column
        df["asin_x"] = vpf.asin(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [0, -1, 0.7, 0.5]})
        df["asin_x"] = vpf.asin(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_asin.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_asin.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [0, -1, 0.7, 0.5]})
        # apply the atan function, creating a "atan_x" column
        df["atan_x"] = vpf.atan(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [0, -1, 0.7, 0.5]})
        df["atan_x"] = vpf.atan(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_atan.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_atan.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [0, -1, 0.7, 0.5],
                          "y": [2, 5, 1, 3]})
        # apply the atan^2 function, creating a "atan2" column
        df["atan2_x"] = vpf.atan2(df["x"], df["y"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [0, -1, 0.7, 0.5], "y": [2, 5, 1, 3]})
        df["atan2_x"] = vpf.atan2(df["x"], df["y"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_atan2.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_atan2.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [1, -2, 3, -4]})
        # apply the cbrt function, creating a "cbrt_x" column
        df["cbrt_x"] = vpf.cbrt(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [1, -2, 3, -4]})
        df["cbrt_x"] = vpf.cbrt(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_cbrt.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_cbrt.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [11.4, -2.5, 3.5, -4.2]})
        # apply the ceil function, creating a "ceil_x" column
        df["ceil_x"] = vpf.ceil(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [11.4, -2.5, 3.5, -4.2]})
        df["ceil_x"] = vpf.ceil(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_ceil.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_ceil.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [2, 10, 16, 33]})
        df["x"].astype("float")
        # apply the comb function, creating a "comb_x" column
        df["comb_x"] = vpf.comb(33, df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [2, 10, 16, 33]})
        df["x"].astype("float")
        df["comb_x"] = vpf.comb(33, df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_comb.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_comb.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [11.4, -2.5, 3.5, -4.2]})
        # apply the cos function, creating a "cos_x" column
        df["cos_x"] = vpf.cos(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [11.4, -2.5, 3.5, -4.2]})
        df["cos_x"] = vpf.cos(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_cos.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_cos.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [11.4, -2.5, 3.5, -4.2]})
        # apply the cosh function, creating a "cosh_x" column
        df["cosh_x"] = vpf.cosh(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [11.4, -2.5, 3.5, -4.2]})
        df["cosh_x"] = vpf.cosh(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_cosh.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_cosh.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [11.4, -2.5, 3.5, -4.2]})
        # apply the cot function, creating a "cot_x" column
        df["cot_x"] = vpf.cot(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [11.4, -2.5, 3.5, -4.2]})
        df["cot_x"] = vpf.cot(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_cot.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_cot.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [3.1415, 6, 4.5, 7]})
        # apply the degrees function, creating a "degrees_x" column
        df["degrees_x"] = vpf.degrees(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [3.1415, 6, 4.5, 7]})
        df["degrees_x"] = vpf.degrees(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_degrees.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_degrees.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"name0": ["Paris"],
                          "lat0": [48.864716],
                          "lon0": [2.349014],
                          "name1": ["Tunis"],
                          "lat1": [33.892166],
                          "lon1": [9.561555]})
        # apply the distance function, creating a "distance" column
        df["distance"] = vpf.distance(df["lat0"], df["lon0"], df["lat1"], df["lon1"])
        display(df[["name0", "name1", "distance"]])

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"name0": ["Paris"],
                        "lat0": [48.864716],
                        "lon0": [2.349014],
                        "name1": ["Tunis"],
                        "lat1": [33.892166],
                        "lon1": [9.561555]})
        df["distance"] = vpf.distance(df["lat0"], df["lon0"], df["lat1"], df["lon1"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_distance.html", "w")
        html_file.write(df[["name0", "name1", "distance"]]._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_distance.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [11.4, -2.5, 3.5, -4.2]})
        # apply the exp function, creating a "exp_x" column
        df["exp_x"] = vpf.exp(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [11.4, -2.5, 3.5, -4.2]})
        df["exp_x"] = vpf.exp(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_exp.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_exp.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [1, 2, 3, 4]})
        # apply the factorial function, creating a "factorial_x" column
        df["factorial_x"] = vpf.factorial(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [1, 2, 3, 4]})
        df["factorial_x"] = vpf.factorial(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_factorial.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_factorial.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [1.32, 2.9, 3.45, 4.33]})
        # apply the floor function, creating a "floor_x" column
        df["floor_x"] = vpf.floor(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [1.32, 2.9, 3.45, 4.33]})
        df["floor_x"] = vpf.floor(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_floor.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_floor.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [1.32, 2.9, 3.45, 4.33]})
        # apply the gamma function, creating a "gamma_x" column
        df["gamma_x"] = vpf.gamma(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [1.32, 2.9, 3.45, 4.33]})
        df["gamma_x"] = vpf.gamma(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_gamma.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_gamma.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": ['banana', 'apple', 'onion', 'potato']})
        # apply the hash function, creating a "hash_x" column
        df["hash_x"] = vpf.hash(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": ['banana', 'apple', 'onion', 'potato']})
        df["hash_x"] = vpf.hash(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_hash.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_hash.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": ['0', 'inf', 'nan', '15']})
        df["x"].astype("float")
        # apply the isfinite function, creating a "isfinite_x" column
        df["isfinite_x"] = vpf.isfinite(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": ['0', 'inf', 'nan', '15']})
        df["x"].astype("float")
        df["isfinite_x"] = vpf.isfinite(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_isfinite.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_isfinite.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": ['0', 'inf', '0.7', '15']})
        df["x"].astype("float")
        # apply the isinf function, creating a "isinf_x" column
        df["isinf_x"] = vpf.isinf(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": ['0', 'inf', '0.7', '15']})
        df["x"].astype("float")
        df["isinf_x"] = vpf.isinf(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_isinf.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_isinf.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": ['0', 'inf', 'nan', '15']})
        df["x"].astype("float")
        # apply the isnan function, creating a "isnan_x" column
        df["isnan_x"] = vpf.isnan(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": ['0', 'inf', 'nan', '15']})
        df["x"].astype("float")
        df["isnan_x"] = vpf.isnan(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_isnan.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_isnan.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [1.32, 2.9, 3.45, 4.33]})
        # apply the lgamma function, creating a "lgamma_x" column
        df["lgamma_x"] = vpf.lgamma(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [1.32, 2.9, 3.45, 4.33]})
        df["lgamma_x"] = vpf.lgamma(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_lgamma.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_lgamma.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [1.32, 2.9, 3.45, 4.33]})
        # apply the ln function, creating a "ln_x" column
        df["ln_x"] = vpf.ln(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [1.32, 2.9, 3.45, 4.33]})
        df["ln_x"] = vpf.ln(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_ln.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_ln.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [2, 10, 16, 33]})
        df["x"].astype("float")
        # apply the log function, creating a "log_x" column
        df["log_x"] = vpf.log(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [2, 10, 16, 33]})
        df["x"].astype("float")
        df["log_x"] = vpf.log(df["x"], 10)
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_log.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_log.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [30, 60, 180, 360]})
        # apply the radians function, creating a "radians_x" column
        df["radians_x"] = vpf.radians(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [30, 60, 180, 360]})
        df["radians_x"] = vpf.radians(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_radians.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_radians.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [2.95, 4.50, 4.63, 8.99]})
        df["x"].astype("float")
        # apply the round function, creating a "round_x" column
        df["round_x"] = vpf.round(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [2.95, 4.50, 4.63, 8.99]})
        df["x"].astype("float")
        df["round_x"] = vpf.round(df["x"], 1)
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_round.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_round.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [5, 10, -5, -14]})
        # apply the sign function, creating a "sign_x" column
        df["sign_x"] = vpf.sign(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [5, 10, -5, -14]})
        df["sign_x"] = vpf.sign(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_sign.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_sign.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [3.1415, 6, 4.5, 7]})
        # apply the sin function, creating a "sin_x" column
        df["sin_x"] = vpf.sin(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [3.1415, 6, 4.5, 7]})
        df["sin_x"] = vpf.sin(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_sin.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_sin.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [3.1415, 6, 4.5, 7]})
        # apply the sinh function, creating a "sinh_x" column
        df["sinh_x"] = vpf.sinh(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [3.1415, 6, 4.5, 7]})
        df["sinh_x"] = vpf.sinh(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_sinh.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_sinh.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [3.1415, 6, 4.5, 7]})
        # apply the sqrt function, creating a "sqrt_x" column
        df["sqrt_x"] = vpf.sqrt(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [3.1415, 6, 4.5, 7]})
        df["sqrt_x"] = vpf.sqrt(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_sqrt.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_sqrt.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [3.1415, 6, 4.5, 7]})
        # apply the tan function, creating a "tan_x" column
        df["tan_x"] = vpf.tan(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [3.1415, 6, 4.5, 7]})
        df["tan_x"] = vpf.tan(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_tan.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_tan.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [3.1415, 6, 4.5, 7]})
        # apply the tanh function, creating a "tanh_x" column
        df["tanh_x"] = vpf.tanh(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [3.1415, 6, 4.5, 7]})
        df["tanh_x"] = vpf.tanh(df["x"])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_tanh.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_tanh.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [2.95, 4.50, 4.63, 8.99]})
        df["x"].astype("float")
        # apply the trunc function, creating a "trunc_x" column
        df["trunc_x"] = vpf.trunc(df["x"], 1)
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [2.95, 4.50, 4.63, 8.99]})
        df["x"].astype("float")
        df["trunc_x"] = vpf.trunc(df["x"], 1)
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_math_trunc.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_math_trunc.html
    """
    expr = format_magic(expr)
    return StringSQL(f"TRUNC({expr}, {places})", "float")
