"""
Copyright  (c)  2018-2023 Open Text  or  one  of its
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

    Examples
    --------
    .. code-block:: python

        from verticapy import TableSample
        import verticapy.stats as st

        df = TableSample({"x": [0.8, -1, None, -2, None]}).to_vdf()
        # apply the coalesce function to create a "coalesce_x" column
        df["coalesce_x"] = st.coalesce(df["x"], 777)
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import TableSample
        import verticapy.stats as st
        df = TableSample({"x": [0.8, -1, None, -2, None]}).to_vdf()
        df["coalesce_x"] = st.coalesce(df["x"], 777)
        html_file = open("figures/sql_functions_null_handling_coalesce.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_null_handling_coalesce.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import TableSample
        import verticapy.stats as st

        df = TableSample({"x": [0, 0, 0.7, 15]}).to_vdf()
        # apply the nullifzero function to create a "nullifzero_x" column
        df["nullifzero_x"] = st.nullifzero(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import TableSample
        import verticapy.stats as st
        df = TableSample({"x": [0, 0, 0.7, 15]}).to_vdf()
        df["nullifzero_x"] = st.nullifzero(df["x"])
        html_file = open("figures/sql_functions_null_handling_nullifzero.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_null_handling_nullifzero.html
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

    Examples
    --------
    .. code-block:: python

        from verticapy import TableSample
        import verticapy.stats as st

        df = TableSample({"x": [0, None, 0.7, None]}).to_vdf()
        # apply the zeroifnull function to create a "zeroifnull_x" column
        df["zeroifnull_x"] = st.zeroifnull(df["x"])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import TableSample
        import verticapy.stats as st
        df = TableSample({"x": [0, None, 0.7, None]}).to_vdf()
        df["zeroifnull_x"] = st.zeroifnull(df["x"])
        html_file = open("figures/sql_functions_null_handling_zeroifnull.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_null_handling_zeroifnull.html
    """
    expr, cat = format_magic(expr, True)
    return StringSQL(f"ZEROIFNULL({expr})", cat)
