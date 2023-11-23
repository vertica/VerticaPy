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


def avg(expr: SQLExpression) -> StringSQL:
    """
    Computes the average (arithmetic mean) of
    an  expression  over  a  group  of  rows.

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

        df = vDataFrame({"x": [2, -11, 7, 12]})
        # apply the avg function
        df.select([str(vpf.avg(df["x"]))])

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [2, -11, 7, 12]})
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_analytic_avg.html", "w")
        html_file.write(df.select([str(vpf.avg(df["x"]))])._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_analytic_avg.html
    """
    expr = format_magic(expr)
    return StringSQL(f"AVG({expr})", "float")


mean = avg


def bool_and(expr: SQLExpression) -> StringSQL:
    """
    Processes  Boolean  values and returns  a  Boolean
    value  result.  If  all  input  values  are  true,
    BOOL_AND returns True. Otherwise it returns False.

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

        df = vDataFrame({"x": [True, False, True, True]})
        # apply the bool_and function
        df.select([str(vpf.bool_and(df["x"]))])

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [True, False, True, True]})
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_analytic_bool_and.html", "w")
        html_file.write(df.select([str(vpf.bool_and(df["x"]))])._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_analytic_bool_and.html
    """
    expr = format_magic(expr)
    return StringSQL(f"BOOL_AND({expr})", "int")


def bool_or(expr: SQLExpression) -> StringSQL:
    """
    Processes Boolean values and returns a Boolean
    value  result. If at least one input value  is
    true,  BOOL_OR  returns  True.  Otherwise,  it
    returns False.

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

        df = vDataFrame({"x": [True, False, True, True]})
        # apply the bool_or function
        df.select([str(vpf.bool_or(df["x"]))])

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [True, False, True, True]})
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_analytic_bool_or.html", "w")
        html_file.write(df.select([str(vpf.bool_or(df["x"]))])._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_analytic_bool_or.html
    """
    expr = format_magic(expr)
    return StringSQL(f"BOOL_OR({expr})", "int")


def bool_xor(expr: SQLExpression) -> StringSQL:
    """
    Processes  Boolean values and  returns a Boolean
    value  result.  If  specifically only one  input
    value is true, BOOL_XOR returns True. Otherwise,
    it returns False.

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

        df = vDataFrame({"x": [True, False, True, True]})
        # apply the bool_xor function
        df.select([str(vpf.bool_xor(df["x"]))])

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [True, False, True, True]})
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_analytic_bool_xor.html", "w")
        html_file.write(df.select([str(vpf.bool_xor(df["x"]))])._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_analytic_bool_xor.html
    """
    expr = format_magic(expr)
    return StringSQL(f"BOOL_XOR({expr})", "int")


def conditional_change_event(expr: SQLExpression) -> StringSQL:
    """
    Assigns an event window number to each row,  starting
    from  0,  and  increments  by  1 when the  result  of
    evaluating the argument expression on the current row
    differs from that on the previous row.

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

        df = vDataFrame({"x": [1, 2, 3, 4, 5, 6],
                          "y": [11.4, -2.5, 3.5, -4.2, 2, 3]})
        # apply the conditional_change_event function
        df["cchange_event"] = vpf.conditional_change_event(df["y"] > 0)._over(order_by = [df["x"]])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [1, 2, 3, 4, 5, 6],
                          "y": [11.4, -2.5, 3.5, -4.2, 2, 3]})
        df["cchange_event"] = vpf.conditional_change_event(df["y"] > 0)._over(order_by = [df["x"]])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_analytic_conditional_change_event.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_analytic_conditional_change_event.html
    """
    expr = format_magic(expr)
    return StringSQL(f"CONDITIONAL_CHANGE_EVENT({expr})", "int")


def conditional_true_event(expr: SQLExpression) -> StringSQL:
    """
    Assigns an  event window number to each row,  starting
    from 0, and increments the number by 1 when the result
    of  the boolean  argument expression  evaluates  true.

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

        df = vDataFrame({"x": [1, 2, 3, 4, 5, 6],
                          "y": [11.4, -2.5, 3.5, -4.2, 2, 3]})
        # apply the conditional_true_event function
        df["ctrue_event"] = vpf.conditional_true_event(df["y"] > 0)._over(order_by = [df["x"]])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [1, 2, 3, 4, 5, 6],
                          "y": [11.4, -2.5, 3.5, -4.2, 2, 3]})
        df["ctrue_event"] = vpf.conditional_true_event(df["y"] > 0)._over(order_by = [df["x"]])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_analytic_conditional_true_event.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_analytic_conditional_true_event.html
    """
    expr = format_magic(expr)
    return StringSQL(f"CONDITIONAL_TRUE_EVENT({expr})", "int")


def count(expr: SQLExpression) -> StringSQL:
    """
    Returns as a BIGINT the number of rows in each group
    where the expression is not NULL.

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

        df = vDataFrame({"x": [2, -11, None, 12]})
        # apply the count function
        df.select([str(vpf.count(df["x"]))])

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [2, -11, None, 12]})
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_analytic_count.html", "w")
        html_file.write(df.select([str(vpf.count(df["x"]))])._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_analytic_count.html
    """
    expr = format_magic(expr)
    return StringSQL(f"COUNT({expr})", "int")


def lag(expr: SQLExpression, offset: int = 1) -> StringSQL:
    """
    Returns the value of the input expression at the given
    offset before the current row within a window.

    Parameters
    ----------
    expr: SQLExpression
        Expression.
    offset: int
        Indicates how great is the lag.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [1, 2, 3, 4],
                          "y": [11.4, -2.5, 3.5, -4.2]})
        # apply the lag function
        df["lag"] = vpf.lag(df["y"], 1)._over(order_by = [df["x"]])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [1, 2, 3, 4],
                          "y": [11.4, -2.5, 3.5, -4.2]})
        df["lag"] = vpf.lag(df["y"], 1)._over(order_by = [df["x"]])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_analytic_lag.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_analytic_lag.html
    """
    expr = format_magic(expr)
    return StringSQL(f"LAG({expr}, {offset})")


def lead(expr: SQLExpression, offset: int = 1) -> StringSQL:
    """
    Returns values  from the row after the current row within
    a window, letting you access more than one row in a table
    at the same time.

    Parameters
    ----------
    expr: SQLExpression
        Expression.
    offset: int
        Indicates how great is the lead.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [1, 2, 3, 4],
                          "y": [11.4, -2.5, 3.5, -4.2]})
        # apply the lead function
        df["lead"] = vpf.lead(df["y"], 1)._over(order_by = [df["x"]])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [1, 2, 3, 4],
                          "y": [11.4, -2.5, 3.5, -4.2]})
        df["lead"] = vpf.lead(df["y"], 1)._over(order_by = [df["x"]])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_analytic_lead.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_analytic_lead.html
    """
    expr = format_magic(expr)
    return StringSQL(f"LEAD({expr}, {offset})")


def max(expr: SQLExpression) -> StringSQL:
    """
    Returns the greatest value of an expression
    over a group of rows.

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

        df = vDataFrame({"x": [2, -11, 7, 12]})
        # apply the max function
        df.select([str(vpf.max(df["x"]))])

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [2, -11, 7, 12]})
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_analytic_max.html", "w")
        html_file.write(df.select([str(vpf.max(df["x"]))])._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_analytic_max.html
    """
    expr = format_magic(expr)
    return StringSQL(f"MAX({expr})", "float")


def median(expr: SQLExpression) -> StringSQL:
    """
    Computes the approximate median of an expression
    over a group of rows.

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

        df = vDataFrame({"x": [2, -11, 7, 12]})
        # apply the median function
        df.select([str(vpf.median(df["x"]))])

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [2, -11, 7, 12]})
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_analytic_median.html", "w")
        html_file.write(df.select([str(vpf.median(df["x"]))])._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_analytic_median.html
    """
    expr = format_magic(expr)
    return StringSQL(f"APPROXIMATE_MEDIAN({expr})", "float")


def min(expr: SQLExpression) -> StringSQL:
    """
    Returns the smallest value of an expression
    over a group of rows.

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

        df = vDataFrame({"x": [2, -11, 7, 12]})
        # apply the min function
        df.select([str(vpf.min(df["x"]))])

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [2, -11, 7, 12]})
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_analytic_min.html", "w")
        html_file.write(df.select([str(vpf.min(df["x"]))])._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_analytic_min.html
    """
    expr = format_magic(expr)
    return StringSQL(f"MIN({expr})", "float")


def nth_value(expr: SQLExpression, row_number: int) -> StringSQL:
    """
    Returns the value evaluated at the row that is
    the nth row of the window (counting from 1).

    Parameters
    ----------
    expr: SQLExpression
        Expression.
    row_number: int
        Specifies the row to evaluate.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [1, 2, 3, 4],
                          "y": [11.4, -2.5, 3.5, -4.2]})
        # apply the nth_value function
        df["nth_value"] = vpf.nth_value(df["y"], 3)._over(order_by = [df["x"]])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [1, 2, 3, 4],
                          "y": [11.4, -2.5, 3.5, -4.2]})
        df["nth_value"] = vpf.nth_value(df["y"], 3)._over(order_by = [df["x"]])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_analytic_nth_value.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_analytic_nth_value.html
    """
    expr = format_magic(expr)
    return StringSQL(f"NTH_VALUE({expr}, {row_number})", "int")


def quantile(expr: SQLExpression, number: float) -> StringSQL:
    """
    Computes  the  approximate  percentile of  an
    expression over a group of rows.

    Parameters
    ----------
    expr: SQLExpression
        Expression.
    number: float
        Percentile value,  which must be  a FLOAT
        constant ranging from 0 to 1 (inclusive).

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [2, -11, 7, 12]})
        # apply the quantile function
        df.select([str(vpf.quantile(df["x"], 0.25))])

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [2, -11, 7, 12]})
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_analytic_quantile.html", "w")
        html_file.write(df.select([str(vpf.quantile(df["x"], 0.25))])._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_analytic_quantile.html
    """
    expr = format_magic(expr)
    return StringSQL(
        f"APPROXIMATE_PERCENTILE({expr} USING PARAMETERS percentile = {number})",
        "float",
    )


def rank() -> StringSQL:
    """
    Within each window partition, ranks all rows in
    the  query  results set according to the  order
    specified  by  the  window's  ORDER BY  clause.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [1, -10, 1000, 7, 7]})
        # apply the rank function
        df["rank"] = vpf.rank()._over(order_by = [df["x"]])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [1, -10, 1000, 7, 7]})
        df["rank"] = vpf.rank()._over(order_by = [df["x"]])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_analytic_rank.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_analytic_rank.html
    """
    return StringSQL("RANK()", "int")


def row_number() -> StringSQL:
    """
    Assigns a sequence of unique numbers, starting
    from 1,  to  each  row in a  window partition.

    Returns
    -------
    StringSQL
        SQL string.

    Examples
    --------
    .. code-block:: python

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf

        df = vDataFrame({"x": [1, -10, 1000, 7, 7]})
        # apply the row_number function
        df["row_number"] = vpf.row_number()._over(order_by = [df["x"]])
        display(df)

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [1, -10, 1000, 7, 7]})
        df["row_number"] = vpf.row_number()._over(order_by = [df["x"]])
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_analytic_row_number.html", "w")
        html_file.write(df._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_analytic_row_number.html
    """
    return StringSQL("ROW_NUMBER()", "int")


def std(expr: SQLExpression) -> StringSQL:
    """
    Evaluates  the  statistical  sample  standard
    deviation  for  each  member  of  the  group.

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

        df = vDataFrame({"x": [2, -11, 7, 12]})
        # apply the std function
        df.select([str(vpf.std(df["x"]))])

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [2, -11, 7, 12]})
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_analytic_std.html", "w")
        html_file.write(df.select([str(vpf.std(df["x"]))])._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_analytic_std.html
    """
    expr = format_magic(expr)
    return StringSQL(f"STDDEV({expr})", "float")


stddev = std


def sum(expr: SQLExpression) -> StringSQL:
    """
    Computes the sum of an expression over a group
    of rows.

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

        df = vDataFrame({"x": [2, -11, 7, 12]})
        # apply the sum function
        df.select([str(vpf.sum(df["x"]))])

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [2, -11, 7, 12]})
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_analytic_sum.html", "w")
        html_file.write(df.select([str(vpf.sum(df["x"]))])._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_analytic_sum.html
    """
    expr = format_magic(expr)
    return StringSQL(f"SUM({expr})", "float")


def var(expr: SQLExpression) -> StringSQL:
    """
    Evaluates the sample variance for each row of
    the group.

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

        df = vDataFrame({"x": [2, -11, 7, 12]})
        # apply the var function
        df.select([str(vpf.var(df["x"]))])

    .. ipython:: python
        :suppress:

        from verticapy import vDataFrame
        import verticapy.sql.functions as vpf
        df = vDataFrame({"x": [2, -11, 7, 12]})
        html_file = open("SPHINX_DIRECTORY/figures/sql_functions_analytic_var.html", "w")
        html_file.write(df.select([str(vpf.var(df["x"]))])._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/sql_functions_analytic_var.html
    """
    expr = format_magic(expr)
    return StringSQL(f"VARIANCE({expr})", "float")


variance = var
