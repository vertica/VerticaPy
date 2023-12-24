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
import copy
from typing import Any, Iterable, Literal, Optional

from verticapy._typing import NoneType, SQLColumns
from verticapy._utils._sql._format import format_magic, format_type


class StringSQL:
    """
    This class is utilized to represent SQL strings, with
    :py:class:`~vDataColumn`, for instance, inheriting from
    this class. Its purpose is to streamline SQL operations
    in Python, enabling the use of Python operators with
    specific SQL strings to simplify operations and enhance
    the usability of the generated SQL.

    Parameters
    ----------
    alias: str
        Name of the :py:class:`~StringSQL`.
    category: str, optional
        Category of the :py:class:`~StringSQL`. This parameter
        is crucial for performing accurate operations. For
        instance, it plays a significant role in distinguishing
        between the treatment of floats and text data.
    init_transf: str, optional
        Initial Transformation. It is employed to streamline
        certain operations on :py:class:`~vDataColumn`.

    Attributes
    ----------
    No relevant attributes present.

    Examples
    --------
    We import :py:mod:`verticapy`:

    .. ipython:: python

        import verticapy as vp

    .. hint::

        By assigning an alias to :py:mod:`verticapy`,
        we mitigate the risk of code collisions with
        other libraries. This precaution is necessary
        because verticapy uses commonly known function
        names like "average" and "median", which can
        potentially lead to naming conflicts. The use
        of an alias ensures that the functions from
        :py:mod:`verticapy` are used as intended without
        interfering with functions from other libraries.

    Let's create various :py:class:`~StringSQL` objects.

    .. ipython:: python

        num1 = vp.StringSQL('numerical_col1', 'float')
        num2 = vp.StringSQL('numerical_col2', 'int')
        num3 = vp.StringSQL('numerical_col3', 'int')
        str1 = vp.StringSQL('varchar_col1', 'text')
        str2 = vp.StringSQL('varchar_col2', 'text')
        bool1 = vp.StringSQL('bool_col1', 'bool')
        bool2 = vp.StringSQL('bool_col2', 'bool')

    The :py:class:`~StringSQL` representation is a
    straightforward string.

    .. ipython:: python

        display(num1)
        display(str1)
        display(bool1)

    Exploring Mathematical Operators
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    All mathematical operators are supported
    for numerical :py:class:`~StringSQL`.

    .. ipython:: python

        import math

        # Mathematical Functions
        abs(num1)
        round(num1, 3)
        math.floor(num1)
        math.ceil(num1)

        # Unary Operators
        - num1
        + num1
        ~ num1

        # Binary Operators
        num1 == num2
        num1 != num2
        num1 + num2
        num1 - num2
        num1 / num2
        num1 // num2
        num1 * num2
        num1 ** num2
        num2 % num3
        num1 > num2
        num1 >= num2
        num1 < num2
        num1 <= num2

        # Extension
        num1._between(num2, num3)

    .. note::

        Most mathematical operators can be applied
        to the date datatype.

    Exploring String Operators
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    A lot of operators are supported
    for text data-type :py:class:`~StringSQL`.

    .. ipython:: python

        # Equality and Inequality
        str1 == str2
        str1 != str2

        # Concatenating two strings
        str1 + str2

        # Repeating a string
        str1 * 3

    Exploring Boolean Operators
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    A lot of operators are supported
    for boolean data-type :py:class:`~StringSQL`.

    .. ipython:: python

        # Equality and Inequality
        bool1 == bool2
        bool1 != bool2

        # AND
        bool1 & bool2

        # OR
        bool1 | bool2

    .. important::

        The '&' and '|' operators in :py:class:`~StringSQL`
        are distinct from the Python 'and' and 'or'
        operators.

        In Python, 'and' and 'or' are logical operators
        used for boolean expressions. They perform short
        -circuit evaluation, meaning that the second
        operand is only evaluated if necessary.

        In :py:class:`~StringSQL`, '&' and '|' are used for
        boolean concatenation, not logical operations.
        They combine two :py:class:`~StringSQL` without
        short-circuiting, meaning both sides of the
        operator are always evaluated.

    General Operators
    ^^^^^^^^^^^^^^^^^^

    Some general operators are available for
    all types.

    .. ipython:: python

        # IN
        str1._in(['A', 'B', 'C'])

        # NOT IN
        str1._not_in(['A', 'B', 'C'])

        # DISTINCT
        str1._distinct()

        # ALIAS
        str1._as('new_name')

    .. note::

        The result is a :py:class:`~StringSQL` object
        that can be reused iteratively until we obtain
        the final SQL statement.

    Using VerticaPy SQL Functions
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Numerous SQL functions are accessible in the
    :mod:`verticapy.sql.functions` module. In this
    example, we will utilize the ``min`` and ``max``
    aggregations to normalize the 'num1' column.
    We will utilize a partition by 'str1' to normalize
    within this specific partition.

    .. ipython:: python

        import verticapy.sql.functions as vpf
        (num1 - vpf.min(num1)._over([str1])) / (vpf.max(num1)._over([str1]) - vpf.min(num1)._over([str1]))

    .. note::

        The ``_over`` operator also includes an
        ``order_by`` parameter for sorting using
        a specific column.

    Combining the different Operators
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    It is possible to combine as many operators as
    desired, as long as they adhere to SQL logic.
    This allows the building of SQL expressions
    using a Pythonic structure.

    .. ipython:: python

        # Example of a numerical expression
        round(abs((num1 ** (num2 + num3)) - 15), 2)

        # Example of a string expression
        ((str1 + str2) ** 2)._in(['ABAB', 'ACAC'])

    .. note::

        It is recommended to use these operators to
        construct VerticaPy code, as they adapt
        seamlessly to the corresponding SQL database.
        These functionalities are easily maintainable
        and extendable.

    Examples Using vDataFrame
    ^^^^^^^^^^^^^^^^^^^^^^^^^^

    :py:class:`~vDataColumn` inherit from
    :py:class:`~StringSQL`, enabling operations
    on :py:class:`~vDataFrame` in a pandas-like
    manner.

    .. ipython:: python

        import verticapy as vp

        vdf = vp.vDataFrame(
            {
                "num1": [10, 20, 30, 40],
                "num2": [5, 10, 15, 20],
            },
        )
        vdf["num3"] = abs(vdf["num1"] * 2 - 5 * vdf["num2"] / 2)

    Let's display the result. It is noteworthy that
    the generated SQL was applied and utilized by
    the :py:class:`~vDataFrame`.

    .. ipython:: python
        :suppress:

        result = vdf
        html_file = open("SPHINX_DIRECTORY/figures/core_string_sql_1.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/core_string_sql_1.html

    For more examples, refer to the
    :py:class:`~vDataFrame` class.
    """

    @property
    def object_type(self) -> Literal["StringSQL"]:
        return "StringSQL"

    def __init__(
        self,
        alias: str,
        category: Optional[str] = None,
        init_transf: Optional[str] = None,
    ) -> None:
        self._alias = alias
        self._category = category
        if not init_transf:
            self._init_transf = alias
        else:
            self._init_transf = init_transf

    def __repr__(self) -> "StringSQL":
        return str(self._init_transf)

    def __str__(self) -> "StringSQL":
        return str(self._init_transf)

    def __abs__(self) -> "StringSQL":
        return StringSQL(f"ABS({self._init_transf})", self.category())

    def __add__(self, x: Any) -> "StringSQL":
        if (hasattr(self, "isarray") and self.isarray()) and (
            hasattr(x, "isarray") and x.isarray()
        ):
            return StringSQL(
                f"ARRAY_CAT({self._init_transf}, {x._init_transf})",
                "complex",
            )
        val = format_magic(x)
        op = (
            "||"
            if self.category() == "text" and isinstance(x, (str, StringSQL))
            else "+"
        )
        return StringSQL(f"({self._init_transf}) {op} ({val})", self.category())

    def __radd__(self, x: Any) -> "StringSQL":
        if (hasattr(self, "isarray") and self.isarray()) and (
            hasattr(x, "isarray") and x.isarray()
        ):
            return StringSQL(
                f"ARRAY_CAT({x._init_transf}, {self._init_transf})",
                "complex",
            )
        val = format_magic(x)
        op = (
            "||"
            if self.category() == "text" and isinstance(x, (str, StringSQL))
            else "+"
        )
        return StringSQL(f"({val}) {op} ({self._init_transf})", self.category())

    def __and__(self, x: Any) -> "StringSQL":
        val = format_magic(x)
        return StringSQL(f"({self._init_transf}) AND ({val})", self.category())

    def __rand__(self, x: Any) -> "StringSQL":
        val = format_magic(x)
        return StringSQL(f"({val}) AND ({self._init_transf})", self.category())

    def _between(self, x: Any, y: Any) -> "StringSQL":
        val1 = str(format_magic(x))
        val2 = str(format_magic(y))
        return StringSQL(
            f"({self._init_transf}) BETWEEN ({val1}) AND ({val2})",
            self.category(),
        )

    def _in(self, *args) -> "StringSQL":
        if (len(args) == 1) and (isinstance(args[0], list)):
            x = args[0]
        elif len(args) == 0:
            ValueError("Method 'in_' doesn't work with no parameters.")
        else:
            x = copy.deepcopy(args)
        assert isinstance(x, Iterable) and not (
            isinstance(x, str)
        ), f"Method '_in' only works on iterable elements other than str. Found {x}."
        val = [str(format_magic(elem)) for elem in x]
        val = ", ".join(val)
        return StringSQL(f"({self._init_transf}) IN ({val})", self.category())

    def _not_in(self, *args) -> "StringSQL":
        if (len(args) == 1) and (isinstance(args[0], list)):
            x = args[0]
        elif len(args) == 0:
            ValueError("Method '_not_in' doesn't work with no parameters.")
        else:
            x = copy.deepcopy(args)
        if not isinstance(x, Iterable) or (isinstance(x, str)):
            raise TypeError(
                "Method '_not_in' only works on iterable "
                f"elements other than str. Found {x}."
            )
        val = [str(format_magic(elem)) for elem in x]
        val = ", ".join(val)
        return StringSQL(f"({self._init_transf}) NOT IN ({val})", self.category())

    def _as(self, x: Any) -> "StringSQL":
        return StringSQL(f"({self._init_transf}) AS {x}", self.category())

    def _distinct(self) -> "StringSQL":
        return StringSQL(f"DISTINCT ({self._init_transf})", self.category())

    def _over(
        self, by: Optional[SQLColumns] = None, order_by: Optional[SQLColumns] = None
    ) -> "StringSQL":
        by, order_by = format_type(by, order_by, dtype=list)
        by = ", ".join([str(elem) for elem in by])
        if by:
            by = f"PARTITION BY {by}"
        order_by = ", ".join([str(elem) for elem in order_by])
        if order_by:
            order_by = f"ORDER BY {order_by}"
        return StringSQL(
            f"{self._init_transf} OVER ({by} {order_by})",
            self.category(),
        )

    def __eq__(self, x: Any) -> "StringSQL":
        op = "IS" if isinstance(x, NoneType) and not isinstance(x, StringSQL) else "="
        val = format_magic(x)
        if val != "NULL":
            val = f"({val})"
        return StringSQL(f"({self._init_transf}) {op} {val}", self.category())

    def __ne__(self, x: Any) -> "StringSQL":
        op = (
            "IS NOT"
            if isinstance(x, NoneType) and not isinstance(x, StringSQL)
            else "!="
        )
        val = format_magic(x)
        if val != "NULL":
            val = f"({val})"
        return StringSQL(f"({self._init_transf}) {op} {val}", self.category())

    def __ge__(self, x: Any) -> "StringSQL":
        val = format_magic(x)
        return StringSQL(f"({self._init_transf}) >= ({val})", self.category())

    def __gt__(self, x: Any) -> "StringSQL":
        val = format_magic(x)
        return StringSQL(f"({self._init_transf}) > ({val})", self.category())

    def __le__(self, x: Any) -> "StringSQL":
        val = format_magic(x)
        return StringSQL(f"({self._init_transf}) <= ({val})", self.category())

    def __lt__(self, x: Any) -> "StringSQL":
        val = format_magic(x)
        return StringSQL(f"({self._init_transf}) < ({val})", self.category())

    def __mul__(self, x: Any) -> "StringSQL":
        if self.category() == "text" and isinstance(x, int):
            return StringSQL(f"REPEAT({self._init_transf}, {x})", self.category())
        val = format_magic(x)
        return StringSQL(f"({self._init_transf}) * ({val})", self.category())

    def __rmul__(self, x: Any) -> "StringSQL":
        if self.category() == "text" and isinstance(x, int):
            return StringSQL(f"REPEAT({self._init_transf}, {x})", self.category())
        val = format_magic(x)
        return StringSQL(f"({val}) * ({self._init_transf})", self.category())

    def __or__(self, x: Any) -> "StringSQL":
        val = format_magic(x)
        return StringSQL(f"({self._init_transf}) OR ({val})", self.category())

    def __ror__(self, x: Any) -> "StringSQL":
        val = format_magic(x)
        return StringSQL(f"({val}) OR ({self._init_transf})", self.category())

    def __pos__(self) -> "StringSQL":
        return StringSQL(f"+({self._init_transf})", self.category())

    def __neg__(self) -> "StringSQL":
        return StringSQL(f"-({self._init_transf})", self.category())

    def __pow__(self, x: Any) -> "StringSQL":
        val = format_magic(x)
        return StringSQL(f"POWER({self._init_transf}, {val})", self.category())

    def __rpow__(self, x: Any) -> "StringSQL":
        val = format_magic(x)
        return StringSQL(f"POWER({val}, {self._init_transf})", self.category())

    def __mod__(self, x: Any) -> "StringSQL":
        val = format_magic(x)
        return StringSQL(f"MOD({self._init_transf}, {val})", self.category())

    def __rmod__(self, x: Any) -> "StringSQL":
        val = format_magic(x)
        return StringSQL(f"MOD({val}, {self._init_transf})", self.category())

    def __sub__(self, x: Any) -> "StringSQL":
        val = format_magic(x)
        return StringSQL(f"({self._init_transf}) - ({val})", self.category())

    def __rsub__(self, x: Any) -> "StringSQL":
        val = format_magic(x)
        return StringSQL(f"({val}) - ({self._init_transf})", self.category())

    def __truediv__(self, x: Any) -> "StringSQL":
        val = format_magic(x)
        return StringSQL(f"({self._init_transf}) / ({val})", self.category())

    def __rtruediv__(self, x: Any) -> "StringSQL":
        val = format_magic(x)
        return StringSQL(f"({val}) / ({self._init_transf})", self.category())

    def __floordiv__(self, x: Any) -> "StringSQL":
        val = format_magic(x)
        return StringSQL(f"({self._init_transf}) // ({val})", self.category())

    def __rfloordiv__(self, x: Any) -> "StringSQL":
        val = format_magic(x)
        return StringSQL(f"({val}) // ({self._init_transf})", self.category())

    def __ceil__(self) -> "StringSQL":
        return StringSQL(f"CEIL({self._init_transf})", self.category())

    def __floor__(self) -> "StringSQL":
        return StringSQL(f"FLOOR({self._init_transf})", self.category())

    def __trunc__(self) -> "StringSQL":
        return StringSQL(f"TRUNC({self._init_transf})", self.category())

    def __invert__(self) -> "StringSQL":
        return StringSQL(f"-({self._init_transf}) - 1", self.category())

    def __round__(self, x) -> "StringSQL":
        return StringSQL(f"ROUND({self._init_transf}, {x})", self.category())

    def category(self) -> str:
        return self._category
