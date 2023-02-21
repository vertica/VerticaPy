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
from typing import Iterable

from verticapy._utils._sql._format import format_magic
from verticapy.errors import ParameterError


class str_sql:
    def __init__(self, alias, category="", init_transf=""):
        self._ALIAS = alias
        self.category_ = category
        if not (init_transf):
            self._INIT_TRANSF = alias
        else:
            self._INIT_TRANSF = init_transf

    def __repr__(self):
        return str(self._INIT_TRANSF)

    def __str__(self):
        return str(self._INIT_TRANSF)

    def __abs__(self):
        return str_sql(f"ABS({self._INIT_TRANSF})", self.category())

    def __add__(self, x):
        from verticapy.core.vdataframe.base import vDataColumn

        if (isinstance(self, vDataColumn) and self.isarray()) and (
            isinstance(x, vDataColumn) and x.isarray()
        ):
            return str_sql(
                f"ARRAY_CAT({self._INIT_TRANSF}, {x._INIT_TRANSF})", "complex",
            )
        val = format_magic(x)
        op = (
            "||" if self.category() == "text" and isinstance(x, (str, str_sql)) else "+"
        )
        return str_sql(f"({self._INIT_TRANSF}) {op} ({val})", self.category())

    def __radd__(self, x):
        from verticapy.core.vdataframe.base import vDataColumn

        if (isinstance(self, vDataColumn) and self.isarray()) and (
            isinstance(x, vDataColumn) and x.isarray()
        ):
            return str_sql(
                f"ARRAY_CAT({x._INIT_TRANSF}, {self._INIT_TRANSF})", "complex",
            )
        val = format_magic(x)
        op = (
            "||" if self.category() == "text" and isinstance(x, (str, str_sql)) else "+"
        )
        return str_sql(f"({val}) {op} ({self._INIT_TRANSF})", self.category())

    def __and__(self, x):
        val = format_magic(x)
        return str_sql(f"({self._INIT_TRANSF}) AND ({val})", self.category())

    def __rand__(self, x):
        val = format_magic(x)
        return str_sql(f"({val}) AND ({self._INIT_TRANSF})", self.category())

    def _between(self, x, y):
        val1 = str(format_magic(x))
        val2 = str(format_magic(y))
        return str_sql(
            f"({self._INIT_TRANSF}) BETWEEN ({val1}) AND ({val2})", self.category(),
        )

    def _in(self, *argv):
        if (len(argv) == 1) and (isinstance(argv[0], list)):
            x = argv[0]
        elif len(argv) == 0:
            ParameterError("Method 'in_' doesn't work with no parameters.")
        else:
            x = [elem for elem in argv]
        assert isinstance(x, Iterable) and not (
            isinstance(x, str)
        ), f"Method '_in' only works on iterable elements other than str. Found {x}."
        val = [str(format_magic(elem)) for elem in x]
        val = ", ".join(val)
        return str_sql(f"({self._INIT_TRANSF}) IN ({val})", self.category())

    def _not_in(self, *argv):
        if (len(argv) == 1) and (isinstance(argv[0], list)):
            x = argv[0]
        elif len(argv) == 0:
            ParameterError("Method '_not_in' doesn't work with no parameters.")
        else:
            x = [elem for elem in argv]
        if not (isinstance(x, Iterable)) or (isinstance(x, str)):
            raise TypeError(
                "Method '_not_in' only works on iterable "
                f"elements other than str. Found {x}."
            )
        val = [str(format_magic(elem)) for elem in x]
        val = ", ".join(val)
        return str_sql(f"({self._INIT_TRANSF}) NOT IN ({val})", self.category())

    def _as(self, x):
        return str_sql(f"({self._INIT_TRANSF}) AS {x}", self.category())

    def _distinct(self):
        return str_sql(f"DISTINCT ({self._INIT_TRANSF})", self.category())

    def _over(self, by: (str, list) = [], order_by: (str, list) = []):
        if isinstance(by, str):
            by = [by]
        if isinstance(order_by, str):
            order_by = [order_by]
        by = ", ".join([str(elem) for elem in by])
        if by:
            by = f"PARTITION BY {by}"
        order_by = ", ".join([str(elem) for elem in order_by])
        if order_by:
            order_by = f"ORDER BY {order_by}"
        return str_sql(f"{self._INIT_TRANSF} OVER ({by} {order_by})", self.category(),)

    def __eq__(self, x):
        op = "IS" if (x == None) and not (isinstance(x, str_sql)) else "="
        val = format_magic(x)
        if val != "NULL":
            val = f"({val})"
        return str_sql(f"({self._INIT_TRANSF}) {op} {val}", self.category())

    def __ne__(self, x):
        op = "IS NOT" if (x == None) and not (isinstance(x, str_sql)) else "!="
        val = format_magic(x)
        if val != "NULL":
            val = f"({val})"
        return str_sql(f"({self._INIT_TRANSF}) {op} {val}", self.category())

    def __ge__(self, x):
        val = format_magic(x)
        return str_sql(f"({self._INIT_TRANSF}) >= ({val})", self.category())

    def __gt__(self, x):
        val = format_magic(x)
        return str_sql(f"({self._INIT_TRANSF}) > ({val})", self.category())

    def __le__(self, x):
        val = format_magic(x)
        return str_sql(f"({self._INIT_TRANSF}) <= ({val})", self.category())

    def __lt__(self, x):
        val = format_magic(x)
        return str_sql(f"({self._INIT_TRANSF}) < ({val})", self.category())

    def __mul__(self, x):
        if self.category() == "text" and isinstance(x, int):
            return str_sql(f"REPEAT({self._INIT_TRANSF}, {x})", self.category())
        val = format_magic(x)
        return str_sql(f"({self._INIT_TRANSF}) * ({val})", self.category())

    def __rmul__(self, x):
        if self.category() == "text" and isinstance(x, int):
            return str_sql(f"REPEAT({self._INIT_TRANSF}, {x})", self.category())
        val = format_magic(x)
        return str_sql(f"({val}) * ({self._INIT_TRANSF})", self.category())

    def __or__(self, x):
        val = format_magic(x)
        return str_sql(f"({self._INIT_TRANSF}) OR ({val})", self.category())

    def __ror__(self, x):
        val = format_magic(x)
        return str_sql(f"({val}) OR ({self._INIT_TRANSF})", self.category())

    def __pos__(self):
        return str_sql(f"+({self._INIT_TRANSF})", self.category())

    def __neg__(self):
        return str_sql(f"-({self._INIT_TRANSF})", self.category())

    def __pow__(self, x):
        val = format_magic(x)
        return str_sql(f"POWER({self._INIT_TRANSF}, {val})", self.category())

    def __rpow__(self, x):
        val = format_magic(x)
        return str_sql(f"POWER({val}, {self._INIT_TRANSF})", self.category())

    def __mod__(self, x):
        val = format_magic(x)
        return str_sql(f"MOD({self._INIT_TRANSF}, {val})", self.category())

    def __rmod__(self, x):
        val = format_magic(x)
        return str_sql(f"MOD({val}, {self._INIT_TRANSF})", self.category())

    def __sub__(self, x):
        val = format_magic(x)
        return str_sql(f"({self._INIT_TRANSF}) - ({val})", self.category())

    def __rsub__(self, x):
        val = format_magic(x)
        return str_sql(f"({val}) - ({self._INIT_TRANSF})", self.category())

    def __truediv__(self, x):
        val = format_magic(x)
        return str_sql(f"({self._INIT_TRANSF}) / ({val})", self.category())

    def __rtruediv__(self, x):
        val = format_magic(x)
        return str_sql(f"({val}) / ({self._INIT_TRANSF})", self.category())

    def __floordiv__(self, x):
        val = format_magic(x)
        return str_sql(f"({self._INIT_TRANSF}) // ({val})", self.category())

    def __rfloordiv__(self, x):
        val = format_magic(x)
        return str_sql(f"({val}) // ({self._INIT_TRANSF})", self.category())

    def __ceil__(self):
        return str_sql(f"CEIL({self._INIT_TRANSF})", self.category())

    def __floor__(self):
        return str_sql(f"FLOOR({self._INIT_TRANSF})", self.category())

    def __trunc__(self):
        return str_sql(f"TRUNC({self._INIT_TRANSF})", self.category())

    def __invert__(self):
        return str_sql(f"-({self._INIT_TRANSF}) - 1", self.category())

    def __round__(self, x):
        return str_sql(f"ROUND({self._INIT_TRANSF}, {x})", self.category())

    def category(self):
        return self.category_
