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
    Class used to represent SQL strings.
    """

    @property
    def object_type(self) -> Literal["StringSQL"]:
        return "StringSQL"

    def __init__(
        self, alias, category: Optional[str] = None, init_transf: Optional[str] = None
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
