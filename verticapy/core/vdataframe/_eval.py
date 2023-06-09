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
import re
from typing import Any, Union, TYPE_CHECKING

from vertica_python.errors import QueryError

from verticapy._utils._object import create_new_vdc
from verticapy._utils._sql._cast import to_category
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import quote_ident
from verticapy.errors import QueryError as vQueryError

from verticapy.core.string_sql.base import StringSQL

from verticapy.core.vdataframe._io import vDFInOut
from verticapy.core.vdataframe._sys import vDCSystem

from verticapy.sql.dtypes import get_data_types
from verticapy.sql.flex import isvmap

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame


class vDFEval(vDFInOut):
    def __setattr__(self, attr: str, val: Any) -> None:
        obj_type = None
        if hasattr(val, "object_type"):
            obj_type = val.object_type

        if isinstance(val, (str, StringSQL, int, float)) and obj_type != "vDataColumn":
            val = str(val)
            if self.is_colname_in(attr):
                self[attr].apply(func=val)
            else:
                self.eval(name=attr, expr=val)
        elif obj_type == "vDataColumn" and not val._init:
            final_trans, n = val._init_transf, len(val._transf)
            for i in range(1, n):
                final_trans = val._transf[i][0].replace("{}", final_trans)
            self.eval(name=attr, expr=final_trans)
        else:
            self.__dict__[attr] = val

    def __setitem__(self, index: str, val: Any) -> None:
        setattr(self, index, val)

    @save_verticapy_logs
    def eval(self, name: str, expr: Union[str, StringSQL]) -> "vDataFrame":
        """
        Evaluates a customized expression.

        Parameters
        ----------
        name: str
            Name of the new vDataColumn.
        expr: str
            Expression  in pure SQL used to compute the new
            feature.
            For example:
            'CASE WHEN "column" > 3 THEN 2 ELSE NULL END' and
            'POWER("column", 2)' will work.

        Returns
        -------
        vDataFrame
            self
        """
        if isinstance(expr, StringSQL):
            expr = str(expr)
        name = quote_ident(name.replace('"', "_"))
        if self.is_colname_in(name):
            raise NameError(
                f"A vDataColumn has already the alias {name}.\n"
                "By changing the parameter 'name', you'll "
                "be able to solve this issue."
            )
        try:
            query = f"SELECT {expr} AS {name} FROM {self} LIMIT 0"
            ctype = get_data_types(
                query,
                name[1:-1].replace("'", "''"),
            )
        except QueryError:
            raise vQueryError(
                f"The expression '{expr}' seems to be incorrect.\nBy "
                "turning on the SQL with the 'set_option' function, "
                "you'll print the SQL code generation and probably "
                "see why the evaluation didn't work."
            )
        if not ctype:
            ctype = "undefined"
        elif (ctype.lower().startswith(("long varbina", "long varchar"))) and (
            self._vars["isflex"]
            or isvmap(
                expr=f"({query}) VERTICAPY_SUBTABLE",
                column=name,
            )
        ):
            category = "vmap"
            ctype = "VMAP(" + "(".join(ctype.split("(")[1:]) if "(" in ctype else "VMAP"
        else:
            category = to_category(ctype=ctype)
        all_cols, max_floor = self.get_columns(), 0
        for column in all_cols:
            column_str = column.replace('"', "")
            if (quote_ident(column) in expr) or (
                re.search(re.compile(f"\\b{column_str}\\b"), expr)
            ):
                max_floor = max(len(self[column]._transf), max_floor)
        transformations = [
            (
                "___VERTICAPY_UNDEFINED___",
                "___VERTICAPY_UNDEFINED___",
                "___VERTICAPY_UNDEFINED___",
            )
            for i in range(max_floor)
        ] + [(expr, ctype, category)]
        new_vDataColumn = create_new_vdc(
            name, parent=self, transformations=transformations
        )
        setattr(self, name, new_vDataColumn)
        setattr(self, name.replace('"', ""), new_vDataColumn)
        new_vDataColumn._init = False
        new_vDataColumn._init_transf = name
        self._vars["columns"] += [name]
        self._add_to_history(
            f"[Eval]: A new vDataColumn {name} was added to the vDataFrame."
        )
        return self


class vDCEval(vDCSystem):
    def __setattr__(self, attr: str, val: Any) -> None:
        self.__dict__[attr] = val
