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

        Examples
        ---------

        We import :py:mod:`verticapy`:

        .. ipython:: python

            import verticapy as vp

        .. hint::

            By assigning an alias to :py:mod:`verticapy`, we mitigate the risk
            of code collisions with other libraries. This precaution is
            necessary because verticapy uses commonly known function names
            like "average" and "median", which can potentially lead to naming
            conflicts. The use of an alias ensures that the functions from
            verticapy are used as intended without interfering with functions
            from other libraries.

        For this example, we will use the Titanic dataset.

        .. code-block:: python

            import verticapy.datasets as vpd

            data = vpd.load_titanic()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/datasets_loaders_load_titanic.html

        .. note::

            VerticaPy offers a wide range of sample datasets that are
            ideal for training and testing purposes. You can explore
            the full list of available datasets in the :ref:`api.datasets`,
            which provides detailed information on each dataset
            and how to use them effectively. These datasets are invaluable
            resources for honing your data analysis and machine learning
            skills within the VerticaPy environment.

        .. ipython:: python
            :suppress:

            import verticapy.datasets as vpd

            data = vpd.load_titanic()

        Let's create a new feature named "family_size".

        .. code-block:: python

            data.eval(
                name = "family_size",
                expr = "parch + sibsp + 1",
            )

        .. ipython:: python
            :suppress:

            res = data.eval(
                name = "family_size",
                expr = "parch + sibsp + 1",
            )
            html_file = open("figures/core_vDataFrame_eval1.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_eval1.html

        .. note::
            You can observe that a new feature "family_size" is added
            to the vDataFrame.

        .. note::

            You can also create a feature in a Pandas-like way by assigning
            a result to a vDataColumn. For example, similar to the above,
            the ``eval`` operation can be expressed as:

            .. code-block:: python

                data["family_size"] = data["parch"] + data["sibsp"] + 1

            Or:

            .. code-block:: python

                data["family_size"] = "parch + sibsp + 1"

        Let's use custom SQL code evaluation to create a new feature
        named "has_life_boat".

        .. code-block:: python

            data.eval(
                name = "has_life_boat",
                expr = "CASE WHEN boat IS NULL THEN 0 ELSE 1 END",
            )

        .. ipython:: python
            :suppress:

            res = data.eval(
                name = "has_life_boat",
                expr = "CASE WHEN boat IS NULL THEN 0 ELSE 1 END",
            )
            html_file = open("figures/core_vDataFrame_eval2.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_eval2.html

        .. note::

            You can also create a feature in a Pandas-like way by assigning
            a result to a vDataColumn. For example, similar to the above,
            the ``eval`` operation can be expressed as:

            .. code-block:: python

                data["has_life_boat"] = "CASE WHEN boat IS NULL THEN 0 ELSE 1 END"

            Or:

            .. code-block:: python

                from verticapy.sql.functions import case_when

                data["has_life_boat"] = case_when(data["boat"] == None, 0, 1)

        .. note::

            You can observe that a new feature "has_life_boat" is added
            to the vDataFrame.

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.analytic` : Advanced analytical function.
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

        category = "undefined"
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
