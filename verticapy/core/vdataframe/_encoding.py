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
import math
import warnings
from typing import Literal, Optional, TYPE_CHECKING

import verticapy._config.config as conf
from verticapy._typing import PythonNumber, SQLColumns
from verticapy._utils._gen import gen_tmp_name
from verticapy._utils._object import get_vertica_mllib, create_new_vdc
from verticapy._utils._sql._cast import to_varchar
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import format_type
from verticapy._utils._sql._sys import _executeSQL

from verticapy.core.string_sql.base import StringSQL

from verticapy.core.vdataframe._fill import vDFFill, vDCFill

from verticapy.sql.drop import drop
from verticapy.sql.functions import case_when, decode

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame


class vDFEncode(vDFFill):
    @save_verticapy_logs
    def case_when(self, name: str, *args) -> "vDataFrame":
        """
        Creates a new feature by evaluating on
        provided conditions.

        Parameters
        ----------
        name: str
            Name of the new feature.
        args: object
            Any number of Expressions.
            The expression is generated in the following format:

            - even:
                CASE ... WHEN args[2 * i] THEN args[2 * i + 1] ... END
            - odd :
                CASE ... WHEN args[2 * i] THEN args[2 * i + 1] ... ELSE args[n] END

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

        Let's create a new feature "age_category".

        .. code-block:: python

            data.case_when(
                "age_category",
                data["age"] < 12, "children",
                data["age"] < 18, "teenagers",
                data["age"] > 60, "seniors",
                data["age"] < 25, "young adults",
                "adults"
            )
            data[["age", "age_category"]]

        .. ipython:: python
            :suppress:

            data.case_when(
                "age_category",
                data["age"] < 12, "children",
                data["age"] < 18, "teenagers",
                data["age"] > 60, "seniors",
                data["age"] < 25, "young adults",
                "adults"
            )
            res = data[["age", "age_category"]]
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_casewhen.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_casewhen.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.decode` : User Defined Encoding.
            | :py:meth:`verticapy.vDataFrame.eval` : Evaluates an expression.
        """
        return self.eval(name=name, expr=case_when(*args))

    @save_verticapy_logs
    def one_hot_encode(
        self,
        columns: Optional[SQLColumns] = None,
        max_cardinality: int = 12,
        prefix_sep: str = "_",
        drop_first: bool = True,
        use_numbers_as_suffix: bool = False,
    ) -> "vDataFrame":
        """
        Encodes the vDataColumns  using the One Hot Encoding
        algorithm.

        Parameters
        ----------
        columns: SQLColumns, optional
            List  of the vDataColumns used to train the  One
            Hot Encoding model. If empty, only the vDataColumns
            with  a cardinality  less than 'max_cardinality'
            are used.
        max_cardinality: int, optional
            Cardinality  threshold  used to  determine whether the
            vDataColumn is taken into account during the encoding
            This parameter is used only if the parameter 'columns'
            is empty.
        prefix_sep: str, optional
            Prefix delimitor of the dummies names.
        drop_first: bool, optional
            Drops  the  first  dummy  to  avoid  the  creation  of
            correlated features.
        use_numbers_as_suffix: bool, optional
            Uses  numbers  as suffix instead of  the  vDataColumns
            categories.

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

        Let's select few categorical features

        .. code-block:: python

            data = data.select(["pclass", "sex", "survived", "embarked"])
            data

        .. ipython:: python
            :suppress:

            data = data.select(["pclass", "sex", "survived", "embarked"])
            res = data
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_ohe1.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_ohe1.html

        Let's apply encoding on all the vcolumns of the datasets

        .. code-block:: python

            data.one_hot_encode()

        .. ipython:: python
            :suppress:

            res = data.one_hot_encode()
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_ohe2.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_ohe2.html

        Let's apply encoding on two specific vcolumns viz. "pclass" and "embarked"

        .. code-block:: python

            data = data.select(["pclass", "sex", "survived", "embarked"])
            data.one_hot_encode(columns = ["pclass", "embarked"])

        .. ipython:: python
            :suppress:

            data = data.select(["pclass", "sex","survived", "embarked"])
            res = data.one_hot_encode(columns = ['pclass', 'embarked'])
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_ohe3.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_ohe3.html

        Let's apply encoding on all features having cardinality less than 3

        .. code-block:: python

            data = data.select(["pclass", "sex", "survived", "embarked"])
            data.one_hot_encode(
                max_cardinality = 3,
                drop_first = False,
            )

        .. ipython:: python
            :suppress:
            :okwarning:

            data = data.select(["pclass", "sex", "survived", "embarked"])
            res = data.one_hot_encode(
                max_cardinality = 3,
                drop_first = False,
            )
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_ohe4.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_ohe4.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.decode` : User Defined Encoding.
            | :py:meth:`verticapy.vDataFrame.label_encode` : Label Encoding.
            | :py:meth:`verticapy.vDataFrame.mean_encode` : Mean Encoding.
            | :py:meth:`verticapy.vDataFrame.discretize` : Discretization.
        """
        columns = format_type(columns, dtype=list)
        columns = self.format_colnames(columns)
        if len(columns) == 0:
            columns = self.get_columns()
        cols_hand = True if (columns) else False
        for column in columns:
            if self[column].nunique(True) < max_cardinality:
                self[column].one_hot_encode(
                    "", prefix_sep, drop_first, use_numbers_as_suffix
                )
            elif cols_hand and conf.get_option("print_info"):
                warning_message = (
                    f"The vDataColumn '{column}' was ignored because of "
                    "its high cardinality.\nIncrease the parameter "
                    "'max_cardinality' to solve this issue or use "
                    "directly the vDataColumn one_hot_encode method."
                )
                warnings.warn(warning_message, Warning)
        return self

    get_dummies = one_hot_encode


class vDCEncode(vDCFill):
    @save_verticapy_logs
    def cut(
        self,
        breaks: list,
        labels: Optional[list] = None,
        include_lowest: bool = True,
        right: bool = True,
    ) -> "vDataFrame":
        """
        Discretizes the vDataColumn using the input list.

        Parameters
        ----------
        breaks: list
            List of values used to cut the vDataColumn.
        labels: list, optional
            Labels used  to name the new categories.  If empty,
            names are generated.
        include_lowest: bool, optional
            If  set to  True,  the lowest element of the  list
            is included.
        right: bool, optional
            How the intervals should be closed. If set to True,
            the intervals are closed on the right.

        Returns
        -------
        vDataFrame
            self._parent

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

        Let's look at "age" vcolumn

        .. code-block:: python

            data["age"]

        .. ipython:: python
            :suppress:

            res = data["age"]
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_cut1.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_cut1.html

        Let's look at the distribution of age.

        .. code-block:: python

            data["age"].bar()

        .. ipython:: python
            :suppress:

            vp.set_option("plotting_lib", "plotly")
            res = data["age"].bar()
            res.write_html("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_cut2.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_cut2.html

        Let's discretize "age" using the :py:meth:`verticapy.vDataFrame.cut` method.

        .. code-block:: python

            data["age"].cut([0, 15, 80])
            data["age"]

        .. ipython:: python
            :suppress:

            data["age"].cut([0, 15, 80])
            res = data["age"]
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_cut3.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_cut3.html

        Let's look at the distribution of age again.

        .. code-block:: python

            data["age"].bar()

        .. ipython:: python
            :suppress:

            res = data["age"].bar()
            res.write_html("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_cut4.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_cut4.html

        Let's discretize "fare" using the :py:meth:`verticapy.vDataFrame.cut` method.

        .. code-block:: python

            data["fare"].cut(
                [0, 15, 800],
                right = False,
                include_lowest = False
            )
            data["fare"]

        .. ipython:: python
            :suppress:

            data["fare"].cut(
                [0, 15, 800],
                right = False,
                include_lowest = False
            )
            res = data["fare"]
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_cut5.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_cut5.html

        Let's look at the distribution of fare.

        .. code-block:: python

            data["fare"].bar()

        .. ipython:: python
            :suppress:

            res = data["fare"].bar()
            res.write_html("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_cut6.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_cut6.html

        Let's discretize "parch" using the :py:meth:`verticapy.vDataFrame.cut` method.

        .. code-block:: python

            data["parch"].cut(
                [0, 5, 10],
                right = False,
                include_lowest = False,
                labels = ["small", "big"]
            )
            data["parch"]

        .. ipython:: python
            :suppress:
            :okwarning:

            data["parch"].cut(
                [0, 5, 10],
                right = False,
                include_lowest = False,
                labels = ["small", "big"]
            )
            res = data["parch"]
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_cut7.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_cut7.html

        Let's look at the distribution of parch.

        .. code-block:: python

            data["parch"].bar()

        .. ipython:: python
            :suppress:

            res = data["parch"].bar()
            res.write_html("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_cut8.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_cut8.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.decode` : User Defined Encoding.
            | :py:meth:`verticapy.vDataFrame.label_encode` : Label Encoding.
            | :py:meth:`verticapy.vDataFrame.mean_encode` : Mean Encoding.
            | :py:meth:`verticapy.vDataFrame.one_hot_encode` : One Hot Encoding.
        """
        labels = format_type(labels, dtype=list)
        assert self.isnum() or self.isdate(), TypeError(
            "cut only works on numerical / date-like vDataColumns."
        )
        assert len(breaks) >= 2, ValueError(
            "Length of parameter 'breaks' must be greater or equal to 2."
        )
        assert len(breaks) == len(labels) + 1 or not labels, ValueError(
            "Length of parameter breaks must be equal to the length of parameter "
            "'labels' + 1 or parameter 'labels' must be empty."
        )
        conditions, column = [], self._alias
        for idx in range(len(breaks) - 1):
            first_elem, second_elem = breaks[idx], breaks[idx + 1]
            if right:
                op1, op2, close_l, close_r = "<", "<=", "]", "]"
            else:
                op1, op2, close_l, close_r = "<=", "<", "[", "["
            if idx == 0 and include_lowest:
                op1, close_l = "<=", "["
            elif idx == 0:
                op1, close_l = "<", "]"
            if labels:
                label = labels[idx]
            else:
                label = f"{close_l}{first_elem};{second_elem}{close_r}"
            conditions += [
                f"'{first_elem}' {op1} {column} AND {column} {op2} '{second_elem}' THEN '{label}'"
            ]
        expr = "CASE WHEN " + " WHEN ".join(conditions) + " END"
        self.apply(func=expr)

    @save_verticapy_logs
    def decode(self, *args) -> "vDataFrame":
        """
        Encodes the vDataColumn using a user-defined encoding.

        Parameters
        ----------
        args: object
            Any number of expressions.
            The expression is generated in the following format:

            - even:
                CASE ... WHEN vDataColumn = args[2 * i]
                THEN args[2 * i + 1] ... END

            - odd :
                CASE ... WHEN vDataColumn = args[2 * i]
                THEN args[2 * i + 1] ... ELSE args[n] END

        Returns
        -------
        vDataFrame
            self._parent

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

        Let's encode "sex" vcolumn and represent "female" category as 1 and
        "male" category as 0.

        .. code-block:: python

            data["sex"].decode("female", 1, "male", 0, 2)
            data["sex"]

        .. ipython:: python
            :suppress:

            data["sex"].decode("female", 1, "male", 0, 2)
            res = data["sex"]
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_decode.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_decode.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.case_when` : Conditional Statement.
            | :py:meth:`verticapy.vDataFrame.eval` : Evaluates an expression.
            | :py:meth:`verticapy.vDataFrame.label_encode` : Label Encoding.
            | :py:meth:`verticapy.vDataFrame.mean_encode` : Mean Encoding.
        """
        return self.apply(func=decode(StringSQL("{}"), *args))

    @save_verticapy_logs
    def discretize(
        self,
        method: Literal["auto", "smart", "same_width", "same_freq", "topk"] = "auto",
        h: PythonNumber = 0,
        nbins: int = -1,
        k: int = 6,
        new_category: str = "Others",
        RFmodel_params: Optional[dict] = None,
        response: Optional[str] = None,
        return_enum_trans: bool = False,
    ) -> "vDataFrame":
        """
        Discretizes the vDataColumn using the input method.

        Parameters
        ----------
        method: str, optional
            The method used to discretize the vDataColumn:

            - auto:
                Uses method 'same_width' for numerical
                vDataColumns, casts the other types to varchar.
            - same_freq:
                Computes bins  with the same number of elements.
            - same_width:
                Computes regular width bins.
            - smart:
                Uses  the Random  Forest on a  response
                column  to   find   the  most  relevant
                interval to use for the discretization.
            - topk:
                Keeps the topk most frequent categories
                and  merge the  other  into one  unique
                category.
        h: PythonNumber, optional
            The  interval  size  used  to  convert  the vDataColumn.
            If this parameter is equal to 0, an optimised interval is
            computed.
        nbins: int, optional
            Number of bins  used for the discretization  (must be > 1)
        k: int, optional
            The integer k of the 'topk' method.
        new_category: str, optional
            The  name of the  merging  category when using the  'topk'
            method.
        RFmodel_params: dict, optional
            Dictionary  of the  Random Forest  model  parameters used  to
            compute the best splits when 'method' is set to 'smart'.
            A RF Regressor is  trained if  the response is numerical
            (except ints and bools), a RF Classifier otherwise.
            Example: Write {"n_estimators": 20, "max_depth": 10} to train
            a Random Forest with 20 trees and a maximum depth of 10.
        response: str, optional
            Response vDataColumn when method is set to 'smart'.
        return_enum_trans: bool, optional
            Returns  the transformation instead of the vDataFrame parent,
            and does not apply the transformation. This parameter is
            useful for testing the look of the final transformation.

        Returns
        -------
        vDataFrame
            self._parent

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

        Let's look at "age" vcolumn

        .. code-block:: python

            data["age"]

        .. ipython:: python
            :suppress:

            res = data["age"]
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_discretize1.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_discretize1.html

        Let's look at the distribution of age.

        .. code-block:: python

            data["age"].bar()

        .. ipython:: python
            :suppress:

            vp.set_option("plotting_lib", "plotly")
            res = data["age"].bar()
            res.write_html("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_discretize2.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_discretize2.html

        Let's discretize "age" using the same bar width.

        .. code-block:: python

            data["age"].discretize(method = "same_width", h = 10)
            data["age"]

        .. ipython:: python
            :suppress:

            data["age"].discretize(method = "same_width", h = 10)
            res = data["age"]
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_discretize3.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_discretize3.html

        Let's look at the distribution of age again.

        .. code-block:: python

            data["age"].bar()

        .. ipython:: python
            :suppress:

            res = data["age"].bar()
            res.write_html("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_discretize4.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_discretize4.html

        Let's discretize "age" using the same frequency per bin.

        .. code-block:: python

            data = vpd.load_titanic() # Reloading the dataset
            data["age"].discretize(method = "same_freq", nbins = 5)
            data["age"]

        .. ipython:: python
            :suppress:

            data = vpd.load_titanic()
            data["age"].discretize(method = "same_freq", nbins = 5)
            res = data["age"]
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_discretize5.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_discretize5.html

        Let's look at the distribution of age again.

        .. code-block:: python

            data["age"].bar()

        .. ipython:: python
            :suppress:

            res = data["age"].bar()
            res.write_html("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_discretize6.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_discretize6.html

        Let's discretize "age" using a response column distribution.

        .. note::

            While discretizing using a response column distribution,
            a Random Forest Model will be created.

        .. code-block:: python

            data = vpd.load_titanic()
            data["age"].discretize(
                method = "smart",
                response = "survived",
                nbins = 6,
                RFmodel_params = {"n_estimators": 20},
            )
            data["age"].topk()

        .. ipython:: python
            :suppress:
            :okwarning:

            data = vpd.load_titanic()
            data["age"].discretize(
                method = "smart",
                response = "survived",
                nbins = 6,
                RFmodel_params = {"n_estimators": 20},
            )
            res = data["age"].topk()
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_discretize7.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_discretize7.html

        Let's look at the distribution of age again.

        .. code-block:: python

            data["age"].bar()

        .. ipython:: python
            :suppress:

            res = data["age"].bar()
            res.write_html("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_discretize8.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_discretize8.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.decode` : User Defined Encoding.
            | :py:meth:`verticapy.vDataFrame.label_encode` : Label Encoding.
            | :py:meth:`verticapy.vDataFrame.mean_encode` : Mean Encoding.
            | :py:meth:`verticapy.vDataFrame.one_hot_encode` : One Hot Encoding.
        """
        RFmodel_params = format_type(RFmodel_params, dtype=dict)
        vml = get_vertica_mllib()
        if self.isnum() and method == "smart":
            schema = conf.get_option("temp_schema")
            tmp_view_name = gen_tmp_name(schema=schema, name="view")
            assert nbins >= 2, ValueError(
                "Parameter 'nbins' must be greater or equals to 2 in case "
                "of discretization using the method 'smart'."
            )
            assert response, ValueError(
                "Parameter 'response' can not be empty in case of "
                "discretization using the method 'smart'."
            )
            response = self._parent.format_colnames(response)
            drop(tmp_view_name, method="view")
            self._parent.to_db(tmp_view_name)
            if self._parent[response].category() == "float":
                model = vml.RandomForestRegressor()
            else:
                model = vml.RandomForestClassifier()
            model.set_params({"n_estimators": 20, "max_depth": 8, "nbins": 100})
            model.set_params(RFmodel_params)
            parameters = model.get_params()
            try:
                model.fit(
                    tmp_view_name,
                    [self._alias],
                    response,
                    return_report=True,
                )
                query = [
                    f"""
                    (SELECT 
                        READ_TREE(USING PARAMETERS 
                            model_name = '{model.model_name}', 
                            tree_id = {i}, 
                            format = 'tabular'))"""
                    for i in range(parameters["n_estimators"])
                ]
                query = f"""
                    SELECT 
                        /*+LABEL('vDataColumn.discretize')*/ split_value 
                    FROM 
                        (SELECT 
                            split_value, 
                            MAX(weighted_information_gain) 
                        FROM ({' UNION ALL '.join(query)}) VERTICAPY_SUBTABLE 
                        WHERE split_value IS NOT NULL 
                        GROUP BY 1 ORDER BY 2 DESC LIMIT {nbins - 1}) VERTICAPY_SUBTABLE 
                    ORDER BY split_value::float"""
                result = _executeSQL(
                    query=query,
                    title="Computing the optimized histogram nbins using Random Forest.",
                    method="fetchall",
                    sql_push_ext=self._parent._vars["sql_push_ext"],
                    symbol=self._parent._vars["symbol"],
                )
                result = [x[0] for x in result]
            finally:
                drop(tmp_view_name, method="view")
                model.drop()
            result = [self.min()] + result + [self.max()]
        elif method == "topk":
            assert k >= 2, ValueError(
                "Parameter 'k' must be greater or equals to 2 in "
                "case of discretization using the method 'topk'"
            )
            distinct = self.topk(k).values["index"]
            category_str = to_varchar(self.category())
            X_str = ", ".join([f"""'{str(x).replace("'", "''")}'""" for x in distinct])
            new_category_str = new_category.replace("'", "''")
            trans = (
                f"""(CASE 
                        WHEN {category_str} IN ({X_str})
                        THEN {category_str} || '' 
                        ELSE '{new_category_str}' 
                     END)""",
                "varchar",
                "text",
            )
        elif self.isnum() and method == "same_freq":
            assert nbins >= 2, ValueError(
                "Parameter 'nbins' must be greater or equals to 2 in case "
                "of discretization using the method 'same_freq'"
            )
            count = self.count()
            nb = int(float(count / int(nbins)))
            assert nb != 0, Exception(
                "Not enough values to compute the Equal Frequency discretization"
            )
            total, query, nth_elems = nb, [], []
            while total < int(float(count / int(nbins))) * int(nbins):
                nth_elems += [str(total)]
                total += nb
            possibilities = ", ".join(["1"] + nth_elems + [str(count)])
            where = f"WHERE _verticapy_row_nb_ IN ({possibilities})"
            query = f"""
                SELECT /*+LABEL('vDataColumn.discretize')*/ 
                    {self} 
                FROM (SELECT 
                        {self}, 
                        ROW_NUMBER() OVER (ORDER BY {self}) AS _verticapy_row_nb_ 
                      FROM {self._parent} 
                      WHERE {self} IS NOT NULL) VERTICAPY_SUBTABLE {where}"""
            result = _executeSQL(
                query=query,
                title="Computing the equal frequency histogram bins.",
                method="fetchall",
                sql_push_ext=self._parent._vars["sql_push_ext"],
                symbol=self._parent._vars["symbol"],
            )
            result = [elem[0] for elem in result]
        elif self.isnum() and not (self.isbool()) and method in ("same_width", "auto"):
            if not h or h <= 0:
                if nbins <= 0:
                    h = self.numh()
                else:
                    h = (self.max() - self.min()) * 1.01 / nbins
                if h > 0.01:
                    h = round(h, 2)
                elif h > 0.0001:
                    h = round(h, 4)
                elif h > 0.000001:
                    h = round(h, 6)
                if self.category() == "int":
                    h = int(max(math.floor(h), 1))
            floor_end = -1 if (self.category() == "int") else ""
            if (h > 1) or (self.category() == "float"):
                trans = (
                    f"'[' || FLOOR({{}} / {h}) * {h} || ';' || (FLOOR({{}} / {h}) * {h} + {h}{floor_end}) || ']'",
                    "varchar",
                    "text",
                )
            else:
                trans = ("FLOOR({}) || ''", "varchar", "text")
        else:
            trans = ("{} || ''", "varchar", "text")
        if (self.isnum() and method == "same_freq") or (
            self.isnum() and method == "smart"
        ):
            n = len(result)
            trans = "(CASE "
            for i in range(1, n):
                trans += f"""
                    WHEN {{}} 
                        BETWEEN {result[i - 1]} 
                        AND {result[i]} 
                    THEN '[{result[i - 1]};{result[i]}]' """
            trans += " ELSE NULL END)"
            trans = (trans, "varchar", "text")
        if return_enum_trans:
            return trans
        else:
            self._transf += [trans]
            sauv = copy.deepcopy(self._catalog)
            self._parent._update_catalog(erase=True, columns=[self._alias])
            if "count" in sauv:
                self._catalog["count"] = sauv["count"]
                parent_cnt = self._parent.shape()[0]
                if parent_cnt == 0:
                    self._catalog["percent"] = 100
                else:
                    self._catalog["percent"] = 100 * sauv["count"] / parent_cnt
            self._parent._add_to_history(
                f"[Discretize]: The vDataColumn {self} was discretized."
            )
        return self._parent

    @save_verticapy_logs
    def one_hot_encode(
        self,
        prefix: Optional[str] = None,
        prefix_sep: str = "_",
        drop_first: bool = True,
        use_numbers_as_suffix: bool = False,
    ) -> "vDataFrame":
        """
        Encodes the vDataColumn with  the One-Hot Encoding algorithm.

        Parameters
        ----------
        prefix: str, optional
            Prefix of the dummies.
        prefix_sep: str, optional
            Prefix delimitor of the dummies.
        drop_first: bool, optional
            Drops the first dummy to avoid the creation of correlated
            features.
        use_numbers_as_suffix: bool, optional
            Uses  numbers  as  suffix  instead  of  the  vDataColumns
            categories.

        Returns
        -------
        vDataFrame
            self._parent

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

        Let's select few categorical features

        .. code-block:: python

            data = data.select(["pclass", "sex", "survived", "embarked"])
            data

        .. ipython:: python
            :suppress:

            data = data.select(["pclass", "sex", "survived", "embarked"])
            res = data
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_ohe1.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_ohe1.html

        Let's apply encoding on "embarked" vcolumn.

        .. code-block:: python

            data["embarked"].one_hot_encode()

        .. ipython:: python
            :suppress:

            res = data["embarked"].one_hot_encode()
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_ohe5.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_ohe5.html

        Let's use numbers as suffix instead of category names.

        .. code-block:: python

            data = data.select(["pclass", "sex", "survived", "embarked"])
            data["embarked"].one_hot_encode(use_numbers_as_suffix = True)

        .. ipython:: python
            :suppress:
            :okwarning:

            data = data.select(["pclass", "sex", "survived", "embarked"])
            res = data["embarked"].one_hot_encode(use_numbers_as_suffix = True)
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_ohe6.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_ohe6.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.decode` : User Defined Encoding.
            | :py:meth:`verticapy.vDataFrame.label_encode` : Label Encoding.
            | :py:meth:`verticapy.vDataFrame.mean_encode` : Mean Encoding.
            | :py:meth:`verticapy.vDataFrame.discretize` : Discretization.
        """
        distinct_elements = self.distinct()
        if distinct_elements not in ([0, 1], [1, 0]) or self.isbool():
            all_new_features = []
            if not prefix:
                prefix = self._alias.replace('"', "") + prefix_sep.replace('"', "_")
            else:
                prefix = prefix.replace('"', "_") + prefix_sep.replace('"', "_")
            n = 1 if drop_first else 0
            for k in range(len(distinct_elements) - n):
                distinct_elements_k = str(distinct_elements[k]).replace('"', "_")
                if use_numbers_as_suffix:
                    name = f'"{prefix}{k}"'
                else:
                    name = f'"{prefix}{distinct_elements_k}"'
                assert not self._parent.is_colname_in(name), NameError(
                    "A vDataColumn has already the alias of one of "
                    f"the dummies ({name}).\nIt can be the result "
                    "of using previously the method on the vDataColumn "
                    "or simply because of ambiguous columns naming."
                    "\nBy changing one of the parameters ('prefix', "
                    "'prefix_sep'), you'll be able to solve this "
                    "issue."
                )
            for k in range(len(distinct_elements) - n):
                distinct_elements_k = str(distinct_elements[k]).replace("'", "''")
                if use_numbers_as_suffix:
                    name = f'"{prefix}{k}"'
                else:
                    name = f'"{prefix}{distinct_elements_k}"'
                name = (
                    name.replace(" ", "_")
                    .replace("/", "_")
                    .replace(",", "_")
                    .replace("'", "_")
                )
                expr = f"DECODE({{}}, '{distinct_elements_k}', 1, 0)"
                transformations = self._transf + [(expr, "bool", "int")]
                new_vDataColumn = create_new_vdc(
                    name,
                    parent=self._parent,
                    transformations=transformations,
                    catalog={
                        "min": 0,
                        "max": 1,
                        "count": self._parent.shape()[0],
                        "percent": 100.0,
                        "unique": 2,
                        "approx_unique": 2,
                        "prod": 0,
                    },
                )
                setattr(self._parent, name, new_vDataColumn)
                setattr(self._parent, name.replace('"', ""), new_vDataColumn)
                self._parent._vars["columns"] += [name]
                all_new_features += [name]
            conj = "s were " if len(all_new_features) > 1 else " was "
            self._parent._add_to_history(
                "[Get Dummies]: One hot encoder was applied to the vDataColumn "
                f"{self}\n{len(all_new_features)} feature{conj}created: "
                f"{', '.join(all_new_features)}."
            )
        return self._parent

    get_dummies = one_hot_encode

    @save_verticapy_logs
    def label_encode(self) -> "vDataFrame":
        """
        Encodes the  vDataColumn using  a bijection from the different
        categories to [0, n - 1] (n being the vDataColumn cardinality).

        Returns
        -------
        vDataFrame
            self._parent

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

        Let's encode "embarked" vcolumn

        .. code-block:: python

            data["embarked"].label_encode()
            data["embarked"]

        .. ipython:: python
            :suppress:

            data["embarked"].label_encode()
            res = data["embarked"]
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_label_encode.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_label_encode.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.decode` : User Defined Encoding.
            | :py:meth:`verticapy.vDataFrame.mean_encode` : Mean Encoding.
            | :py:meth:`verticapy.vDataFrame.discretize` : Discretization.
            | :py:meth:`verticapy.vDataFrame.one_hot_encode` : One Hot Encoding.
        """
        if self.category() in ["date", "float"]:
            warning_message = (
                "label_encode is only available for categorical variables."
            )
            warnings.warn(warning_message, Warning)
        else:
            distinct_elements = self.distinct()
            expr = ["DECODE({}"]
            text_info = "\n"
            for k in range(len(distinct_elements)):
                distinct_elements_k = str(distinct_elements[k]).replace("'", "''")
                expr += [f"'{distinct_elements_k}', {k}"]
                text_info += f"\t{distinct_elements[k]} => {k}"
            expr = f"{', '.join(expr)}, {len(distinct_elements)})"
            self._transf += [(expr, "int", "int")]
            self._parent._update_catalog(erase=True, columns=[self._alias])
            self._catalog["count"] = self._parent.shape()[0]
            self._catalog["percent"] = 100
            self._parent._add_to_history(
                "[Label Encoding]: Label Encoding was applied to the vDataColumn"
                f" {self} using the following mapping:{text_info}"
            )
        return self._parent

    @save_verticapy_logs
    def mean_encode(self, response: str) -> "vDataFrame":
        """
        Encodes the vDataColumn using the average of the response
        partitioned by the different vDataColumn categories.

        Parameters
        ----------
        response: str
            Response vDataColumn.

        Returns
        -------
        vDataFrame
            self._parent

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

        Let's look at the avg of survived partitioned by embarked

        .. code-block:: python

            data.groupby(["embarked"], ["AVG(survived) AS survived"])

        .. ipython:: python
            :suppress:

            res = data.groupby(["embarked"], ["AVG(survived) AS survived"])
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_mean_encode1.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_mean_encode1.html

        Let's apply mean encoding which will replace each category of
        "embarked" vcolumn by the average of the response

        .. code-block:: python

            data["embarked"].mean_encode(response = "survived")
            data["embarked"]

        .. ipython:: python
            :suppress:

            data["embarked"].mean_encode(response = "survived")
            res = data["embarked"]
            html_file = open("SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_mean_encode2.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_encoding_mean_encode2.html

        .. seealso::

            | :py:meth:`verticapy.vDataFrame.decode` : User Defined Encoding.
            | :py:meth:`verticapy.vDataFrame.label_encode` : Label Encoding.
            | :py:meth:`verticapy.vDataFrame.discretize` : Discretization.
            | :py:meth:`verticapy.vDataFrame.one_hot_encode` : One Hot Encoding.
        """
        response = self._parent.format_colnames(response)
        assert self._parent[response].isnum(), TypeError(
            "The response column must be numerical to use a mean encoding"
        )
        max_floor = len(self._parent[response]._transf) - len(self._transf)
        self._transf += [("{}", self.ctype(), self.category())] * max_floor
        self._transf += [
            (
                f"AVG({response}) OVER (PARTITION BY {{}})",
                "int",
                "float",
            )
        ]
        self._parent._update_catalog(erase=True, columns=[self._alias])
        self._parent._add_to_history(
            f"[Mean Encode]: The vDataColumn {self} was transformed "
            f"using a mean encoding with {response} as Response Column."
        )
        if conf.get_option("print_info"):
            print("The mean encoding was successfully done.")
        return self._parent
