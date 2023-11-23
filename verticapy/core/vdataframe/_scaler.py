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

from vertica_python.errors import QueryError

import verticapy._config.config as conf
from verticapy._typing import NoneType, SQLColumns
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import format_type
from verticapy._utils._sql._sys import _executeSQL

from verticapy.core.vdataframe._text import vDFText, vDCText

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame


class vDFScaler(vDFText):
    @save_verticapy_logs
    def scale(
        self,
        columns: Optional[SQLColumns] = None,
        method: Literal["zscore", "robust_zscore", "minmax"] = "zscore",
    ) -> "vDataFrame":
        """
        Scales the input vDataColumns using the input method.

        Parameters
        ----------
        columns: SQLColumns, optional
            List of the vDataColumns names. If empty, all
            numerical vDataColumns are used.
        method: str, optional
            Method used to scale the data.
             - zscore:
                Normalization using the Z-Score.

                .. math::

                    Z_{score}(x) = (x - x_{avg}) / x_{std}

             - robust_zscore:
                Normalization using the Robust Z-Score.

                .. math::

                    Z_{rscore}(x) = (x - x_{med}) / (1.4826 * x_{mad})

             - minmax:
                Normalization using the MinMax.

                .. math::

                    Z_{minmax}(x) = (x - x_{min}) / (x_{max} - x_{min})

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

        Let's look at the "fare" and "age" of the passengers.

        .. code-block:: python

            data[["age", "fare"]]

        .. ipython:: python
            :suppress:

            res = data.select(["age", "fare"])
            html_file = open("figures/core_vDataFrame_scaler_scale1.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_scaler_scale1.html

        .. note::
            You can observe that "age" and "fare" features lie in
            different numerical intervals so it's probably a good
            idea to normalize them.

        Let's use the :py:meth:`verticapy.vDataFrame.scale` method to
        normalize the data.

        .. code-block:: python

            data.scale(
                method = "minmax",
                columns = ["age", "fare"],
            )
            data[["age", "fare"]]

        .. ipython:: python
            :suppress:

            data.scale(
                method = "minmax",
                columns = ["age", "fare"],
            )
            res = data[["age", "fare"]]
            html_file = open("figures/core_vDataFrame_scaler_scale2.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_scaler_scale2.html

        .. note::

            You can observe that both "age" and "fare" features now scale
            in [0,1] interval.
        """
        columns = format_type(columns, dtype=list)
        no_cols = len(columns) == 0
        columns = self.numcol() if not columns else self.format_colnames(columns)
        for column in columns:
            if self[column].isnum() and not self[column].isbool():
                self[column].scale(method=method)
            elif (no_cols) and (self[column].isbool()):
                pass
            elif conf.get_option("print_info"):
                warning_message = (
                    f"The vDataColumn {column} was skipped.\n"
                    "Scaler only accept numerical data types."
                )
                warnings.warn(warning_message, Warning)
        return self

    normalize = scale


class vDCScaler(vDCText):
    @save_verticapy_logs
    def scale(
        self,
        method: Literal["zscore", "robust_zscore", "minmax"] = "zscore",
        by: Optional[SQLColumns] = None,
        return_trans: bool = False,
    ) -> "vDataFrame":
        """
        Scales the input vDataColumns using the input method.

        Parameters
        ----------
        method: str, optional
            Method used to scale the data.
             - zscore:
                Normalization using the Z-Score.

                .. math::

                    Z_{score}(x) = (x - x_{avg}) / x_{std}

             - robust_zscore:
                Normalization using the Robust Z-Score.

                .. math::

                    Z_{rscore}(x) = (x - x_{med}) / (1.4826 * x_{mad})

             - minmax:
                Normalization using the MinMax.

                .. math::

                    Z_{minmax}(x) = (x - x_{min}) / (x_{max} - x_{min})
        by: SQLColumns, optional
            vDataColumns used in the partition.
        return_trans: bool, optimal
            If  set to True,  the method  returns the  transformation
            used instead of the parent vDataFrame. This parameter is used
            for testing purposes.

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

        Let's look at the "fare" and "age" of the passengers.

        .. code-block:: python

            data[["age", "fare"]]

        .. ipython:: python
            :suppress:

            res = data.select(["age", "fare"])
            html_file = open("figures/core_vDataFrame_scaler_scale1.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_scaler_scale1.html

        .. note::
            You can observe that "age" and "fare" features lie
            in different numerical intervals so it's probably a
            good idea to normalize them.

        Let's use the :py:meth:`verticapy.vDataColumn.scale` method to
        normalize the data.

        .. code-block:: python

            data["age"].scale(method = "minmax")
            data["fare"].scale(method = "minmax")
            data[["age", "fare"]]

        .. ipython:: python
            :suppress:

            data["age"].scale(method = "minmax")
            data["fare"].scale(method = "minmax")
            res = data[["age", "fare"]]
            html_file = open("figures/core_vDataFrame_scaler_scale2.html", "w")
            html_file.write(res._repr_html_())
            html_file.close()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/core_vDataFrame_scaler_scale2.html

        .. note::
            You can observe that both "age" and "fare" features now scale
            in [0,1] interval.
        """
        method = method.lower()
        by = format_type(by, dtype=list)
        by = self._parent.format_colnames(by)
        nullifzero, n = 1, len(by)

        if self.isbool():
            warning_message = "Scaler doesn't work on booleans"
            warnings.warn(warning_message, Warning)

        elif self.isnum():
            if method == "zscore":
                if n == 0:
                    nullifzero = 0
                    avg, stddev = self.aggregate(["avg", "std"]).values[self._alias]
                    if stddev == 0:
                        warning_message = (
                            f"Can not scale {self} using a "
                            "Z-Score - The Standard Deviation is null !"
                        )
                        warnings.warn(warning_message, Warning)
                        return self
                elif (n == 1) and (self._parent[by[0]].nunique() < 50):
                    try:
                        result = _executeSQL(
                            query=f"""
                                SELECT 
                                    /*+LABEL('vDataColumn.scale')*/ 
                                    {by[0]}, 
                                    AVG({self}), 
                                    STDDEV({self}) 
                                FROM {self._parent} GROUP BY {by[0]}""",
                            title="Computing the different categories to scale.",
                            method="fetchall",
                            sql_push_ext=self._parent._vars["sql_push_ext"],
                            symbol=self._parent._vars["symbol"],
                        )
                        for i in range(len(result)):
                            if not isinstance(result[i][2], NoneType) and math.isnan(
                                result[i][2]
                            ):
                                result[i][2] = None
                        avg = "DECODE({}, {}, NULL)".format(
                            by[0],
                            ", ".join(
                                [
                                    "{}, {}".format(
                                        "'{}'".format(str(x[0]).replace("'", "''"))
                                        if not isinstance(x[0], NoneType)
                                        else "NULL",
                                        x[1]
                                        if not isinstance(x[1], NoneType)
                                        else "NULL",
                                    )
                                    for x in result
                                    if not isinstance(x[1], NoneType)
                                ]
                            ),
                        )
                        stddev = "DECODE({}, {}, NULL)".format(
                            by[0],
                            ", ".join(
                                [
                                    "{}, {}".format(
                                        "'{}'".format(str(x[0]).replace("'", "''"))
                                        if not isinstance(x[0], NoneType)
                                        else "NULL",
                                        x[2]
                                        if not isinstance(x[2], NoneType)
                                        else "NULL",
                                    )
                                    for x in result
                                    if not isinstance(x[2], NoneType)
                                ]
                            ),
                        )
                        _executeSQL(
                            query=f"""
                                SELECT 
                                    /*+LABEL('vDataColumn.scale')*/ 
                                    {avg},
                                    {stddev} 
                                FROM {self._parent} 
                                LIMIT 1""",
                            print_time_sql=False,
                            sql_push_ext=self._parent._vars["sql_push_ext"],
                            symbol=self._parent._vars["symbol"],
                        )
                    except QueryError:
                        avg, stddev = (
                            f"AVG({self}) OVER (PARTITION BY {', '.join(by)})",
                            f"STDDEV({self}) OVER (PARTITION BY {', '.join(by)})",
                        )
                else:
                    avg, stddev = (
                        f"AVG({self}) OVER (PARTITION BY {', '.join(by)})",
                        f"STDDEV({self}) OVER (PARTITION BY {', '.join(by)})",
                    )
                nullifzero = "NULLIFZERO" if (nullifzero) else ""
                if return_trans:
                    return f"({self} - {avg}) / {nullifzero}({stddev})"
                else:
                    final_transformation = [
                        (
                            f"({{}} - {avg}) / {nullifzero}({stddev})",
                            "float",
                            "float",
                        )
                    ]

            elif method == "robust_zscore":
                if n > 0:
                    warning_message = (
                        "The method 'robust_zscore' is available only if the "
                        "parameter 'by' is empty\nIf you want to scale the data by "
                        "grouping by elements, please use a method in zscore|minmax"
                    )
                    warnings.warn(warning_message, Warning)
                    return self
                mad, med = self.aggregate(["mad", "approx_median"]).values[self._alias]
                mad *= 1.4826
                if mad != 0:
                    if return_trans:
                        return f"({self} - {med}) / ({mad})"
                    else:
                        final_transformation = [
                            (
                                f"({{}} - {med}) / ({mad})",
                                "float",
                                "float",
                            )
                        ]
                else:
                    warning_message = (
                        f"Can not scale {self} using a "
                        "Robust Z-Score - The MAD is null !"
                    )
                    warnings.warn(warning_message, Warning)
                    return self

            elif method == "minmax":
                if n == 0:
                    nullifzero = 0
                    cmin, cmax = self.aggregate(["min", "max"]).values[self._alias]
                    if cmax - cmin == 0:
                        warning_message = (
                            f"Can not scale {self} using "
                            "the MIN and the MAX. MAX = MIN !"
                        )
                        warnings.warn(warning_message, Warning)
                        return self
                elif n == 1:
                    try:
                        result = _executeSQL(
                            query=f"""
                                SELECT 
                                    /*+LABEL('vDataColumn.scale')*/ 
                                    {by[0]}, 
                                    MIN({self}), 
                                    MAX({self})
                                FROM {self._parent} 
                                GROUP BY {by[0]}""",
                            title=f"Computing the different categories {by[0]} to scale.",
                            method="fetchall",
                            sql_push_ext=self._parent._vars["sql_push_ext"],
                            symbol=self._parent._vars["symbol"],
                        )
                        cmin = "DECODE({}, {}, NULL)".format(
                            by[0],
                            ", ".join(
                                [
                                    "{}, {}".format(
                                        "'{}'".format(str(x[0]).replace("'", "''"))
                                        if not isinstance(x[0], NoneType)
                                        else "NULL",
                                        x[1]
                                        if not isinstance(x[1], NoneType)
                                        else "NULL",
                                    )
                                    for x in result
                                    if not isinstance(x[1], NoneType)
                                ]
                            ),
                        )
                        cmax = "DECODE({}, {}, NULL)".format(
                            by[0],
                            ", ".join(
                                [
                                    "{}, {}".format(
                                        "'{}'".format(str(x[0]).replace("'", "''"))
                                        if not isinstance(x[0], NoneType)
                                        else "NULL",
                                        x[2]
                                        if not isinstance(x[2], NoneType)
                                        else "NULL",
                                    )
                                    for x in result
                                    if not isinstance(x[2], NoneType)
                                ]
                            ),
                        )
                        _executeSQL(
                            query=f"""
                                SELECT 
                                    /*+LABEL('vDataColumn.scale')*/ 
                                    {cmax}, 
                                    {cmin} 
                                FROM {self._parent} 
                                LIMIT 1""",
                            print_time_sql=False,
                            sql_push_ext=self._parent._vars["sql_push_ext"],
                            symbol=self._parent._vars["symbol"],
                        )
                    except QueryError:
                        cmax, cmin = (
                            f"MAX({self}) OVER (PARTITION BY {', '.join(by)})",
                            f"MIN({self}) OVER (PARTITION BY {', '.join(by)})",
                        )
                else:
                    cmax, cmin = (
                        f"MAX({self}) OVER (PARTITION BY {', '.join(by)})",
                        f"MIN({self}) OVER (PARTITION BY {', '.join(by)})",
                    )
                nullifzero = "NULLIFZERO" if (nullifzero) else ""
                if return_trans:
                    return f"({self} - {cmin}) / {nullifzero}({cmax} - {cmin})"
                else:
                    final_transformation = [
                        (
                            f"({{}} - {cmin}) / {nullifzero}({cmax} - {cmin})",
                            "float",
                            "float",
                        )
                    ]

            if method != "robust_zscore":
                max_floor = 0
                for elem in by:
                    if len(self._parent[elem]._transf) > max_floor:
                        max_floor = len(self._parent[elem]._transf)
                max_floor -= len(self._transf)
                self._transf += [("{}", self.ctype(), self.category())] * max_floor
            self._transf += final_transformation
            sauv = copy.deepcopy(self._catalog)
            self._parent._update_catalog(erase=True, columns=[self._alias])

            parent_cnt = self._parent.shape()[0]

            if "count" in sauv:
                self._catalog["count"] = sauv["count"]
                if parent_cnt == 0:
                    self._catalog["percent"] = 100
                else:
                    self._catalog["percent"] = 100 * sauv["count"] / parent_cnt

            for elem in sauv:
                if "top" in elem:
                    if "percent" in elem:
                        self._catalog[elem] = sauv[elem]
                    elif isinstance(elem, NoneType):
                        self._catalog[elem] = None
                    elif method == "robust_zscore":
                        self._catalog[elem] = (sauv[elem] - sauv["approx_50%"]) / (
                            1.4826 * sauv["mad"]
                        )
                    elif method == "zscore":
                        self._catalog[elem] = (sauv[elem] - sauv["mean"]) / sauv["std"]
                    elif method == "minmax":
                        self._catalog[elem] = (sauv[elem] - sauv["min"]) / (
                            sauv["max"] - sauv["min"]
                        )

            if method == "robust_zscore":
                self._catalog["median"] = 0
                self._catalog["mad"] = 1 / 1.4826
            elif method == "zscore":
                self._catalog["mean"] = 0
                self._catalog["std"] = 1
            elif method == "minmax":
                self._catalog["min"] = 0
                self._catalog["max"] = 1
            self._parent._add_to_history(
                f"[Scaler]: The vDataColumn '{self}' was "
                f"scaled with the method '{method}'."
            )
        else:
            raise TypeError("The vDataColumn must be numerical for Normalization")
        return self._parent

    normalize = scale
