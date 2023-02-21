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
import math, warnings
from typing import Literal, Union

from verticapy._config.config import OPTIONS
from verticapy._utils._collect import save_verticapy_logs
from verticapy._utils._sql._execute import _executeSQL


class vDFNORM:
    @save_verticapy_logs
    def normalize(
        self,
        columns: Union[str, list] = [],
        method: Literal["zscore", "robust_zscore", "minmax"] = "zscore",
    ):
        """
    Normalizes the input vDataColumns using the input method.

    Parameters
    ----------
    columns: str / list, optional
        List of the vDataColumns names. If empty, all numerical vDataColumns will be 
        used.
    method: str, optional
        Method to use to normalize.
            zscore        : Normalization using the Z-Score (avg and std).
                (x - avg) / std
            robust_zscore : Normalization using the Robust Z-Score (median and mad).
                (x - median) / (1.4826 * mad)
            minmax        : Normalization using the MinMax (min and max).
                (x - min) / (max - min)

    Returns
    -------
    vDataFrame
        self

    See Also
    --------
    vDataFrame.outliers    : Computes the vDataFrame Global Outliers.
    vDataFrame[].normalize : Normalizes the vDataColumn. This method is more complete 
        than the vDataFrame.normalize method by allowing more parameters.
        """
        if isinstance(columns, str):
            columns = [columns]
        no_cols = True if not (columns) else False
        columns = self.numcol() if not (columns) else self._format_colnames(columns)
        for column in columns:
            if self[column].isnum() and not (self[column].isbool()):
                self[column].normalize(method=method)
            elif (no_cols) and (self[column].isbool()):
                pass
            elif OPTIONS["print_info"]:
                warning_message = (
                    f"The vDataColumn {column} was skipped.\n"
                    "Normalize only accept numerical data types."
                )
                warnings.warn(warning_message, Warning)
        return self


class vDCNORM:
    @save_verticapy_logs
    def normalize(
        self,
        method: Literal["zscore", "robust_zscore", "minmax"] = "zscore",
        by: Union[str, list] = [],
        return_trans: bool = False,
    ):
        """
    Normalizes the input vDataColumns using the input method.

    Parameters
    ----------
    method: str, optional
        Method to use to normalize.
            zscore        : Normalization using the Z-Score (avg and std).
                (x - avg) / std
            robust_zscore : Normalization using the Robust Z-Score (median and mad).
                (x - median) / (1.4826 * mad)
            minmax        : Normalization using the MinMax (min and max).
                (x - min) / (max - min)
    by: str / list, optional
        vDataColumns used in the partition.
    return_trans: bool, optimal
        If set to True, the method will return the transformation used instead of
        the parent vDataFrame. This parameter is used for testing purpose.

    Returns
    -------
    vDataFrame
        self._PARENT

    See Also
    --------
    vDataFrame.outliers : Computes the vDataFrame Global Outliers.
        """
        if isinstance(by, str):
            by = [by]
        method = method.lower()
        by = self._PARENT._format_colnames(by)
        nullifzero, n = 1, len(by)
        if self.isbool():

            warning_message = "Normalize doesn't work on booleans"
            warnings.warn(warning_message, Warning)

        elif self.isnum():

            if method == "zscore":

                if n == 0:
                    nullifzero = 0
                    avg, stddev = self.aggregate(["avg", "std"]).values[self._ALIAS]
                    if stddev == 0:
                        warning_message = (
                            f"Can not normalize {self._ALIAS} using a "
                            "Z-Score - The Standard Deviation is null !"
                        )
                        warnings.warn(warning_message, Warning)
                        return self
                elif (n == 1) and (self._PARENT[by[0]].nunique() < 50):
                    try:
                        result = _executeSQL(
                            query=f"""
                                SELECT 
                                    /*+LABEL('vDataColumn.normalize')*/ {by[0]}, 
                                    AVG({self._ALIAS}), 
                                    STDDEV({self._ALIAS}) 
                                FROM {self._PARENT._genSQL()} GROUP BY {by[0]}""",
                            title="Computing the different categories to normalize.",
                            method="fetchall",
                            sql_push_ext=self._PARENT._VARS["sql_push_ext"],
                            symbol=self._PARENT._VARS["symbol"],
                        )
                        for i in range(len(result)):
                            if result[i][2] == None:
                                pass
                            elif math.isnan(result[i][2]):
                                result[i][2] = None
                        avg_stddev = []
                        for i in range(1, 3):
                            if x[0] != None:
                                x0 = f"""'{str(x[0]).replace("'", "''")}'"""
                            else:
                                x0 = "NULL"
                            x_tmp = [
                                f"""{x0}, {x[i] if x[i] != None else "NULL"}"""
                                for x in result
                                if x[i] != None
                            ]
                            avg_stddev += [
                                f"""DECODE({by[0]}, {", ".join(x_tmp)}, NULL)"""
                            ]
                        avg, stddev = avg_stddev
                        _executeSQL(
                            query=f"""
                                SELECT 
                                    /*+LABEL('vDataColumn.normalize')*/ 
                                    {avg},
                                    {stddev} 
                                FROM {self._PARENT._genSQL()} 
                                LIMIT 1""",
                            print_time_sql=False,
                            sql_push_ext=self._PARENT._VARS["sql_push_ext"],
                            symbol=self._PARENT._VARS["symbol"],
                        )
                    except:
                        avg, stddev = (
                            f"AVG({self._ALIAS}) OVER (PARTITION BY {', '.join(by)})",
                            f"STDDEV({self._ALIAS}) OVER (PARTITION BY {', '.join(by)})",
                        )
                else:
                    avg, stddev = (
                        f"AVG({self._ALIAS}) OVER (PARTITION BY {', '.join(by)})",
                        f"STDDEV({self._ALIAS}) OVER (PARTITION BY {', '.join(by)})",
                    )
                nullifzero = "NULLIFZERO" if (nullifzero) else ""
                if return_trans:
                    return f"({self._ALIAS} - {avg}) / {nullifzero}({stddev})"
                else:
                    final_transformation = [
                        (f"({{}} - {avg}) / {nullifzero}({stddev})", "float", "float",)
                    ]

            elif method == "robust_zscore":

                if n > 0:
                    warning_message = (
                        "The method 'robust_zscore' is available only if the "
                        "parameter 'by' is empty\nIf you want to normalize by "
                        "grouping by elements, please use a method in zscore|minmax"
                    )
                    warnings.warn(warning_message, Warning)
                    return self
                mad, med = self.aggregate(["mad", "approx_median"]).values[self._ALIAS]
                mad *= 1.4826
                if mad != 0:
                    if return_trans:
                        return f"({self._ALIAS} - {med}) / ({mad})"
                    else:
                        final_transformation = [
                            (f"({{}} - {med}) / ({mad})", "float", "float",)
                        ]
                else:
                    warning_message = (
                        f"Can not normalize {self._ALIAS} using a "
                        "Robust Z-Score - The MAD is null !"
                    )
                    warnings.warn(warning_message, Warning)
                    return self

            elif method == "minmax":

                if n == 0:
                    nullifzero = 0
                    cmin, cmax = self.aggregate(["min", "max"]).values[self._ALIAS]
                    if cmax - cmin == 0:
                        warning_message = (
                            f"Can not normalize {self._ALIAS} using "
                            "the MIN and the MAX. MAX = MIN !"
                        )
                        warnings.warn(warning_message, Warning)
                        return self
                elif n == 1:
                    try:
                        result = _executeSQL(
                            query=f"""
                                SELECT 
                                    /*+LABEL('vDataColumn.normalize')*/ {by[0]}, 
                                    MIN({self._ALIAS}), 
                                    MAX({self._ALIAS})
                                FROM {self._PARENT._genSQL()} 
                                GROUP BY {by[0]}""",
                            title=f"Computing the different categories {by[0]} to normalize.",
                            method="fetchall",
                            sql_push_ext=self._PARENT._VARS["sql_push_ext"],
                            symbol=self._PARENT._VARS["symbol"],
                        )
                        cmin_cmax = []
                        for i in range(1, 3):
                            if x[0] != None:
                                x0 = f"""'{str(x[0]).replace("'", "''")}'"""
                            else:
                                x0 = "NULL"
                            x_tmp = [
                                f"""{x0}, {x[i] if x[i] != None else "NULL"}"""
                                for x in result
                                if x[i] != None
                            ]
                            cmin_cmax += [
                                f"""DECODE({by[0]}, {", ".join(x_tmp)}, NULL)"""
                            ]
                        cmin, cmax = cmin_cmax
                        _executeSQL(
                            query=f"""
                                SELECT 
                                    /*+LABEL('vDataColumn.normalize')*/ 
                                    {cmin_cmax[1]}, 
                                    {cmin_cmax[0]} 
                                FROM {self._PARENT._genSQL()} 
                                LIMIT 1""",
                            print_time_sql=False,
                            sql_push_ext=self._PARENT._VARS["sql_push_ext"],
                            symbol=self._PARENT._VARS["symbol"],
                        )
                    except:
                        cmax, cmin = (
                            f"MAX({self._ALIAS}) OVER (PARTITION BY {', '.join(by)})",
                            f"MIN({self._ALIAS}) OVER (PARTITION BY {', '.join(by)})",
                        )
                else:
                    cmax, cmin = (
                        f"MAX({self._ALIAS}) OVER (PARTITION BY {', '.join(by)})",
                        f"MIN({self._ALIAS}) OVER (PARTITION BY {', '.join(by)})",
                    )
                nullifzero = "NULLIFZERO" if (nullifzero) else ""
                if return_trans:
                    return f"({self._ALIAS} - {cmin}) / {nullifzero}({cmax} - {cmin})"
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
                    if len(self._PARENT[elem]._TRANSF) > max_floor:
                        max_floor = len(self._PARENT[elem]._TRANSF)
                max_floor -= len(self._TRANSF)
                for k in range(max_floor):
                    self._TRANSF += [("{}", self.ctype(), self.category())]
            self._TRANSF += final_transformation
            sauv = {}
            for elem in self._CATALOG:
                sauv[elem] = self._CATALOG[elem]
            self._PARENT._update_catalog(erase=True, columns=[self._ALIAS])
            try:

                if "count" in sauv:
                    self._CATALOG["count"] = sauv["count"]
                    self._CATALOG["percent"] = (
                        100 * sauv["count"] / self._PARENT.shape()[0]
                    )

                for elem in sauv:

                    if "top" in elem:

                        if "percent" in elem:
                            self._CATALOG[elem] = sauv[elem]
                        elif elem == None:
                            self._CATALOG[elem] = None
                        elif method == "robust_zscore":
                            self._CATALOG[elem] = (sauv[elem] - sauv["approx_50%"]) / (
                                1.4826 * sauv["mad"]
                            )
                        elif method == "zscore":
                            self._CATALOG[elem] = (sauv[elem] - sauv["mean"]) / sauv[
                                "std"
                            ]
                        elif method == "minmax":
                            self._CATALOG[elem] = (sauv[elem] - sauv["min"]) / (
                                sauv["max"] - sauv["min"]
                            )

            except:
                pass
            if method == "robust_zscore":
                self._CATALOG["median"] = 0
                self._CATALOG["mad"] = 1 / 1.4826
            elif method == "zscore":
                self._CATALOG["mean"] = 0
                self._CATALOG["std"] = 1
            elif method == "minmax":
                self._CATALOG["min"] = 0
                self._CATALOG["max"] = 1
            self._PARENT._add_to_history(
                f"[Normalize]: The vDataColumn '{self._ALIAS}' was "
                f"normalized with the method '{method}'."
            )
        else:
            raise TypeError("The vDataColumn must be numerical for Normalization")
        return self._PARENT
