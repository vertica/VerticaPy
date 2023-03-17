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
import copy, math
from typing import Callable, Literal, Optional, Union, TYPE_CHECKING
import numpy as np

import matplotlib.colors as plt_colors

import verticapy._config.config as conf
from verticapy._config.colors import get_colors
from verticapy._typing import ArrayLike, PythonNumber, PythonScalar, SQLColumns
from verticapy._utils._sql._cast import to_varchar
from verticapy._utils._sql._format import clean_query, quote_ident
from verticapy._utils._sql._sys import _executeSQL
from verticapy.errors import ParameterError

from verticapy.core.string_sql.base import StringSQL
from verticapy.core.tablesample.base import TableSample

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame, vDataColumn

if conf._get_import_success("dateutil"):
    from dateutil.parser import parse


class PlottingBase:

    # Properties.

    @property
    def _compute_method(self) -> Literal[None]:
        """Must be overridden in child class"""
        return None

    @property
    def _dimension_bounds(self) -> tuple[PythonNumber, PythonNumber]:
        """Must be overridden in child class"""
        return (-np.inf, np.inf)

    @property
    def _only_standard(self) -> Literal[False]:
        return False

    # System Methods.

    def __init__(self, *args, **kwargs) -> None:
        kwds = copy.deepcopy(kwargs)
        if "misc_data" in kwds:
            misc_data = copy.deepcopy(kwds["misc_data"])
            del kwds["misc_data"]
        else:
            misc_data = {}
        if "data" not in kwds or "layout" not in kwds:
            functions = {
                "1D": self._compute_plot_params,
                "2D": self._compute_pivot_table,
                "aggregate": self._compute_aggregate,
                "contour": self._compute_contour_grid,
                "describe": self._compute_statistics,
                "hist": self._compute_hists_params,
                "line": self._filter_line,
                "matrix": self._compute_scatter_matrix,
                "outliers": self._compute_outliers_params,
                "range": self._compute_range,
                "rollup": self._compute_rollup,
                "sample": self._sample,
            }
            if self._compute_method in functions:
                functions[self._compute_method](*args, **kwds)
        else:
            self.data = copy.deepcopy(kwds["data"])
            self.layout = copy.deepcopy(kwds["layout"])
        if hasattr(self, "data"):
            self.data = {
                **self.data,
                **misc_data,
            }
        self._init_style()
        return None

    def _init_check(self, dim: int, is_standard: bool) -> None:
        lower, upper = self._dimension_bounds
        if not (lower <= dim <= upper):
            if lower == upper:
                message = f"exactly {lower}"
            else:
                message = f"between {lower} and {upper}."
            raise ParameterError(
                f"The number of columns to draw the plot must be {message}. Found {n}."
            )
        if self._only_standard and not (is_standard):
            raise ParameterError(
                f"When drawing {self._kind} {self._category}s, the parameter "
                "'method' can not represent a customized aggregation."
            )

    # Styling Methods.

    def _init_style(self) -> None:
        """Must be overridden in child class"""
        self.init_style = {}
        return None

    @staticmethod
    def get_colors(
        d: Optional[dict] = {}, idx: Optional[int] = None
    ) -> Union[list, str]:
        return get_colors(d=d, idx=idx)

    def get_cmap(
        self,
        color: Union[None, str, list] = None,
        reverse: bool = False,
        idx: Optional[int] = None,
    ) -> Union[
        tuple[plt_colors.LinearSegmentedColormap, plt_colors.LinearSegmentedColormap],
        plt_colors.LinearSegmentedColormap,
    ]:
        """
        Returns the CMAP associated to the input color.
        If  empty, VerticaPy uses  the colors stored as 
        a global variable.
        """
        cmap_from_list = plt_colors.LinearSegmentedColormap.from_list
        kwargs = {"N": 1000}
        args = ["verticapy_cmap"]
        if not (color):
            args1 = args + [["#FFFFFF", self.get_colors(idx=0)]]
            args2 = args + [[self.get_colors(idx=1), "#FFFFFF", self.get_colors(idx=0)]]
            cm1 = cmap_from_list(*args1, **kwargs)
            cm2 = cmap_from_list(*args2, **kwargs)
            if idx == None:
                return (cm1, cm2)
            elif idx == 0:
                return cm1
            else:
                return cm2
        else:
            if isinstance(color, list):
                args += [color]
            elif reverse:
                args += [[color, "#FFFFFF"]]
            else:
                args += [["#FFFFFF", color]]
            return cmap_from_list(*args, **kwargs)

    # Formatting Methods.

    @staticmethod
    def _map_method(method: str, of: str) -> tuple[str, str, Optional[Callable], bool]:
        is_standard = True
        fun_map = {
            "avg": np.mean,
            "min": min,
            "max": max,
            "sum": sum,
        }
        method = method.lower()
        if method == "median":
            method = "50%"
        elif method == "mean":
            method = "avg"
        if (
            method not in ["avg", "min", "max", "sum", "density", "count"]
            and "%" != method[-1]
        ) and of:
            raise ParameterError(
                "Parameter 'of' must be empty when using customized aggregations."
            )
        if (
            (method in ["avg", "min", "max", "sum"]) or (method and method[-1] == "%")
        ) and (of):
            if method in ["avg", "min", "max", "sum"]:
                aggregate = f"{method.upper()}({quote_ident(of)})"
                fun = fun_map[method]
            elif method and method[-1] == "%":
                q = float(method[0:-1]) / 100
                aggregate = f"""
                    APPROXIMATE_PERCENTILE({quote_ident(of)} 
                        USING PARAMETERS
                        percentile = {q})"""
                fun = lambda x: np.quantile(x, q)
            else:
                raise ValueError(
                    "The parameter 'method' must be in [avg|mean|min|max|sum|"
                    f"median|q%] or a customized aggregation. Found {method}."
                )
        elif method in ["density", "count"]:
            aggregate = "count(*)"
            fun = sum
        elif isinstance(method, str):
            aggregate = method
            fun = None
            is_standard = False
        else:
            raise ParameterError(
                "The parameter 'method' must be in [avg|mean|min|max|sum|"
                f"median|q%] or a customized aggregation. Found {method}."
            )
        return method, aggregate, fun, is_standard

    @staticmethod
    def _parse_datetime(D: list) -> list:
        """
        Parses the list and casts the value to the datetime
        format if possible.
        """
        try:
            return np.array([parse(d) for d in D])
        except:
            return copy.deepcopy(D)

    @staticmethod
    def _update_dict(d1: dict, d2: dict, color_idx: int = 0,) -> dict:
        """
        Updates the input dictionary using another one.
        """
        d = {}
        for elem in d1:
            d[elem] = d1[elem]
        for elem in d2:
            if elem == "color":
                if isinstance(d2["color"], str):
                    d["color"] = d2["color"]
                elif color_idx < 0:
                    d["color"] = [elem for elem in d2["color"]]
                else:
                    d["color"] = d2["color"][color_idx % len(d2["color"])]
            else:
                d[elem] = d2[elem]
        return d

    # Attributes Computations.

    def _compute_contour_grid(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        func: Union[str, StringSQL, Callable],
        nbins: int = 100,
        func_name: Optional[str] = None,
    ) -> None:
        from verticapy.datasets.generators import gen_meshgrid

        columns = vdf._format_colnames(columns)
        aggregations = vdf.agg(["min", "max"], columns).to_numpy()
        self.data = {
            "min": aggregations[:, 0],
            "max": aggregations[:, 1],
        }
        if isinstance(func, Callable):
            x_grid = np.linspace(self.data["min"][0], self.data["max"][0], nbins)
            y_grid = np.linspace(self.data["min"][1], self.data["max"][1], nbins)
            X, Y = np.meshgrid(x_grid, y_grid)
            Z = func(X, Y)
        elif isinstance(func, (str, StringSQL)):
            d = {}
            for i in range(2):
                d[quote_ident(columns[i])[1:-1]] = {
                    "type": float,
                    "range": [self.data["min"][i], self.data["max"][i]],
                    "nbins": nbins,
                }
            vdf_tmp = gen_meshgrid(d)
            if "{0}" in func and "{1}" in func:
                vdf_tmp = vdf._new_vdataframe(func.format("_contour_Z", vdf_tmp))
            else:
                vdf_tmp["_contour_Z"] = func
            dataset = (
                vdf_tmp[[columns[1], columns[0], "_contour_Z"]].sort(columns).to_numpy()
            )
            i, y_start, y_new = 0, dataset[0][1], dataset[0][1]
            n = len(dataset)
            X, Y, Z = [], [], []
            while i < n:
                x_tmp, y_tmp, z_tmp = [], [], []
                j, last_non_null_value = 0, 0
                while y_start == y_new and i < n and j < nbins:
                    if dataset[i][2] != None:
                        last_non_null_value = float(dataset[i][2])
                    x_tmp += [float(dataset[i][1])]
                    y_tmp += [float(dataset[i][0])]
                    z_tmp += [
                        float(
                            dataset[i][2]
                            if (dataset[i][2] != None)
                            else last_non_null_value
                        )
                    ]
                    y_new = dataset[i][1]
                    i, j = i + 1, j + 1
                    if j == nbins:
                        while y_start == y_new and i < n:
                            y_new = dataset[i][1]
                            i += 1
                y_start = y_new
                X, Y, Z = X + [x_tmp], Y + [y_tmp], Z + [z_tmp]
        else:
            raise TypeError
        self.data = {**self.data, "X": np.array(X), "Y": np.array(Y), "Z": np.array(Z)}
        func_repr = func.__name__ if isinstance(func, Callable) else str(func)
        self.layout = {
            "columns": columns,
            "func": func,
            "func_repr": func_name if func_name != None else func_repr,
        }
        return None

    def _compute_plot_params(
        self,
        vdc: "vDataColumn",
        method: str = "density",
        of: Optional[str] = None,
        max_cardinality: int = 6,
        nbins: int = 0,
        h: float = 0.0,
        pie: bool = False,
    ) -> None:
        """
	    Computes the aggregations needed to draw a 1D graphic 
	    using the Matplotlib API.
	    """
        other_columns = ""
        of = vdc._parent._format_colnames(of)
        method, aggregate, aggregate_fun, is_standard = self._map_method(method, of)
        if not (is_standard):
            other_columns = ", " + ", ".join(
                vdc._parent.get_columns(exclude_columns=[vdc._alias])
            )
        # depending on the cardinality, the type, the vDataColumn
        # can be treated as categorical or not
        cardinality, count, is_numeric, is_date, is_categorical = (
            vdc.nunique(True),
            vdc._parent.shape()[0],
            vdc.isnum() and not (vdc.isbool()),
            (vdc.category() == "date"),
            False,
        )
        rotation = 0 if ((is_numeric) and (cardinality > max_cardinality)) else 90
        # case when categorical
        if (((cardinality <= max_cardinality) or not (is_numeric)) or pie) and not (
            is_date
        ):
            if (is_numeric) and not (pie):
                query = f"""
	                SELECT 
	                    {vdc._alias},
	                    {aggregate}
	                FROM {vdc._parent._genSQL()} 
	                WHERE {vdc._alias} IS NOT NULL 
	                GROUP BY {vdc._alias} 
	                ORDER BY {vdc._alias} ASC 
	                LIMIT {max_cardinality}"""
            else:
                table = vdc._parent._genSQL()
                if (pie) and (is_numeric):
                    enum_trans = (
                        vdc.discretize(h=h, return_enum_trans=True)[0].replace(
                            "{}", vdc._alias
                        )
                        + " AS "
                        + vdc._alias
                    )
                    if of:
                        enum_trans += f" , {of}"
                    table = (
                        f"(SELECT {enum_trans + other_columns} FROM {table}) enum_table"
                    )
                cast_alias = to_varchar(vdc.category(), vdc._alias)
                query = f"""
	                (SELECT 
	                    /*+LABEL('plotting._matplotlib._compute_plot_params')*/ 
	                    {cast_alias} AS {vdc._alias},
	                    {aggregate}
	                 FROM {table} 
	                 GROUP BY {cast_alias} 
	                 ORDER BY 2 DESC 
	                 LIMIT {max_cardinality})"""
                if cardinality > max_cardinality:
                    query += f"""
	                    UNION 
	                    (SELECT 
	                        'Others',
	                        {aggregate} 
	                     FROM {table}
	                     WHERE {vdc._alias} NOT IN
	                     (SELECT 
	                        {vdc._alias} 
	                      FROM {table}
	                      GROUP BY {vdc._alias}
	                      ORDER BY {aggregate} DESC
	                      LIMIT {max_cardinality}))"""
            query_result = _executeSQL(
                query=query, title="Computing the histogram heights", method="fetchall"
            )
            if query_result[-1][1] == None:
                del query_result[-1]
            y = (
                [
                    item[1] / float(count) if item[1] != None else 0
                    for item in query_result
                ]
                if (method.lower() == "density")
                else [item[1] if item[1] != None else 0 for item in query_result]
            )
            x = [0.4 * i + 0.2 for i in range(0, len(y))]
            adj_width = 0.39
            labels = [item[0] for item in query_result]
            is_categorical = True
        # case when date
        elif is_date:
            if (h <= 0) and (nbins <= 0):
                h = vdc.numh()
            elif nbins > 0:
                query_result = _executeSQL(
                    query=f"""
	                    SELECT 
	                        /*+LABEL('plotting._matplotlib._compute_plot_params')*/
	                        DATEDIFF('second', MIN({vdc._alias}), MAX({vdc._alias}))
	                    FROM {vdc._parent._genSQL()}""",
                    title="Computing the histogram interval",
                    method="fetchrow",
                )
                h = float(query_result[0]) / nbins
            min_date = vdc.min()
            converted_date = f"DATEDIFF('second', '{min_date}', {vdc._alias})"
            query_result = _executeSQL(
                query=f"""
	                SELECT 
	                    /*+LABEL('plotting._matplotlib._compute_plot_params')*/
	                    FLOOR({converted_date} / {h}) * {h}, 
	                    {aggregate} 
	                FROM {vdc._parent._genSQL()}
	                WHERE {vdc._alias} IS NOT NULL 
	                GROUP BY 1 
	                ORDER BY 1""",
                title="Computing the histogram heights",
                method="fetchall",
            )
            x = [float(item[0]) for item in query_result]
            y = (
                [item[1] / float(count) for item in query_result]
                if (method.lower() == "density")
                else [item[1] for item in query_result]
            )
            query = "("
            for idx, item in enumerate(query_result):
                query_tmp = f"""
	                (SELECT 
	                    {{}}
	                    TIMESTAMPADD('second',
	                                 {math.floor(h * idx)},
	                                 '{min_date}'::timestamp))"""
                if idx == 0:
                    query += query_tmp.format(
                        "/*+LABEL('plotting._matplotlib._compute_plot_params')*/"
                    )
                else:
                    query += f" UNION {query_tmp.format('')}"
            query += ")"
            query_result = _executeSQL(
                query, title="Computing the datetime intervals.", method="fetchall"
            )
            adj_width = 0.94 * h
            labels = [item[0] for item in query_result]
            labels.sort()
            is_categorical = True
        # case when numerical
        else:
            if (h <= 0) and (nbins <= 0):
                h = vdc.numh()
            elif nbins > 0:
                h = float(vdc.max() - vdc.min()) / nbins
            if (vdc.ctype == "int") or (h == 0):
                h = max(1.0, h)
            query_result = _executeSQL(
                query=f"""
	                SELECT
	                    /*+LABEL('plotting._matplotlib._compute_plot_params')*/
	                    FLOOR({vdc._alias} / {h}) * {h},
	                    {aggregate} 
	                FROM {vdc._parent._genSQL()}
	                WHERE {vdc._alias} IS NOT NULL
	                GROUP BY 1
	                ORDER BY 1""",
                title="Computing the histogram heights",
                method="fetchall",
            )
            y = (
                [item[1] / float(count) for item in query_result]
                if (method.lower() == "density")
                else [item[1] for item in query_result]
            )
            x = [float(item[0]) + h / 2 for item in query_result]
            adj_width = 0.94 * h
            labels = None
        if pie:
            y.reverse()
            labels.reverse()
        self.data = {
            "x": x,
            "y": y,
            "width": h,
            "adj_width": adj_width,
            "is_categorical": is_categorical,
        }
        self.layout = {
            "labels": labels,
            "column": vdc._alias,
            "method": method,
            "method_of": method + f"({of})" if of else method,
            "of": of,
            "of_cat": vdc._parent[of].category() if of else None,
            "aggregate": clean_query(aggregate),
            "aggregate_fun": aggregate_fun,
            "is_standard": is_standard,
        }
        return None

    def _compute_outliers_params(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        max_nb_points: int = 1000,
        threshold: float = 3.0,
    ) -> None:
        if isinstance(columns, str):
            columns = [columns]
        columns = vdf._format_colnames(columns)
        aggregations = vdf.agg(
            func=["avg", "std", "min", "max"], columns=columns
        ).to_numpy()
        self.data = {
            "X": vdf[columns].sample(n=max_nb_points).to_numpy().astype(float),
            "avg": aggregations[:, 0],
            "std": aggregations[:, 1],
            "min": aggregations[:, 2],
            "max": aggregations[:, 3],
            "th": threshold,
        }
        self.layout = {
            "columns": copy.deepcopy(columns),
        }
        return None

    def _compute_aggregate(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        method: str = "count",
        of: Optional[str] = None,
    ) -> None:
        if isinstance(columns, str):
            columns = [columns]
        columns = vdf._format_colnames(columns)
        method, aggregate, aggregate_fun, is_standard = self._map_method(method, of)
        self._init_check(dim=len(columns), is_standard=is_standard)
        if method == "density":
            over = "/" + str(float(vdf.shape()[0]))
        else:
            over = ""
        X = np.array(
            _executeSQL(
                query=f"""
                SELECT
                    /*+LABEL('plotting._compute_aggregate')*/
                    {", ".join(columns)},
                    {aggregate}{over}
                FROM {vdf._genSQL()}
                GROUP BY {", ".join(columns)}""",
                title="Grouping all the elements for the Hexbin Plot",
                method="fetchall",
            )
        )
        self.data = {"X": X}
        self.layout = {
            "columns": copy.deepcopy(columns),
            "method": method,
            "method_of": method + f"({of})" if of else method,
            "of": of,
            "of_cat": vdf[of].category() if of else None,
            "aggregate": clean_query(aggregate),
            "aggregate_fun": aggregate_fun,
            "is_standard": is_standard,
        }
        return None

    def _compute_hists_params(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        by: Optional[str] = None,
        method: str = "density",
        of: Optional[str] = None,
        h: PythonNumber = 0.0,
        h_by: PythonNumber = 0.0,
        max_cardinality: int = 8,
        cat_priority: Union[None, PythonScalar, ArrayLike] = None,
    ) -> None:
        if isinstance(columns, str):
            columns = [columns]
        elif not (columns):
            columns = vdf.numcol()
        if not (columns):
            raise ValueError("No numerical columns found to compute the statistics.")
        columns, by = vdf._format_colnames(columns, by)
        method, aggregate, aggregate_fun, is_standard = self._map_method(method, of)
        self._init_check(dim=len(columns), is_standard=is_standard)
        if by and len(columns) == 1:
            column = columns[0]
            cols = [column]
            if not (h) or h <= 0:
                h = vdf[column].numh()
            data, categories = {"width": h}, []
            vdf_tmp = vdf[by].isin(cat_priority) if cat_priority else vdf.copy()
            if vdf_tmp[by].isnum():
                vdf_tmp[by].discretize(h=h_by)
            else:
                vdf_tmp[by].discretize(
                    k=max_cardinality, method="topk",
                )
            uniques = vdf_tmp[by].distinct()
            for idx, category in enumerate(uniques):
                self._compute_plot_params(
                    vdf_tmp[by].isin(category)[column],
                    method=method,
                    of=of,
                    max_cardinality=1,
                    h=h,
                )
                categories += [category]
                data[category] = copy.deepcopy(self.data)
        else:
            h_, categories = [], None
            if h == None or h <= 0:
                for idx, column in enumerate(columns):
                    h_ += [vdf[column].numh()]
                h = min(h_)
            data, cols = {"width": h}, []
            for idx, column in enumerate(columns):
                if vdf[column].isnum():
                    self._compute_plot_params(
                        vdf[column], method=method, of=of, max_cardinality=1, h=h
                    )
                    cols += [column]
                    data[column] = copy.deepcopy(self.data)
                else:
                    if vdf._vars["display"]["print_info"]:
                        warning_message = (
                            f"The Virtual Column {column} is not numerical."
                            " Its histogram will not be drawn."
                        )
                        warnings.warn(warning_message, Warning)
        self.data = data
        self.layout = {
            "columns": cols,
            "categories": categories,
            "by": by,
            "method": method,
            "method_of": method + f"({of})" if of else method,
            "of": of,
            "of_cat": vdf[of].category() if of else None,
            "aggregate": clean_query(aggregate),
            "aggregate_fun": aggregate_fun,
            "is_standard": is_standard,
            "has_category": bool(categories),
        }
        return None

    def _compute_range(
        self,
        vdf: "vDataFrame",
        order_by: str,
        columns: SQLColumns,
        q: tuple = (0.25, 0.75),
        order_by_start: PythonScalar = None,
        order_by_end: PythonScalar = None,
    ) -> None:
        if isinstance(columns, str):
            columns = [columns]
        columns, order_by = vdf._format_colnames(columns, order_by)
        expr = []
        for column in columns:
            expr += [
                f"APPROXIMATE_PERCENTILE({column} USING PARAMETERS percentile = {q[0]})",
                f"APPROXIMATE_MEDIAN({column})",
                f"APPROXIMATE_PERCENTILE({column} USING PARAMETERS percentile = {q[1]})",
            ]
        X = (
            vdf.between(column=order_by, start=order_by_start, end=order_by_end)
            .groupby(columns=[order_by], expr=expr,)
            .sort(columns=[order_by])
            .to_numpy()
        )
        self.data = {
            "x": self._parse_datetime(X[:, 0]),
            "Y": X[:, 1:].astype(float),
        }
        self.layout = {
            "columns": columns,
            "order_by": order_by,
        }
        return None

    def _compute_rollup(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        method: str = "density",
        of: Optional[str] = None,
        max_cardinality: Union[int, tuple, list] = None,
        h: Union[int, tuple, list] = None,
    ) -> None:
        if isinstance(columns, str):
            columns = [columns]
        method, aggregate, aggregate_fun, is_standard = self._map_method(method, of)
        n = len(columns)
        if isinstance(h, (int, float, type(None))):
            h = (h,) * n
        if isinstance(max_cardinality, (int, float, type(None))):
            if max_cardinality == None:
                max_cardinality = (6,) * n
            else:
                max_cardinality = (max_cardinality,) * n
        vdf_tmp = vdf[columns]
        for idx, column in enumerate(columns):
            vdf_tmp[column].discretize(h=h[idx])
            vdf_tmp[column].discretize(method="topk", k=max_cardinality[idx])
        groups = []
        for i in range(0, n):
            groups += [
                vdf_tmp.groupby(columns[: n - i], [aggregate])
                .sort(columns[: n - i])
                .to_numpy()
                .T
            ]
        self.data = {"groups": np.array(groups, dtype=object)}
        self.layout = {
            "columns": copy.deepcopy(columns),
            "method": method,
            "method_of": method + f"({of})" if of else method,
            "of": of,
            "of_cat": vdf[of].category() if of else None,
            "aggregate": clean_query(aggregate),
            "aggregate_fun": aggregate_fun,
            "is_standard": is_standard,
        }
        return None

    def _compute_statistics(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        by: Optional[str] = None,
        q: tuple = (0.25, 0.75),
        h: PythonNumber = 0.0,
        max_cardinality: int = 8,
        cat_priority: Union[None, PythonScalar, ArrayLike] = None,
        whis: float = 1.5,
        max_nb_fliers: int = 30,
    ) -> None:
        if not (0 <= q[0] < 0.5 < q[1] <= 1):
            raise ValueError
        if isinstance(columns, str):
            columns = [columns]
        elif not (columns):
            columns = vdf.numcol()
        if not (columns):
            raise ValueError("No numerical columns found to compute the statistics.")
        columns, by = vdf._format_colnames(columns, by)
        if len(columns) == 1 and (by):
            expr = [
                f"APPROXIMATE_PERCENTILE({columns[0]} USING PARAMETERS percentile = {q[0]})",
                f"APPROXIMATE_MEDIAN({columns[0]})",
                f"APPROXIMATE_PERCENTILE({columns[0]} USING PARAMETERS percentile = {q[1]})",
            ]
            if vdf[by].isnum():
                _by = vdf[by].discretize(h=h, return_enum_trans=True)
                is_num_transf = True
            else:
                _by = vdf[by].discretize(
                    k=max_cardinality, method="topk", return_enum_trans=True
                )
                is_num_transf = False
            _by = _by[0].replace("{}", by) + f" AS {by}"
            vdf_tmp = vdf.copy()
            if cat_priority:
                vdf_tmp = vdf_tmp[by].isin(cat_priority)
            vdf_tmp = vdf_tmp[[_by] + columns]
            X = vdf_tmp.groupby(columns=[by], expr=expr,).sort(columns=[by]).to_numpy()
            if is_num_transf:
                try:
                    X_num = np.array(
                        [
                            float(x[1:].split(";")[0]) if isinstance(x, str) else x
                            for x in X[:, 0]
                        ]
                    ).astype(float)
                except ValueError:
                    X_num = np.array(
                        [float(x) if isinstance(x, str) else x for x in X[:, 0]]
                    ).astype(float)
                X = X[X_num.argsort()]
            self.layout = {
                "x_label": by,
                "y_label": columns[0],
                "labels": X[:, 0],
                "has_category": True,
            }
            X = X[:, 1:].astype(float)
        else:
            self.layout = {
                "labels": copy.deepcopy(columns),
                "has_category": False,
            }
            X = vdf.quantile(
                q=[q[0], 0.5, q[1]], columns=columns, approx=True
            ).to_numpy()
        X = np.transpose(X)
        IQR = X[2] - X[0]
        X = np.vstack([X, X[2] + whis * IQR])
        X = np.vstack([X[0] - whis * IQR, X])
        n, m = X.shape
        fliers = []
        for i in range(m):
            if max_nb_fliers > 0:
                if self.layout["has_category"]:
                    f, k = vdf_tmp[by].isin(self.layout["labels"][i]), 0
                else:
                    f, k = vdf, i
                f = f[[columns[k]]].search(
                    f"{columns[k]} < {X[:, i][0]} OR {columns[k]} > {X[:, i][-1]}"
                )
                try:
                    fliers += [f.sample(n=max_nb_fliers).to_numpy()[:, 0]]
                except ZeroDivisionError:
                    fliers += [np.array([])]
            else:
                fliers += [np.array([])]
        self.data = {
            "X": X,
            "fliers": fliers,
            "whis": whis,
        }

    def _filter_line(
        self,
        vdf: "vDataFrame",
        order_by: str,
        columns: SQLColumns,
        order_by_start: PythonScalar = None,
        order_by_end: PythonScalar = None,
    ) -> None:
        if isinstance(columns, str):
            columns = [columns]
        elif not (columns):
            columns = vdf.numcol()
        columns, order_by = vdf._format_colnames(columns, order_by)
        X = (
            vdf.between(
                column=order_by, start=order_by_start, end=order_by_end, inplace=False
            )[[order_by] + columns]
            .sort(columns=[order_by])
            .to_numpy()
        )
        if not (vdf[columns[-1]].isnum()):
            self.data = {
                "x": X[:, 0],
                "Y": X[:, 1:-1].astype(float),
                "z": X[:, -1],
            }
            has_category = True
        else:
            self.data = {
                "x": X[:, 0],
                "Y": X[:, 1:],
            }
            has_category = False
        self.layout = {
            "columns": columns,
            "order_by": order_by,
            "has_category": has_category,
        }
        return None

    def _compute_pivot_table(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        method: str = "count",
        of: Optional[str] = None,
        h: tuple[Optional[float], Optional[float]] = (None, None),
        max_cardinality: tuple[int, int] = (20, 20),
        fill_none: float = 0.0,
    ) -> None:
        """
        Draws a pivot table using the Matplotlib API.
        """
        other_columns = ""
        method, aggregate, aggregate_fun, is_standard = self._map_method(method, of)
        self._init_check(dim=len(columns), is_standard=is_standard)
        if not (is_standard):
            other_columns = ", " + ", ".join(vdf.get_columns(exclude_columns=columns))
        if isinstance(columns, str):
            columns = [columns]
        columns, of = vdf._format_colnames(columns, of)
        is_column_date = [False, False]
        timestampadd = ["", ""]
        matrix = []
        for idx, column in enumerate(columns):
            is_numeric = vdf[column].isnum() and (vdf[column].nunique(True) > 2)
            is_date = vdf[column].isdate()
            where = []
            if is_numeric:
                if h[idx] == None:
                    interval = vdf[column].numh()
                    if interval > 0.01:
                        interval = round(interval, 2)
                    elif interval > 0.0001:
                        interval = round(interval, 4)
                    elif interval > 0.000001:
                        interval = round(interval, 6)
                    if vdf[column].category() == "int":
                        interval = int(max(math.floor(interval), 1))
                else:
                    interval = h[idx]
                if vdf[column].category() == "int":
                    floor_end = "-1"
                    interval = int(max(math.floor(interval), 1))
                else:
                    floor_end = ""
                expr = f"""'[' 
                          || FLOOR({column} 
                                 / {interval}) 
                                 * {interval} 
                          || ';' 
                          || (FLOOR({column} 
                                  / {interval}) 
                                  * {interval} 
                                  + {interval}{floor_end}) 
                          || ']'"""
                if (interval > 1) or (vdf[column].category() == "float"):
                    matrix += [expr]
                else:
                    matrix += [f"FLOOR({column}) || ''"]
                order_by = f"""ORDER BY MIN(FLOOR({column} 
                                          / {interval}) * {interval}) ASC"""
                where += [f"{column} IS NOT NULL"]
            elif is_date:
                if h[idx] == None:
                    interval = vdf[column].numh()
                else:
                    interval = max(math.floor(h[idx]), 1)
                min_date = vdf[column].min()
                matrix += [
                    f"""FLOOR(DATEDIFF('second',
                                       '{min_date}',
                                       {column})
                            / {interval}) * {interval}"""
                ]
                is_column_date[idx] = True
                sql = f"""TIMESTAMPADD('second', {columns[idx]}::int, 
                                                 '{min_date}'::timestamp)"""
                timestampadd[idx] = sql
                order_by = "ORDER BY 1 ASC"
                where += [f"{column} IS NOT NULL"]
            else:
                matrix += [column]
                order_by = "ORDER BY 1 ASC"
                distinct = vdf[column].topk(max_cardinality[idx]).values["index"]
                distinct = ["'" + str(c).replace("'", "''") + "'" for c in distinct]
                if len(distinct) < max_cardinality[idx]:
                    cast = to_varchar(vdf[column].category(), column)
                    where += [f"({cast} IN ({', '.join(distinct)}))"]
                else:
                    where += [f"({column} IS NOT NULL)"]
        where = f" WHERE {' AND '.join(where)}"
        over = "/" + str(vdf.shape()[0]) if (method == "density") else ""
        if len(columns) == 1:
            cast = to_varchar(vdf[columns[0]].category(), matrix[-1])
            res = TableSample.read_sql(
                query=f"""
                    SELECT 
                        {cast} AS {columns[0]},
                        {aggregate}{over} 
                    FROM {vdf._genSQL()}
                    {where}
                    GROUP BY 1 {order_by}"""
            ).to_numpy()
            matrix = res[:, 1].astype(float)
            x_labels = list(res[:, 0])
            y_labels = [method]
        else:
            aggr = f", {of}" if (of) else ""
            cols, cast = [], []
            for i in range(2):
                if is_column_date[0]:
                    cols += [f"{timestampadd[i]} AS {columns[i]}"]
                else:
                    cols += [columns[i]]
                cast += [to_varchar(vdf[columns[i]].category(), columns[i])]
            query_result = _executeSQL(
                query=f"""
                    SELECT 
                        /*+LABEL('plotting._matplotlib.pivot_table')*/
                        {cast[0]} AS {columns[0]},
                        {cast[1]} AS {columns[1]},
                        {aggregate}{over}
                    FROM (SELECT 
                              {cols[0]},
                              {cols[1]}
                              {aggr}
                              {other_columns} 
                          FROM 
                              (SELECT 
                                  {matrix[0]} AS {columns[0]},
                                  {matrix[1]} AS {columns[1]}
                                  {aggr}
                                  {other_columns} 
                               FROM {vdf._genSQL()}{where}) 
                               pivot_table) pivot_table_date
                    WHERE {columns[0]} IS NOT NULL 
                      AND {columns[1]} IS NOT NULL
                    GROUP BY {columns[0]}, {columns[1]}
                    ORDER BY {columns[0]}, {columns[1]} ASC""",
                title="Grouping the features to compute the pivot table",
                method="fetchall",
            )
            matrix_categories = []
            for i in range(2):
                L = list(set([str(item[i]) for item in query_result]))
                L.sort()
                try:
                    try:
                        order = []
                        for item in L:
                            order += [float(item.split(";")[0].split("[")[1])]
                    except:
                        order = [float(item) for item in L]
                    L = [x for _, x in sorted(zip(order, L))]
                except:
                    pass
                matrix_categories += [copy.deepcopy(L)]
            x_labels, y_labels = matrix_categories
            X = np.array([[fill_none for item in y_labels] for item in x_labels])
            for item in query_result:
                i = x_labels.index(str(item[0]))
                j = y_labels.index(str(item[1]))
                X[i][j] = item[2]
        self.data = {
            "X": X,
        }
        self.layout = {
            "x_labels": x_labels,
            "y_labels": y_labels,
            "vmax": None,
            "vmin": None,
            "columns": copy.deepcopy(columns),
            "method": method,
            "method_of": method + f"({of})" if of else method,
            "of": of,
            "of_cat": vdf[of].category() if of else None,
            "aggregate": clean_query(aggregate),
            "aggregate_fun": aggregate_fun,
            "is_standard": is_standard,
        }
        return None

    def _compute_scatter_matrix(
        self, vdf: "vDataFrame", columns: SQLColumns, max_nb_points: int = 20000,
    ) -> None:
        if isinstance(columns, str):
            columns = [columns]
        elif not (columns):
            columns = vdf.numcol()
        columns = vdf._format_colnames(columns)
        n = len(columns)
        data = {
            "scatter": {"X": vdf[columns].sample(n=max_nb_points).to_numpy()},
            "hist": {},
        }
        for i in range(n):
            for j in range(n):
                if columns[i] == columns[j]:
                    self._compute_plot_params(
                        vdf[columns[i]], method="density", max_cardinality=1
                    )
                    data["hist"][columns[i]] = copy.deepcopy(self.data)
        self.data = data
        self.layout = {
            "columns": copy.deepcopy(columns),
        }
        return None

    def _sample(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        size_bubble_col: Optional[str] = None,
        catcol: Optional[str] = None,
        cmap_col: Optional[str] = None,
        max_nb_points: int = 20000,
        h: PythonNumber = 0.0,
        max_cardinality: int = 8,
        cat_priority: Union[None, PythonScalar, ArrayLike] = None,
    ) -> None:
        if isinstance(columns, str):
            columns = [columns]
        columns = vdf._format_colnames(columns)
        cols_to_select = copy.deepcopy(columns)
        vdf_tmp = vdf.copy()
        has_category, has_cmap, has_size = False, False, False
        if max_nb_points > 0:
            if size_bubble_col != None:
                cols_to_select += [vdf._format_colnames(size_bubble_col)]
                has_size = True
            if catcol != None:
                has_category = True
                catcol = vdf._format_colnames(catcol)
                if vdf[catcol].isnum():
                    cols_to_select += [
                        vdf[catcol]
                        .discretize(h=h, return_enum_trans=True)[0]
                        .replace("{}", catcol)
                        + f" AS {catcol}"
                    ]
                else:
                    cols_to_select += [
                        vdf[catcol]
                        .discretize(
                            k=max_cardinality, method="topk", return_enum_trans=True
                        )[0]
                        .replace("{}", catcol)
                        + f" AS {catcol}"
                    ]
                if cat_priority:
                    vdf_tmp = vdf_tmp[catcol].isin(cat_priority)
            elif cmap_col != None:
                cols_to_select += [vdf._format_colnames(cmap_col)]
                has_cmap = True
            X = vdf_tmp[cols_to_select].sample(n=max_nb_points).to_numpy()
            n = len(columns)
            if len(X) > 0:
                X = X[:max_nb_points, :n].astype(float)
        else:
            X = np.array([])
        self.data = {"X": X, "s": None, "c": None}
        self.layout = {
            "columns": columns,
            "size": size_bubble_col,
            "c": catcol if (catcol != None) else cmap_col,
            "has_category": has_category,
            "has_cmap": has_cmap,
            "has_size": has_size,
        }
        if (size_bubble_col != None) and (max_nb_points > 0):
            self.data["s"] = X[:, n].astype(float)
        if ((catcol != None) or (cmap_col != None)) and (max_nb_points > 0):
            self.data["c"] = X[:, -1]
        return None
