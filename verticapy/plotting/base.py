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
import random
import warnings
from typing import Callable, Literal, Optional, Union, TYPE_CHECKING

import numpy as np

import matplotlib.colors as plt_colors

from vertica_python.errors import QueryError

import verticapy._config.config as conf
from verticapy._typing import (
    ArrayLike,
    NoneType,
    PythonNumber,
    PythonScalar,
    SQLColumns,
)
from verticapy._utils._object import create_new_vdf
from verticapy._utils._sql._cast import to_varchar
from verticapy._utils._sql._format import clean_query, format_type, quote_ident
from verticapy._utils._sql._sys import _executeSQL

from verticapy.core.string_sql.base import StringSQL
from verticapy.core.tablesample.base import TableSample

from verticapy.plotting.sql import PlottingBaseSQL

if conf.get_import_success("dateutil"):
    from dateutil.parser import parse

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame, vDataColumn

"""
Colors Options: They are used when drawing graphics.
"""

COLORS_OPTIONS: dict[str, list] = {
    "rgb": ["red", "green", "blue", "orange", "yellow", "gray"],
    "sunset": ["#36688D", "#F3CD05", "#F49F05", "#F18904", "#BDA589"],
    "retro": ["#A7414A", "#282726", "#6A8A82", "#A37C27", "#563838"],
    "shimbg": ["#0444BF", "#0584F2", "#0AAFF1", "#EDF259", "#A79674"],
    "swamp": ["#6465A5", "#6975A6", "#F3E96B", "#F28A30", "#F05837"],
    "med": ["#ABA6BF", "#595775", "#583E2E", "#F1E0D6", "#BF9887"],
    "orchid": ["#192E5B", "#1D65A6", "#72A2C0", "#00743F", "#F2A104"],
    "magenta": ["#DAA2DA", "#DBB4DA", "#DE8CF0", "#BED905", "#93A806"],
    "orange": ["#A3586D", "#5C4A72", "#F3B05A", "#F4874B", "#F46A4E"],
    "vintage": ["#80ADD7", "#0ABDA0", "#EBF2EA", "#D4DCA9", "#BF9D7A"],
    "vivid": ["#C0334D", "#D6618F", "#F3D4A0", "#F1931B", "#8F715B"],
    "berries": ["#BB1924", "#EE6C81", "#F092A5", "#777CA8", "#AFBADC"],
    "refreshing": ["#003D73", "#0878A4", "#1ECFD6", "#EDD170", "#C05640"],
    "summer": ["#728CA3", "#73C0F4", "#E6EFF3", "#F3E4C6", "#8F4F06"],
    "tropical": ["#7B8937", "#6B7436", "#F4D9C1", "#D72F01", "#F09E8C"],
    "india": ["#F1445B", "#65734B", "#94A453", "#D9C3B1", "#F03625"],
    "old": ["#FE5016", "#263133", "#0073E7", "#FDE159", "#33C180", "#FF454F"],
    "default": [
        "#1A6AFF",
        "#000000",
        "#9C9C9C",
        "#FE5016",
        "#FDE159",
        "#33C180",
        "#4E2A84",
        "#00008B",
    ],
}


def color_validator(val: Union[str, list, None]) -> Literal[True]:
    """
    Validator used to check and change the colors.
    """
    if (isinstance(val, str) and val in COLORS_OPTIONS) or isinstance(
        val, (list, NoneType)
    ):
        return True
    else:
        raise ValueError(
            "The option must be a list of colors, None, or in"
            f" [{'|'.join(COLORS_OPTIONS)}]"
        )


colors_option = conf.Option("colors", None, "", color_validator, COLORS_OPTIONS)
conf.register_option(colors_option)

"""
Plotting Base Class.
"""


class PlottingBase(PlottingBaseSQL):
    # Properties.

    @property
    def _compute_method(self) -> Literal[None]:
        """Must be overridden in child class"""

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
        if "misc_layout" in kwds:
            misc_layout = copy.deepcopy(kwds["misc_layout"])
            del kwds["misc_layout"]
        else:
            misc_layout = {}
        if "query" in kwds:
            functions = {
                "1D": self._compute_plot_params_sql,
                "2D": self._compute_pivot_table_sql,
                "aggregate": self._compute_aggregate_sql,
                "candle": self._compute_candle_aggregate_sql,
                # "contour": self._compute_contour_grid_sql, NOT POSSIBLE
                "describe": self._compute_statistics_sql,
                "hist": self._compute_hists_params_sql,
                "line": self._filter_line_sql,
                # "line_bubble": self._filter_line_animated_scatter_sql, TOO MUCH COMPLEX
                "matrix": self._compute_scatter_matrix_sql,
                "outliers": self._compute_outliers_params_sql,
                "range": self._compute_range_sql,
                "rollup": self._compute_rollup_sql,
                "sample": self._sample_sql,
            }
            functions[self._compute_method](*args, **kwds)
        elif "data" not in kwds or "layout" not in kwds:
            functions = {
                "1D": self._compute_plot_params,
                "2D": self._compute_pivot_table,
                "aggregate": self._compute_aggregate,
                "candle": self._compute_candle_aggregate,
                "contour": self._compute_contour_grid,
                "describe": self._compute_statistics,
                "hist": self._compute_hists_params,
                "line": self._filter_line,
                "line_bubble": self._filter_line_animated_scatter,
                "matrix": self._compute_scatter_matrix,
                "outliers": self._compute_outliers_params,
                "range": self._compute_range,
                "rollup": self._compute_rollup,
                "sample": self._sample,
                "tsa": self._compute_tsa,
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
        if hasattr(self, "layout"):
            self.layout = {
                **self.layout,
                **misc_layout,
            }
        if hasattr(self, "_kind") and self._kind == "importance":
            self._compute_importance()
        self._init_style()

    def _init_check(self, dim: int, is_standard: bool) -> None:
        lower, upper = self._dimension_bounds
        if not lower <= dim <= upper:
            if lower == upper:
                message = f"exactly {lower}"
            else:
                message = f"between {lower} and {upper}."
            raise ValueError(
                f"The number of columns to draw the plot must be {message}. Found {dim}."
            )
        if self._only_standard and not is_standard:
            raise ValueError(
                f"When drawing {self._kind} {self._category}s, the parameter "
                "'method' can not represent a customized aggregation."
            )

    # Columns formatting methods.

    def _clean_quotes(self, columns: SQLColumns) -> SQLColumns:
        if isinstance(columns, NoneType):
            return None
        elif isinstance(columns, str):
            return quote_ident(columns)[1:-1]
        else:
            return [self._clean_quotes(col) for col in columns]

    # Styling Methods.

    def _init_style(self) -> None:
        """Must be overridden in child class"""
        self.init_style = {}

    def _get_final_color(
        self,
        style_kwargs: dict,
        idx: int = 0,
    ) -> str:
        for key in ["colors", "color", "c"]:
            if key in style_kwargs:
                if isinstance(style_kwargs[key], list):
                    n = len(style_kwargs[key])
                    return style_kwargs[key][idx % n]
                elif idx == 0:
                    return style_kwargs[key]
        return self.get_colors(idx=idx)

    def _get_final_style_kwargs(
        self,
        style_kwargs: dict,
        idx: int,
    ) -> dict:
        kwargs = copy.deepcopy(style_kwargs)
        for key in ["colors", "color", "c"]:
            if key in kwargs:
                del kwargs[key]
        kwargs["color"] = self._get_final_color(
            style_kwargs=style_kwargs,
            idx=idx,
        )
        return kwargs

    def _fix_color_style_kwargs(
        self,
        style_kwargs: dict,
    ) -> dict:
        if "colors" in style_kwargs:
            style_kwargs["color"] = style_kwargs["colors"]
            del style_kwargs["colors"]
        return style_kwargs

    def get_colors(
        self, d: Optional[dict] = None, idx: Optional[int] = None
    ) -> Union[list, str]:
        """
        If a color or list of colours is available in the
        input dictionary, return it. Otherwise, this function
        returns the current module str or list of colors.
        """
        d = format_type(d, dtype=dict)
        if "color" in d:
            if isinstance(d["color"], str):
                return d["color"]
            else:
                if isinstance(idx, NoneType):
                    idx = 0
                return d["color"][idx % len(d["color"])]
        elif isinstance(idx, NoneType):
            if not conf.get_option("colors"):
                colors = COLORS_OPTIONS["default"]
                all_colors = [plt_colors.cnames[key] for key in plt_colors.cnames]
                random.shuffle(all_colors)
                for c in all_colors:
                    if c not in colors:
                        colors += [c]
                return colors
            else:
                return conf.get_option("colors")
        else:
            colors = self.get_colors()
            return colors[idx % len(colors)]

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
        if not color:
            args1 = args + [["#FFFFFF", self.get_colors(idx=0)]]
            args2 = args + [[self.get_colors(idx=1), "#FFFFFF", self.get_colors(idx=0)]]
            cm1 = cmap_from_list(*args1, **kwargs)
            cm2 = cmap_from_list(*args2, **kwargs)
            if isinstance(idx, NoneType):
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
            raise ValueError(
                "Parameter 'of' must be empty when using customized aggregations."
            )
        if (
            (method in ["avg", "min", "max", "sum"])
            or (isinstance(method, str) and method.endswith("%"))
        ) and (of):
            if method in ["avg", "min", "max", "sum"]:
                aggregate = f"{method.upper()}({quote_ident(of)})"
                fun = fun_map[method]
            elif isinstance(method, str) and method.endswith("%"):
                q = float(method[0:-1]) / 100
                aggregate = f"""
                    APPROXIMATE_PERCENTILE({quote_ident(of)} 
                        USING PARAMETERS
                        percentile = {q})"""

                def fun(x: ArrayLike) -> float:
                    return np.quantile(x, q)

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
            raise ValueError(
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
    def _update_dict(
        d1: dict,
        d2: dict,
        color_idx: int = 0,
    ) -> dict:
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
                    d["color"] = list(d2["color"])
                else:
                    d["color"] = d2["color"][color_idx % len(d2["color"])]
            else:
                d[elem] = d2[elem]
        return d

    # Attributes Computations.

    # Features Importance.

    def _compute_importance(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coef_names = np.array(self.layout["columns"])
        importances = self.data["importance"].astype(float)
        signs = np.sign(importances)
        importances = abs(importances)
        coef_names = coef_names[importances != np.nan]
        signs = signs[importances != np.nan]
        importances = importances[importances != np.nan]
        importances = importances[coef_names != None]
        signs = signs[coef_names != None]
        coef_names = coef_names[coef_names != None]
        importances, coef_names, signs = zip(
            *sorted(zip(importances, coef_names, signs))
        )
        self.data["importance"] = np.array(importances)
        self.data["signs"] = np.array(signs)
        self.layout["columns"] = coef_names

    # 1D AGG Graphics: BAR / PIE ...

    def _compute_plot_params(
        self,
        vdc: "vDataColumn",
        method: str = "density",
        of: Optional[str] = None,
        max_cardinality: int = 6,
        nbins: int = 0,
        h: float = 0.0,
        pie: bool = False,
        bargap: float = 0.06,
    ) -> None:
        """
        Computes the aggregations needed to draw a 1D graphic.
        """
        if not 0.0 < bargap <= 1.0:
            raise ValueError("Parameter 'bargap' must be between 0 and 1.")
        other_columns = ""
        of = vdc._parent.format_colnames(of)
        method, aggregate, aggregate_fun, is_standard = self._map_method(method, of)
        if not is_standard:
            other_columns = ", " + ", ".join(
                vdc._parent.get_columns(exclude_columns=[vdc._alias])
            )
        # depending on the cardinality, the type, the vDataColumn
        # can be treated as categorical or not
        try:
            cardinality = vdc.nunique(True)
        except QueryError:
            cardinality = vdc.nunique(False)
        count = vdc._parent.shape()[0]
        is_numeric = vdc.isnum() and not vdc.isbool()
        is_date = vdc.isdate()
        is_bool = vdc.isbool()
        cast = "::int" if is_bool else ""
        is_categorical = False
        # case when categorical
        if (((cardinality <= max_cardinality) or not is_numeric) or pie) and not (
            is_date
        ):
            if ((is_numeric) and not pie) or (is_bool):
                query = f"""
                    SELECT 
                        {vdc},
                        {aggregate}
                    FROM {vdc._parent} 
                    WHERE {vdc} IS NOT NULL 
                    GROUP BY {vdc} 
                    ORDER BY {vdc} ASC 
                    LIMIT {max_cardinality}"""
            else:
                table = vdc._parent.current_relation()
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
                        COALESCE({cast_alias}, 'NULL') AS {vdc},
                        {aggregate}
                     FROM {table} 
                     GROUP BY 1
                     ORDER BY 2 DESC 
                     LIMIT {max_cardinality})"""
                if cardinality > max_cardinality:
                    query += f"""
                        UNION 
                        (SELECT 
                            'Others',
                            {aggregate} 
                         FROM {table}
                         WHERE {vdc} NOT IN
                            (SELECT 
                                {vdc} 
                            FROM
                             (SELECT 
                                COALESCE({cast_alias}, 'NULL') AS {vdc},
                                {aggregate}
                             FROM {table} 
                             GROUP BY 1
                             ORDER BY 2 DESC 
                             LIMIT {max_cardinality}) x))"""
            query_result = _executeSQL(
                query=query, title="Computing the histogram heights", method="fetchall"
            )
            y = (
                [
                    item[1] / float(count) if not isinstance(item[1], NoneType) else 0
                    for item in query_result
                ]
                if (method.lower() == "density")
                else [
                    item[1] if not isinstance(item[1], NoneType) else 0
                    for item in query_result
                ]
            )
            x = [0.4 * i + 0.2 for i in range(0, len(y))]
            adj_width = 0.4 * (1 - bargap)
            labels = [item[0] for item in query_result]
            is_categorical = True
        # case when date
        elif is_date:
            if (isinstance(h, NoneType) or (h <= 0)) and (
                isinstance(nbins, NoneType) or (nbins <= 0)
            ):
                h = vdc.numh()
            elif nbins > 0:
                query_result = _executeSQL(
                    query=f"""
                        SELECT 
                            /*+LABEL('plotting._matplotlib._compute_plot_params')*/
                            DATEDIFF('second', MIN({vdc}), MAX({vdc}))
                        FROM {vdc._parent}""",
                    title="Computing the histogram interval",
                    method="fetchrow",
                )
                h = float(query_result[0]) / nbins
            min_date = vdc.min()
            converted_date = f"DATEDIFF('second', '{min_date}', {vdc})"
            query_result = _executeSQL(
                query=f"""
                    SELECT 
                        /*+LABEL('plotting._matplotlib._compute_plot_params')*/
                        FLOOR({converted_date} / {h}) * {h}, 
                        {aggregate} 
                    FROM {vdc._parent}
                    WHERE {vdc} IS NOT NULL 
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
            adj_width = (1.0 - bargap) * h
            labels = [item[0] for item in query_result]
            labels.sort()
            is_categorical = True
        # case when numerical
        else:
            if (isinstance(h, NoneType) or (h <= 0)) and (
                isinstance(nbins, NoneType) or (nbins <= 0)
            ):
                h = vdc.numh()
            elif nbins > 0:
                h = float(vdc.max() - vdc.min()) / nbins
            if (vdc.ctype == "int") or (h == 0):
                h = max(1.0, h)
            query_result = _executeSQL(
                query=f"""
                    SELECT
                        /*+LABEL('plotting._matplotlib._compute_plot_params')*/
                        FLOOR({vdc}{cast} / {h}) * {h},
                        {aggregate} 
                    FROM {vdc._parent}
                    WHERE {vdc} IS NOT NULL
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
            adj_width = (1.0 - bargap) * h
            labels = [xi - round(h / 2, 10) for xi in x]
            labels = [(li, li + h) for li in labels]
        if pie:
            y.reverse()
            labels.reverse()
        self.data = {
            "x": x,
            "y": y,
            "width": h,
            "adj_width": adj_width,
            "bargap": bargap,
            "is_categorical": is_categorical,
        }
        self.layout = {
            "labels": [li if not isinstance(li, NoneType) else "None" for li in labels],
            "column": self._clean_quotes(vdc._alias),
            "method": method,
            "method_of": method + f"({of})" if of else method,
            "of": self._clean_quotes(of),
            "of_cat": vdc._parent[of].category() if of else None,
            "aggregate": clean_query(aggregate),
            "aggregate_fun": aggregate_fun,
            "is_standard": is_standard,
        }

    # HISTOGRAMS

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
        if not columns:
            columns = vdf.numcol()
        else:
            columns = format_type(columns, dtype=list)
        columns_ = []
        for col in columns:
            if vdf[col].isnum() and not (vdf[col].isbool()):
                columns_ += [col]
            elif conf.get_option("print_info"):
                warning_message = (
                    f"The Virtual Column {col} is not numerical."
                    " Its histogram will not be drawn."
                )
                warnings.warn(warning_message, Warning)
        if not columns_:
            raise ValueError("No quantitative feature to plot.")
        columns_, by = vdf.format_colnames(columns_, by)
        method, aggregate, aggregate_fun, is_standard = self._map_method(method, of)
        self._init_check(dim=len(columns_), is_standard=is_standard)
        if by and len(columns_) == 1:
            column = columns_[0]
            cols = [column]
            if not h or h <= 0:
                h = vdf[column].numh()
            data, categories = {"width": h}, []
            vdf_tmp = vdf[by].isin(cat_priority) if cat_priority else vdf.copy()
            if vdf_tmp[by].isnum():
                vdf_tmp[by].discretize(h=h_by)
            else:
                vdf_tmp[by].discretize(
                    k=max_cardinality,
                    method="topk",
                )
            uniques = vdf_tmp[by].distinct()
            for category in uniques:
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
            if isinstance(h, NoneType) or h <= 0:
                for idx, column in enumerate(columns_):
                    h_ += [vdf[column].numh()]
                h = min(h_)
            data, cols = {"width": h}, []
            for idx, column in enumerate(columns_):
                if vdf[column].isnum():
                    self._compute_plot_params(
                        vdf[column], method=method, of=of, max_cardinality=1, h=h
                    )
                    cols += [column]
                    data[self._clean_quotes(column)] = copy.deepcopy(self.data)
        self.data = data
        self.layout = {
            "columns": self._clean_quotes(cols),
            "categories": categories,
            "by": self._clean_quotes(by),
            "method": method,
            "method_of": method + f"({of})" if of else method,
            "of": self._clean_quotes(of),
            "of_cat": vdf[of].category() if of else None,
            "aggregate": clean_query(aggregate),
            "aggregate_fun": aggregate_fun,
            "is_standard": is_standard,
            "has_category": bool(categories),
        }

    # BOXPLOTS

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
        if not 0 <= q[0] < 0.5 < q[1] <= 1:
            raise ValueError
        if not columns:
            columns = vdf.numcol()
        else:
            columns = format_type(columns, dtype=list)
        if not columns:
            raise ValueError("No numerical columns found to compute the statistics.")
        columns, by = vdf.format_colnames(columns, by)
        if len(columns) == 1 and (by):
            expr = [
                f"MIN({columns[0]})",
                f"APPROXIMATE_PERCENTILE({columns[0]} USING PARAMETERS percentile = {q[0]})",
                f"APPROXIMATE_MEDIAN({columns[0]})",
                f"APPROXIMATE_PERCENTILE({columns[0]} USING PARAMETERS percentile = {q[1]})",
                f"MAX({columns[0]})",
            ]
            if vdf[by].isnum() and not (vdf[by].isbool()):
                _by = vdf[by].discretize(h=h, return_enum_trans=True)
                is_num_transf = True
            elif vdf[by].isbool():
                _by = ("{}::varchar", "varchar", "text")
                is_num_transf = False
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
            X = (
                vdf_tmp.groupby(
                    columns=[by],
                    expr=expr,
                )
                .sort(columns=[by])
                .to_numpy()
            )
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
                "x_label": self._clean_quotes(by),
                "y_label": self._clean_quotes(columns[0]),
                "labels": X[:, 0],
                "has_category": True,
            }
            X = X[:, 1:].astype(float)
        else:
            self.layout = {
                "x_label": None,
                "y_label": None,
                "labels": self._clean_quotes(columns),
                "has_category": False,
            }
            X = vdf.quantile(
                q=[0.0, q[0], 0.5, q[1], 1.0], columns=columns, approx=True
            ).to_numpy()
        X = np.transpose(X)
        Xmin, Xmax = X[0], X[-1]
        X = np.delete(X, 0, 0)
        X = np.delete(X, -1, 0)
        IQR = X[2] - X[0]
        Xmax_adj = X[2] + whis * IQR
        Xmin_adj = X[0] - whis * IQR
        Xmaxf = np.array((Xmax, Xmax_adj)).min(axis=0)
        Xminf = np.array((Xmin, Xmin_adj)).max(axis=0)
        X = np.vstack((X, Xmaxf))
        X = np.vstack((Xminf, X))
        m = X.shape[1]
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
                    fliers += [f.sample(n=max_nb_fliers).to_numpy()[:, 0].astype(float)]
                except ZeroDivisionError:
                    fliers += [np.array([])]
            else:
                fliers += [np.array([])]
        self.data = {
            "X": X,
            "xmin": Xmin,
            "xmax": Xmax,
            "fliers": fliers,
            "whis": whis,
            "q": q,
        }

    # 2D AGG Graphics: BAR / PIE / HEATMAP / CONTOUR / HEXBIN ...

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
        Computes a pivot table.
        """
        other_columns = ""
        method, aggregate, aggregate_fun, is_standard = self._map_method(method, of)
        self._init_check(dim=len(columns), is_standard=is_standard)
        if not is_standard:
            other_columns = ", " + ", ".join(vdf.get_columns(exclude_columns=columns))
        columns = format_type(columns, dtype=list)
        columns, of = vdf.format_colnames(columns, of)
        is_column_date = [False, False]
        timestampadd = ["", ""]
        matrix = []
        for idx, column in enumerate(columns):
            is_numeric = vdf[column].isnum() and (vdf[column].nunique(True) > 2)
            is_date = vdf[column].isdate()
            cast = "::int" if vdf[column].isbool() else ""
            where = []
            if is_numeric:
                if isinstance(h[idx], NoneType):
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
                          || FLOOR({column}{cast}
                                 / {interval}) 
                                 * {interval} 
                          || ';' 
                          || (FLOOR({column}{cast}
                                  / {interval}) 
                                  * {interval} 
                                  + {interval}{floor_end}) 
                          || ']'"""
                if (interval > 1) or (vdf[column].category() == "float"):
                    matrix += [expr]
                else:
                    matrix += [f"FLOOR({column}{cast}) || ''"]
                order_by = f"""ORDER BY MIN(FLOOR({column}{cast} 
                                          / {interval}) * {interval}) ASC"""
                where += [f"{column} IS NOT NULL"]
            elif is_date:
                if isinstance(h[idx], NoneType):
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
                if 0 < len(distinct) < max_cardinality[idx]:
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
                    FROM {vdf}
                    {where}
                    GROUP BY 1 {order_by}"""
            ).to_numpy()
            X = res[:, 1:2].astype(float)
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
                               FROM {vdf}{where}) 
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
            "columns": self._clean_quotes(columns),
            "method": method,
            "method_of": method + f"({of})" if of else method,
            "of": self._clean_quotes(of),
            "of_cat": vdf[of].category() if of else None,
            "aggregate": clean_query(aggregate),
            "aggregate_fun": aggregate_fun,
            "is_standard": is_standard,
        }

    def _compute_contour_grid(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        func: Union[str, StringSQL, Callable],
        nbins: int = 100,
        func_name: Optional[str] = None,
    ) -> None:
        from verticapy.datasets.generators import gen_meshgrid

        columns = vdf.format_colnames(columns)
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
                vdf_tmp = create_new_vdf(func.format("_contour_Z", vdf_tmp))
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
                    if not isinstance(dataset[i][2], NoneType):
                        last_non_null_value = float(dataset[i][2])
                    x_tmp += [float(dataset[i][1])]
                    y_tmp += [float(dataset[i][0])]
                    z_tmp += [
                        float(
                            dataset[i][2]
                            if not isinstance(dataset[i][2], NoneType)
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
            "columns": self._clean_quotes(columns),
            "func": func,
            "func_repr": func_name
            if not isinstance(func_name, NoneType)
            else func_repr,
        }

    def _compute_contour_grid(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        func: Union[str, StringSQL, Callable],
        nbins: int = 100,
        func_name: Optional[str] = None,
    ) -> None:
        from verticapy.datasets.generators import gen_meshgrid

        if hasattr(self, "_max_nbins"):
            nbins = min(nbins, self._max_nbins)
        columns = vdf.format_colnames(columns)
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
                vdf_tmp = create_new_vdf(func.format("_contour_Z", vdf_tmp))
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
                    if not isinstance(dataset[i][2], NoneType):
                        last_non_null_value = float(dataset[i][2])
                    x_tmp += [float(dataset[i][1])]
                    y_tmp += [float(dataset[i][0])]
                    z_tmp += [
                        float(
                            dataset[i][2]
                            if not isinstance(dataset[i][2], NoneType)
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
            "columns": self._clean_quotes(columns),
            "func": func,
            "func_repr": func_name
            if not isinstance(func_name, NoneType)
            else func_repr,
        }

    def _compute_aggregate(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        method: str = "count",
        of: Optional[str] = None,
    ) -> None:
        columns = format_type(columns, dtype=list)
        columns = vdf.format_colnames(columns)
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
                FROM {vdf}
                GROUP BY {", ".join(columns)}""",
                title="Grouping all the elements for the Hexbin Plot",
                method="fetchall",
            )
        )
        self.data = {"X": X}
        self.layout = {
            "columns": self._clean_quotes(columns),
            "method": method,
            "method_of": method + f"({of})" if of else method,
            "of": self._clean_quotes(of),
            "of_cat": vdf[of].category() if of else None,
            "aggregate": clean_query(aggregate),
            "aggregate_fun": aggregate_fun,
            "is_standard": is_standard,
        }

    # SAMPLE: SCATTERS / OUTLIERS / ML PLOTS ...

    def _sample(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        size: Optional[str] = None,
        by: Optional[str] = None,
        cmap_col: Optional[str] = None,
        max_nb_points: int = 20000,
        h: PythonNumber = 0.0,
        max_cardinality: int = 8,
        cat_priority: Union[None, PythonScalar, ArrayLike] = None,
    ) -> None:
        columns = format_type(columns, dtype=list)
        columns = vdf.format_colnames(columns)
        cols_to_select = copy.deepcopy(columns)
        vdf_tmp = vdf.copy()
        has_category, has_cmap, has_size = False, False, False
        if max_nb_points > 0:
            if not isinstance(size, NoneType):
                cols_to_select += [vdf.format_colnames(size)]
                has_size = True
            if not isinstance(by, NoneType):
                has_category = True
                by = vdf.format_colnames(by)
                if vdf[by].isnum():
                    cols_to_select += [
                        vdf[by]
                        .discretize(h=h, return_enum_trans=True)[0]
                        .replace("{}", by)
                        + f" AS {by}"
                    ]
                else:
                    cols_to_select += [
                        vdf[by]
                        .discretize(
                            k=max_cardinality, method="topk", return_enum_trans=True
                        )[0]
                        .replace("{}", by)
                        + f" AS {by}"
                    ]
                if cat_priority:
                    vdf_tmp = vdf_tmp[by].isin(cat_priority)
            elif not isinstance(cmap_col, NoneType):
                cols_to_select += [vdf.format_colnames(cmap_col)]
                has_cmap = True
            X = vdf_tmp[cols_to_select].sample(n=max_nb_points).to_numpy()
            if len(X) > 0:
                X = X[:max_nb_points]
        else:
            X = np.array([])
        n = len(columns)
        self.data = {"X": X[:, :n].astype(float), "s": None, "c": None}
        self.layout = {
            "columns": self._clean_quotes(columns),
            "size": self._clean_quotes(size),
            "c": self._clean_quotes(by) if (not isinstance(by, NoneType)) else cmap_col,
            "has_category": has_category,
            "has_cmap": has_cmap,
            "has_size": has_size,
        }
        if not isinstance(size, NoneType) and (max_nb_points > 0):
            self.data["s"] = X[:, n].astype(float)
        if (
            (not isinstance(by, NoneType)) or (not isinstance(cmap_col, NoneType))
        ) and (max_nb_points > 0):
            self.data["c"] = X[:, -1]

    def _compute_outliers_params(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        max_nb_points: int = 1000,
        threshold: float = 3.0,
    ) -> None:
        columns = format_type(columns, dtype=list)
        columns = vdf.format_colnames(columns)
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
            "columns": self._clean_quotes(columns),
        }
        min0, max0 = self.data["min"][0], self.data["max"][0]
        avg0, std0 = self.data["avg"][0], self.data["std"][0]
        x_grid = np.linspace(min0, max0, 1000)
        if len(self.layout["columns"]) == 1:
            zvals = [-threshold * std0 + avg0, threshold * std0 + avg0]
            avg1, std1 = 0, 1
            y_grid = np.linspace(-1, 1, 1000)
            X, Y = np.meshgrid(x_grid, y_grid)
            Z = (X - avg0) / std0
            x = self.data["X"][:, 0]
            zs = abs(x - avg0) / std0
            inliers = x[abs(zs) <= threshold]
            n = len(inliers)
            inliers = np.column_stack(
                (inliers, [2 * (random.random() - 0.5) for i in range(n)])
            )
            outliers = x[abs(zs) > threshold]
            n = len(outliers)
            outliers = np.column_stack(
                (outliers, [2 * (random.random() - 0.5) for i in range(n)])
            )
        else:
            zvals = None
            min1, max1 = self.data["min"][1], self.data["max"][1]
            avg1, std1 = self.data["avg"][1], self.data["std"][1]
            y_grid = np.linspace(min1, max1, 1000)
            X, Y = np.meshgrid(x_grid, y_grid)
            Z = np.sqrt(((X - avg0) / std0) ** 2 + ((Y - avg1) / std1) ** 2)
            x = self.data["X"][:, 0]
            y = self.data["X"][:, 1]
            inliers = self.data["X"][
                (abs(x - avg0) / std0 <= threshold)
                & (abs(y - avg1) / std1 <= threshold)
            ]
            outliers = self.data["X"][
                (abs(x - avg0) / std0 > threshold) | (abs(y - avg1) / std1 > threshold)
            ]
        a = threshold * std0
        b = threshold * std1
        outliers_circle = [
            [
                avg0 + a * np.cos(2 * np.pi * x / 1000),
                avg1 + b * np.sin(2 * np.pi * x / 1000),
            ]
            for x in range(-1000, 1000, 1)
        ]
        self.data["map"] = {
            "X": X,
            "Y": Y,
            "Z": Z,
            "zvals": zvals,
            "outliers_circle": np.array(outliers_circle),
        }
        self.data["inliers"] = np.array(inliers)
        self.data["outliers"] = np.array(outliers)

    def _compute_scatter_matrix(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        max_nb_points: int = 20000,
    ) -> None:
        if not columns:
            columns = vdf.numcol()
        else:
            columns = format_type(columns, dtype=list)
        columns = vdf.format_colnames(columns)
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
                    data["hist"][self._clean_quotes(columns[i])] = copy.deepcopy(
                        self.data
                    )
        self.data = data
        self.layout = {
            "columns": self._clean_quotes(columns),
        }

    # TIME SERIES: LINE / RANGE

    def _filter_line(
        self,
        vdf: "vDataFrame",
        order_by: str,
        columns: SQLColumns,
        order_by_start: Optional[PythonScalar] = None,
        order_by_end: Optional[PythonScalar] = None,
        limit: int = -1,
        limit_over: int = -1,
    ) -> None:
        if not columns:
            columns = vdf.numcol()
        else:
            columns = format_type(columns, dtype=list)
        columns, order_by = vdf.format_colnames(columns, order_by)
        X = vdf.between(
            column=order_by, start=order_by_start, end=order_by_end, inplace=False
        )[[order_by] + columns].sort(columns=[order_by])
        if limit_over > 0:
            X = create_new_vdf(
                f"""
                SELECT * FROM {X}
                LIMIT {limit_over} OVER 
                    (PARTITION BY {order_by} 
                     ORDER BY {columns[1]} DESC)"""
            ).sort(columns=[order_by, columns[1]])
        if limit > 0:
            X = X[:limit]
        X = X.to_numpy()
        if not vdf[columns[-1]].isnum():
            Y = X[:, 1:-1]
            try:
                Y = Y.astype(float)
            except ValueError:
                pass
            self.data = {
                "x": X[:, 0],
                "Y": Y,
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
            "columns": self._clean_quotes(columns),
            "order_by": self._clean_quotes(order_by),
            "order_by_cat": vdf[order_by].category(),
            "has_category": has_category,
            "limit": limit,
            "limit_over": limit_over,
        }

    def _compute_tsa(
        self,
        vdf: "vDataFrame",
        order_by: str,
        columns: str,
        prediction: "vDataFrame",
        start: Optional[int],
        dataset_provided: bool,
        method: str,
    ) -> None:
        columns, order_by = vdf.format_colnames(columns, order_by)
        X = vdf[[order_by, columns]].sort(columns=[order_by]).to_numpy()
        X_pred = prediction.to_numpy()
        self.data = {
            "x": X[:, 0],
            "y": X[:, 1],
            "y_pred": X_pred[:, 0],
        }
        if not (dataset_provided):
            if isinstance(start, NoneType):
                start = 1
            j = -1
        else:
            if isinstance(start, NoneType):
                start = 1
                j = -1
            else:
                j = start
                start = 0
        has_se = False
        if X_pred.shape[1] > 1:
            self.data["se"] = np.array([0.0] + list(X_pred[:, 1]))
            has_se = True
        delta = self.data["x"][1] - self.data["x"][0]
        n = len(self.data["y_pred"])
        self.data["x_pred"] = np.array(
            [self.data["x"][j]]
            + [self.data["x"][j] + delta * i for i in range(start, n + start)]
        )
        self.data["y_pred"] = np.array([self.data["y"][j]] + list(self.data["y_pred"]))
        if has_se:
            self.data["se_x"] = self.data["x_pred"]
            self.data["se_low"] = self.data["y_pred"] - 1.96 * self.data["se"]
            self.data["se_high"] = self.data["y_pred"] + 1.96 * self.data["se"]
        else:
            self.data["se_x"] = None
            self.data["se_low"] = None
            self.data["se_high"] = None
        if str(method).lower() != "forecast" and j > 0:
            is_forecast = False
            m = len(self.data["x"])
            self.data["x_pred_one"] = self.data["x_pred"][: m - j + 1]
            self.data["y_pred_one"] = self.data["y_pred"][: m - j + 1]
            self.data["x_pred"] = self.data["x_pred"][m - j :]
            self.data["y_pred"] = self.data["y_pred"][m - j :]
        else:
            is_forecast = True
        self.layout = {
            "columns": self._clean_quotes(columns),
            "order_by": self._clean_quotes(order_by),
            "has_se": has_se,
            "is_forecast": is_forecast,
        }

    def _compute_range(
        self,
        vdf: "vDataFrame",
        order_by: str,
        columns: SQLColumns,
        q: tuple = (0.25, 0.75),
        order_by_start: Optional[PythonScalar] = None,
        order_by_end: Optional[PythonScalar] = None,
    ) -> None:
        columns = format_type(columns, dtype=list)
        columns, order_by = vdf.format_colnames(columns, order_by)
        expr = []
        for column in columns:
            expr += [
                f"APPROXIMATE_PERCENTILE({column} USING PARAMETERS percentile = {q[0]})",
                f"APPROXIMATE_MEDIAN({column})",
                f"APPROXIMATE_PERCENTILE({column} USING PARAMETERS percentile = {q[1]})",
            ]
        X = (
            vdf.between(
                column=order_by, start=order_by_start, end=order_by_end, inplace=False
            )
            .groupby(
                columns=[order_by],
                expr=expr,
            )
            .sort(columns=[order_by])
            .to_numpy()
        )
        self.data = {
            "x": self._parse_datetime(X[:, 0]),
            "Y": X[:, 1:].astype(float),
            "q": q,
        }
        self.layout = {
            "columns": self._clean_quotes(columns),
            "order_by": self._clean_quotes(order_by),
            "order_by_cat": vdf[order_by].category(),
        }

    def _compute_candle_aggregate(
        self,
        vdf: "vDataFrame",
        order_by: str,
        column: str,
        method: str = "sum",
        q: tuple = (0.25, 0.75),
        order_by_start: Optional[PythonScalar] = None,
        order_by_end: Optional[PythonScalar] = None,
    ) -> None:
        order_by, column = vdf.format_colnames(order_by, column)
        try:
            of = column
            method, aggregate, aggregate_fun, is_standard = self._map_method(method, of)
        except ValueError:
            of = None
            method, aggregate, aggregate_fun, is_standard = self._map_method(method, of)
        expr = [
            f"MIN({column})",
            f"APPROXIMATE_PERCENTILE({column} USING PARAMETERS percentile = {q[0]})",
            f"APPROXIMATE_PERCENTILE({column} USING PARAMETERS percentile = {q[1]})",
            f"MAX({column})",
            aggregate,
        ]
        X = (
            vdf.between(
                column=order_by, start=order_by_start, end=order_by_end, inplace=False
            )
            .groupby(order_by, expr)
            .sort(order_by)
            .to_numpy()
        )
        self.data = {"x": X[:, 0], "Y": X[:, 1:5], "z": X[:, 5], "q": q}
        self.layout = {
            "column": self._clean_quotes(column),
            "order_by": self._clean_quotes(order_by),
            "method": method,
            "method_of": method + f"({column})" if of else method,
            "aggregate": clean_query(aggregate),
            "aggregate_fun": aggregate_fun,
            "is_standard": is_standard,
        }

    def _filter_line_animated_scatter(
        self,
        vdf: "vDataFrame",
        order_by: str,
        columns: SQLColumns,
        by: Optional[str] = None,
        catcol: Optional[str] = None,
        order_by_start: Optional[PythonScalar] = None,
        order_by_end: Optional[PythonScalar] = None,
        limit_over: int = 10,
        limit: int = 1000000,
        lim_labels: int = 6,
    ) -> None:
        columns = format_type(columns, dtype=list)
        columns, order_by, by, catcol = vdf.format_colnames(
            columns, order_by, by, catcol
        )
        cols = copy.deepcopy(columns)
        if len(cols) == 2:
            cols += [1]
        by_ = 1 if isinstance(by, NoneType) else by
        if catcol:
            cols += [catcol]
        where = f" AND {order_by} > '{order_by_start}'" if (order_by_start) else ""
        where += f" AND {order_by} < '{order_by_end}'" if (order_by_end) else ""
        query_result = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('plotting._matplotlib._filter_line_animated_scatter')*/ * 
                FROM 
                    (SELECT 
                        {order_by}, 
                        {", ".join([str(column) for column in cols])}, 
                        {by_} 
                     FROM {vdf.current_relation(split=True)} 
                     WHERE  {cols[0]} IS NOT NULL 
                        AND {cols[1]} IS NOT NULL 
                        AND {cols[2]} IS NOT NULL
                        AND {order_by} IS NOT NULL
                        AND {by_} IS NOT NULL{where} 
                     LIMIT {limit_over} OVER (PARTITION BY {order_by} 
                                    ORDER BY {order_by}, {cols[2]} DESC)) x 
                ORDER BY {order_by}, 4 DESC, 3 DESC, 2 DESC 
                LIMIT {limit}""",
            title="Selecting points to draw the animated bubble plot.",
            method="fetchall",
        )
        self.data = {
            "x": np.array([x[0] for x in query_result]),
            "Y": np.array([x[1:] for x in query_result]),
        }
        self.layout = {
            "columns": self._clean_quotes(columns),
            "order_by": self._clean_quotes(order_by),
            "by": self._clean_quotes(by),
            "catcol": self._clean_quotes(catcol),
            "by_is_num": vdf[by].isnum() if not isinstance(by, NoneType) else None,
            "limit": limit,
            "limit_over": limit_over,
        }

    # ND AGG Graphics: BAR / PIE / DRILLDOWNS ...

    def _compute_rollup(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        method: str = "density",
        of: Optional[str] = None,
        max_cardinality: Union[int, tuple, list] = None,
        h: Union[int, tuple, list] = None,
    ) -> None:
        columns = format_type(columns, dtype=list)
        method, aggregate, aggregate_fun, is_standard = self._map_method(method, of)
        n = len(columns)
        if isinstance(h, (int, float, NoneType)):
            h = (h,) * n
        if isinstance(max_cardinality, (int, float, NoneType)):
            if isinstance(max_cardinality, NoneType):
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
            "columns": self._clean_quotes(columns),
            "method": method,
            "method_of": method + f"({of})" if of else method,
            "of": self._clean_quotes(of),
            "of_cat": vdf[of].category() if of else None,
            "aggregate": clean_query(aggregate),
            "aggregate_fun": aggregate_fun,
            "is_standard": is_standard,
        }
