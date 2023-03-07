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
import warnings
from typing import Optional, TYPE_CHECKING

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from verticapy._config.colors import get_colors
import verticapy._config.config as conf
from verticapy._typing import SQLColumns
from verticapy._utils._sql._format import quote_ident
from verticapy._utils._sql._sys import _executeSQL

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame

from verticapy.plotting.base import PlottingBase


class LinePlot(PlottingBase):
    def multi_ts_plot(
        self,
        vdf: "vDataFrame",
        order_by: str,
        columns: SQLColumns = [],
        order_by_start: str = "",
        order_by_end: str = "",
        kind: str = "line",
        ax: Optional[Axes] = None,
        **style_kwds,
    ) -> Axes:
        """
        Draws a multi-time series plot using the Matplotlib API.
        """
        if isinstance(columns, str):
            columns = [columns]
        if len(columns) == 1 and kind != "area_percent":
            if kind in ("line", "step"):
                area = False
            else:
                area = True
            if kind == "step":
                step = True
            else:
                step = False
            return vdf[columns[0]].plot(
                ts=order_by,
                start_date=order_by_start,
                end_date=order_by_end,
                area=area,
                step=step,
                **style_kwds,
            )
        if not (columns):
            columns = vdf.numcol()
        for column in columns:
            if not (vdf[column].isnum()):
                if vdf._vars["display"]["print_info"]:
                    warning_message = (
                        f"The Virtual Column {column} is "
                        "not numerical.\nIt will be ignored."
                    )
                    warnings.warn(warning_message, Warning)
                columns.remove(column)
        if not (columns):
            raise EmptyParameter("No numerical columns found to draw the multi TS plot")
        colors = get_colors()
        order_by_start_str, order_by_end_str = "", ""
        if order_by_start:
            order_by_start_str = f" AND {order_by} > '{order_by_start}'"
        if order_by_end:
            order_by_end_str = f" AND {order_by} < '{order_by_end}'"
        condition = " AND " + " AND ".join(
            [f"{column} IS NOT NULL" for column in columns]
        )
        query_result = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('plotting._matplotlib.multi_ts_plot')*/ 
                    {order_by}, 
                    {", ".join(columns)} 
                FROM {vdf._genSQL()} 
                WHERE {order_by} IS NOT NULL
                {condition}
                ORDER BY {order_by}""",
            title="Selecting the needed points to draw the curves",
            method="fetchall",
        )
        order_by_values = [item[0] for item in query_result]
        if isinstance(order_by_values[0], str) and conf._get_import_success("dateutil"):
            order_by_values = self.parse_datetime(order_by_values)
        alpha = 0.3
        if not (ax):
            fig, ax = plt.subplots()
            if conf._get_import_success("jupyter"):
                fig.set_size_inches(8, 6)
            ax.grid(axis="y")
            ax.set_axisbelow(True)
        prec = [0 for item in query_result]
        for i in range(0, len(columns)):
            if kind == "area_percent":
                points = [
                    sum(item[1 : i + 2]) / max(sum(item[1:]), 1e-70)
                    for item in query_result
                ]
            elif kind == "area_stacked":
                points = [sum(item[1 : i + 2]) for item in query_result]
            else:
                points = [item[i + 1] for item in query_result]
            param = {"linewidth": 1}
            param_style = {
                "marker": "o",
                "markevery": 0.05,
                "markerfacecolor": colors[i],
                "markersize": 7,
            }
            if kind in ("line", "step"):
                color = colors[i]
                if len(order_by_values) < 20:
                    param = {
                        **param_style,
                        "markeredgecolor": "black",
                    }
                param["label"] = columns[i]
                param["linewidth"] = 2
            elif kind == "area_percent":
                color = "white"
                if len(order_by_values) < 20:
                    param = {
                        **param_style,
                        "markeredgecolor": "white",
                    }
            else:
                color = "black"
                if len(order_by_values) < 20:
                    param = {
                        **param_style,
                        "markeredgecolor": "black",
                    }
            param["color"] = color
            if "color" in style_kwds and len(order_by_values) < 20:
                param["markerfacecolor"] = get_colors(style_kwds, i)
            if kind == "step":
                ax.step(order_by_values, points, **param)
            else:
                ax.plot(order_by_values, points, **param)
            if kind not in ("line", "step"):
                tmp_style = {}
                for elem in style_kwds:
                    tmp_style[elem] = style_kwds[elem]
                if "color" not in tmp_style:
                    tmp_style["color"] = colors[i]
                else:
                    if not (isinstance(tmp_style["color"], str)):
                        tmp_style["color"] = tmp_style["color"][
                            i % len(tmp_style["color"])
                        ]
                ax.fill_between(
                    order_by_values, prec, points, label=columns[i], **tmp_style
                )
                prec = points
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        if kind == "area_percent":
            ax.set_ylim(0, 1)
        elif kind == "area_stacked":
            ax.set_ylim(0)
        ax.set_xlim(min(order_by_values), max(order_by_values))
        ax.set_xlabel(order_by)
        ax.legend(loc="center left", bbox_to_anchor=[1, 0.5])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax

    def ts_plot(
        self,
        vdf: "vDataFrame",
        order_by: str,
        by: str = "",
        order_by_start: str = "",
        order_by_end: str = "",
        area: bool = False,
        step: bool = False,
        ax: Optional[Axes] = None,
        **style_kwds,
    ) -> Axes:
        """
        Draws a time series plot using the Matplotlib API.
        """
        if order_by_start:
            order_by_start_str = f" AND {order_by} > '{order_by_start}'"
        else:
            order_by_start_str = ""
        if order_by_end:
            order_by_end_str = f" AND {order_by} < '{order_by_end}'"
        else:
            order_by_end_str = ""
        query = f"""
            SELECT 
                /*+LABEL('plotting._matplotlib.ts_plot')*/ 
                {order_by},
                {vdf._alias}
            FROM {vdf._parent._genSQL()}
            WHERE {order_by} IS NOT NULL 
              AND {vdf._alias} IS NOT NULL
              {order_by_start_str}
              {order_by_end_str}
              {{}}
            ORDER BY {order_by}, {vdf._alias}"""
        title = "Selecting points to draw the curve"
        if not (ax):
            fig, ax = plt.subplots()
            if conf._get_import_success("jupyter"):
                fig.set_size_inches(8, 6)
            ax.grid(axis="y")
        colors = get_colors()
        plot_fun = ax.step if step else ax.plot
        plot_param = {
            "marker": "o",
            "markevery": 0.05,
            "markersize": 7,
            "markeredgecolor": "black",
        }
        if not (by):
            query_result = _executeSQL(
                query=query.format(""), title=title, method="fetchall",
            )
            order_by_values = [item[0] for item in query_result]
            if isinstance(order_by_values[0], str) and conf._get_import_success(
                "dateutil"
            ):
                order_by_values = self.parse_datetime(order_by_values)
            column_values = [float(item[1]) for item in query_result]
            param = {
                "color": colors[0],
                "linewidth": 2,
            }
            if len(order_by_values) < 20:
                param = {
                    **plot_param,
                    **param,
                    "markerfacecolor": "white",
                }
            plot_fun(
                order_by_values, column_values, **self.updated_dict(param, style_kwds)
            )
            if area and not (step):
                if "color" in self.updated_dict(param, style_kwds):
                    color = self.updated_dict(param, style_kwds)["color"]
                ax.fill_between(
                    order_by_values, column_values, facecolor=color, alpha=0.2
                )
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
            ax.set_xlabel(order_by)
            ax.set_ylabel(vdf._alias)
            ax.set_xlim(min(order_by_values), max(order_by_values))
        else:
            by = quote_ident(by)
            cat = vdf._parent[by].distinct()
            all_data = []
            for column in cat:
                column_str = str(column).replace("'", "''")
                query_result = _executeSQL(
                    query=query.format(f"AND {by} = '{column_str}'"),
                    title=title,
                    method="fetchall",
                )
                all_data += [
                    [
                        [item[0] for item in query_result],
                        [float(item[1]) for item in query_result],
                        column,
                    ]
                ]
                if isinstance(all_data[-1][0][0], str) and conf._get_import_success(
                    "dateutil"
                ):
                    all_data[-1][0] = self.parse_datetime(all_data[-1][0])
            for idx, d in enumerate(all_data):
                param = {"color": colors[idx % len(colors)]}
                if len(d[0]) < 20:
                    param = {
                        **plot_param,
                        **param,
                        "markerfacecolor": colors[idx % len(colors)],
                    }
                param["markerfacecolor"] = get_colors(style_kwds, idx)
                plot_fun(
                    d[0], d[1], label=d[2], **self.updated_dict(param, style_kwds, idx)
                )
            ax.set_xlabel(order_by)
            ax.set_ylabel(vdf._alias)
            ax.legend(title=by, loc="center left", bbox_to_anchor=[1, 0.5])
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
        return ax
