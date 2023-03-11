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
import copy, warnings
from typing import Literal, Optional, TYPE_CHECKING
import numpy as np

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from verticapy._config.colors import get_colors
from verticapy._typing import PythonScalar, SQLColumns
from verticapy._utils._sql._format import quote_ident
from verticapy._utils._sql._sys import _executeSQL

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame, vDataColumn

from verticapy.plotting._matplotlib.base import MatplotlibBase


class LinePlot(MatplotlibBase):
    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["line"]:
        return "line"

    def draw(
        self,
        vdc: "vDataColumn",
        order_by: str,
        by: str = "",
        order_by_start: PythonScalar = None,
        order_by_end: PythonScalar = None,
        area: bool = False,
        step: bool = False,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a time series plot using the Matplotlib API.
        """
        ax, fig = self._get_ax_fig(ax, size=(8, 6), set_axis_below=True, grid="y")
        plot_fun = ax.step if step else ax.plot
        colors = get_colors()
        plot_param = {
            "marker": "o",
            "markevery": 0.05,
            "markersize": 7,
            "markeredgecolor": "black",
        }
        if not (by):
            matrix = vdc._parent.between(
                column=order_by, start=order_by_start, end=order_by_end, inplace=False
            )[[order_by, vdc._alias]].to_numpy()
            params = {
                "color": colors[0],
                "linewidth": 2,
            }
            if len(matrix[:, 0]) < 20:
                params = {
                    **plot_param,
                    **params,
                    "markerfacecolor": "white",
                }
            args = [matrix[:, 0], matrix[:, 1].astype(float)]
            params = self._update_dict(params, style_kwargs)
            plot_fun(*args, **params)
            if area and not (step):
                if "color" in self._update_dict(params, style_kwargs):
                    color = self._update_dict(params, style_kwargs)["color"]
                else:
                    color = colors[0]
                ax.fill_between(*args, facecolor=color, alpha=0.2)
            ax.set_xlim(min(matrix[:, 0]), max(matrix[:, 0]))
        else:
            uniques = vdc._parent[by].distinct()
            for i, c in enumerate(uniques):
                matrix = vdc._parent.between(
                    column=order_by,
                    start=order_by_start,
                    end=order_by_end,
                    inplace=False,
                )
                condition = f"""{quote_ident(by)} = '{str(c).replace("'", "''")}'"""
                matrix = matrix.search(condition)[[order_by, vdc._alias]].to_numpy()
                params = {
                    "color": colors[i % len(colors)],
                    "markerfacecolor": colors[i % len(colors)],
                }
                if len(matrix[:, 0]) < 20:
                    params = {
                        **plot_param,
                        **params,
                    }
                params = self._update_dict(params, style_kwargs, i)
                plot_fun(matrix[:, 0], matrix[:, 1].astype(float), label=c, **params)
        ax.set_xlabel(order_by)
        ax.set_ylabel(vdc._alias)
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        if by:
            ax.legend(title=by, loc="center left", bbox_to_anchor=[1, 0.5])
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax


class MultiLinePlot(MatplotlibBase):
    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["line"]:
        return "line"

    def draw(
        self,
        vdf: "vDataFrame",
        order_by: str,
        columns: SQLColumns = [],
        order_by_start: PythonScalar = None,
        order_by_end: PythonScalar = None,
        kind: str = "line",
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a multi-time series plot using the Matplotlib API.
        """
        if isinstance(columns, str):
            columns = [columns]
        if len(columns) == 1 and kind != "area_percent":
            return LinePlot().ts_plot(
                vdf[columns[0]],
                order_by=order_by,
                order_by_start=order_by_start,
                order_by_end=order_by_end,
                area=(kind in ("line", "step")),
                step=(kind == "step"),
                ax=ax,
                **style_kwargs,
            )
        if not (columns):
            columns = vdf.numcol()
        else:
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
        matrix = vdf.between(
            column=order_by, start=order_by_start, end=order_by_end, inplace=False
        )[[order_by] + columns].to_numpy()
        n, m = matrix.shape
        ax, fig = self._get_ax_fig(ax, size=(8, 6), set_axis_below=True, grid="y")
        plot_fun = ax.step if (kind == "step") else ax.plot
        prec = [0 for j in range(n)]
        for i in range(1, m):
            if kind in ("area_percent", "area_stacked"):
                points = np.sum(matrix[:, 1 : i + 1], axis=1).astype(float)
                if kind == "area_percent":
                    points /= np.sum(matrix[:, 1:], axis=1).astype(float)
            else:
                points = matrix[:, i].astype(float)
            param = {"linewidth": 1}
            param_style = {
                "marker": "o",
                "markevery": 0.05,
                "markerfacecolor": colors[i - 1],
                "markersize": 7,
            }
            if kind in ("line", "step"):
                color = colors[i - 1]
                if n < 20:
                    param = {
                        **param_style,
                        "markeredgecolor": "black",
                    }
                param["label"] = columns[i - 1]
                param["linewidth"] = 2
            elif kind == "area_percent":
                color = "white"
                if n < 20:
                    param = {
                        **param_style,
                        "markeredgecolor": "white",
                    }
            else:
                color = "black"
                if n < 20:
                    param = {
                        **param_style,
                        "markeredgecolor": "black",
                    }
            param["color"] = color
            if "color" in style_kwargs and n < 20:
                param["markerfacecolor"] = get_colors(style_kwargs, i)
            plot_fun(matrix[:, 0], points, **param)
            if kind not in ("line", "step"):
                args = [matrix[:, 0], prec, points]
                kwargs = {
                    "label": columns[i - 1],
                    "color": colors[i - 1],
                    **style_kwargs,
                }
                if not (isinstance(kwargs["color"], str)):
                    kwargs["color"] = kwargs["color"][(i - 1) % len(kwargs["color"])]
                ax.fill_between(*args, **kwargs)
                prec = points
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        if kind == "area_percent":
            ax.set_ylim(0, 1)
        elif kind == "area_stacked":
            ax.set_ylim(0)
        ax.set_xlim(min(matrix[:, 0]), max(matrix[:, 0]))
        ax.set_xlabel(order_by)
        ax.legend(loc="center left", bbox_to_anchor=[1, 0.5])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax
