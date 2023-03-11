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
from typing import Literal, Optional, TYPE_CHECKING

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from verticapy._config.colors import get_colors
import verticapy._config.config as conf
from verticapy._typing import ArrayLike, PythonScalar
from verticapy._utils._sql._sys import _executeSQL

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame

from verticapy.plotting._matplotlib.base import MatplotlibBase


class RangeCurve(MatplotlibBase):
    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["range"]:
        return "range"

    def range_curve(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        param_name: str = "",
        score_name: str = "score",
        ax: Optional[Axes] = None,
        labels: ArrayLike = [],
        without_scatter: bool = False,
        plot_median: bool = True,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a range curve using the Matplotlib API.
        """
        ax, fig = self._get_ax_fig(ax, size=(8, 6), set_axis_below=False, grid=False)
        for i, y in enumerate(Y):
            if labels:
                label = labels[i]
            else:
                label = ""
            if plot_median:
                alpha1, alpha2 = 0.3, 0.5
            else:
                alpha1, alpha2 = 0.5, 0.9
            param = {"facecolor": get_colors(style_kwargs, i)}
            ax.fill_between(X, y[0], y[2], alpha=alpha1, **param)
            param = {"color": get_colors(style_kwargs, i)}
            for j in [0, 2]:
                ax.plot(
                    X, y[j], alpha=alpha2, **self._update_dict(param, style_kwargs, i),
                )
            if plot_median:
                ax.plot(
                    X, y[1], label=label, **self._update_dict(param, style_kwargs, i)
                )
            if (not (without_scatter) or len(X) < 20) and plot_median:
                ax.scatter(
                    X, y[1], c="white", marker="o", s=60, edgecolors="black", zorder=3,
                )
        ax.set_xlabel(param_name)
        ax.set_ylabel(score_name)
        if labels:
            ax.legend(loc="center left", bbox_to_anchor=[1, 0.5])
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        ax.set_xlim(X[0], X[-1])
        return ax

    def draw(
        self,
        vdf: "vDataFrame",
        order_by: str,
        q: tuple = (0.25, 0.75),
        order_by_start: PythonScalar = None,
        order_by_end: PythonScalar = None,
        plot_median: bool = True,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a range curve using the Matplotlib API.
        """
        order_by_start_str, order_by_end_str = "", ""
        if order_by_start:
            order_by_start_str = f" AND {order_by} > '{order_by_start}'"
        if order_by_end:
            order_by_end_str = f" AND {order_by} < '{order_by_end}'"
        query_result = _executeSQL(
            query=f"""
            SELECT 
                /*+LABEL('plotting._matplotlib.range_curve_vdf')*/ 
                {order_by}, 
                APPROXIMATE_PERCENTILE({vdf._alias} USING PARAMETERS percentile = {q[0]}),
                APPROXIMATE_MEDIAN({vdf._alias}),
                APPROXIMATE_PERCENTILE({vdf._alias} USING PARAMETERS percentile = {q[1]})
            FROM {vdf._parent._genSQL()} 
            WHERE {order_by} IS NOT NULL 
              AND {vdf._alias} IS NOT NULL
              {order_by_start_str}
              {order_by_end_str}
            GROUP BY 1 ORDER BY 1""",
            title="Selecting points to draw the curve",
            method="fetchall",
        )
        order_by_values = [item[0] for item in query_result]
        if isinstance(order_by_values[0], str) and conf._get_import_success("dateutil"):
            order_by_values = self._parse_datetime(order_by_values)
        column_values = [
            [
                [float(item[1]) for item in query_result],
                [float(item[2]) for item in query_result],
                [float(item[3]) for item in query_result],
            ]
        ]
        return self.range_curve(
            order_by_values,
            column_values,
            order_by,
            vdf._alias,
            ax,
            [],
            True,
            plot_median,
            **style_kwargs,
        )
