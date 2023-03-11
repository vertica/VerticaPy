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
from typing import Literal, Optional, TYPE_CHECKING

from matplotlib.axes import Axes
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from verticapy._config.colors import get_colors
import verticapy._config.config as conf
from verticapy._typing import SQLColumns
from verticapy._utils._sql._sys import _executeSQL

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame

if conf._get_import_success("jupyter"):
    from IPython.display import HTML

from verticapy.plotting._matplotlib.base import MatplotlibBase


class AnimatedLinePlot(MatplotlibBase):
    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["line"]:
        return "animated_line"

    def draw(
        self,
        vdf: "vDataFrame",
        order_by: str,
        columns: SQLColumns = [],
        order_by_start: str = "",
        order_by_end: str = "",
        limit: int = 1000000,
        fixed_xy_lim: bool = False,
        window_size: int = 100,
        step: int = 10,
        interval: int = 5,
        repeat: bool = True,
        return_html: bool = True,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> animation.Animation:
        """
        Draws an animated Time Series plot using the Matplotlib API.
        """
        if not (columns):
            columns = vdf.numcol()
        if isinstance(columns, str):
            columns = [columns]
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
            raise EmptyParameter(
                "No numerical columns found to draw the animated multi TS plot"
            )
        order_by_start_str, order_by_end_str, limit_str = "", "", ""
        if order_by_start:
            order_by_start_str = f" AND {order_by} > '{order_by_start}'"
        if order_by_end:
            order_by_end_str = f" AND {order_by} < '{order_by_end}'"
        if limit:
            limit_str = f" LIMIT {limit}"
        condition = " AND ".join([f"{column} IS NOT NULL" for column in columns])
        query_result = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('plotting._matplotlib.AnimatedLinePlot.animated_ts_plot')*/ 
                    {order_by},
                    {", ".join(columns)} 
                FROM {vdf._genSQL()} 
                WHERE {order_by} IS NOT NULL
                      {order_by_start_str}
                      {order_by_end_str}
                  AND {condition}
                ORDER BY {order_by}
                {limit_str}
                """,
            title="Selecting the needed points to draw the curves",
            method="fetchall",
        )
        order_by_values = [column[0] for column in query_result]
        if isinstance(order_by_values[0], str) and conf._get_import_success("dateutil"):
            order_by_values = self._parse_datetime(order_by_values)
        alpha = 0.3
        if not (ax):
            fig, ax = plt.subplots()
            if conf._get_import_success("jupyter"):
                fig.set_size_inches(8, 6)
            ax.grid(axis="y")
            ax.set_axisbelow(True)
        else:
            fig = plt
        all_plots = []
        colors = get_colors()
        for i in range(0, len(columns)):
            param = {
                "linewidth": 1,
                "label": columns[i],
                "linewidth": 2,
                "color": colors[i % len(colors)],
            }
            all_plots += [
                ax.plot([], [], **self._update_dict(param, style_kwargs, i))[0]
            ]
        if len(columns) > 1:
            ax.legend(loc="center left", bbox_to_anchor=[1, 0.5])
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.set_xlabel(order_by)
        if fixed_xy_lim:
            ax.set_xlim(order_by_values[0], order_by_values[-1])
            y_tmp = []
            for m in range(0, len(columns)):
                y_tmp += [item[m + 1] for item in query_result]
            ax.set_ylim(min(y_tmp), max(y_tmp))

        def animate(i):
            k = max(i - window_size, 0)
            x = [elem for elem in order_by_values]
            all_y = []
            for m in range(0, len(columns)):
                y = [item[m + 1] for item in query_result]
                all_plots[m].set_xdata(x[0:i])
                all_plots[m].set_ydata(y[0:i])
                all_y += y[0:i]
            if not (fixed_xy_lim):
                if i > 0:
                    ax.set_ylim(min(all_y), max(all_y))
                if i > window_size:
                    ax.set_xlim(x[k], x[i])
                else:
                    ax.set_xlim(x[0], x[window_size])
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
            return (ax,)

        myAnimation = animation.FuncAnimation(
            fig,
            animate,
            frames=range(0, len(order_by_values) - 1, step),
            interval=interval,
            blit=False,
            repeat=repeat,
        )
        if conf._get_import_success("jupyter") and return_html:
            anim = myAnimation.to_jshtml()
            plt.close("all")
            return HTML(anim)
        else:
            return myAnimation
