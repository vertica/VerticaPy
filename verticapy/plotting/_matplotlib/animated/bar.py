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
from typing import Callable, Literal, Optional, TYPE_CHECKING

from matplotlib.axes import Axes
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

import verticapy._config.config as conf
from verticapy._typing import SQLColumns
from verticapy._utils._sql._sys import _executeSQL

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame

if conf._get_import_success("jupyter"):
    from IPython.display import HTML

from verticapy.plotting._matplotlib.base import MatplotlibBase


class AnimatedBarChart(MatplotlibBase):
    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["bar"]:
        return "animated_bar"

    def draw(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        order_by: str,
        by: str = "",
        order_by_start: str = "",
        order_by_end: str = "",
        limit_over: int = 6,
        limit: int = 1000000,
        fixed_xy_lim: bool = False,
        date_in_title: bool = False,
        date_f: Optional[Callable] = None,
        date_style_dict: dict = {},
        interval: int = 10,
        repeat: bool = True,
        return_html: bool = True,
        pie: bool = False,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> animation.Animation:
        """
        Draws an animated bar chart using the Matplotlib API.
        """
        if not (date_style_dict):
            date_style_dict = {
                "fontsize": 50,
                "alpha": 0.6,
                "color": "gray",
                "ha": "right",
                "va": "center",
            }
        if by:
            columns += [by]
        if date_f == None:

            def date_f(x):
                return str(x)

        colors = self.get_colors()
        for c in ["color", "colors"]:
            if c in style_kwargs:
                colors = style_kwargs[c]
                del style_kwargs[c]
                break
        if isinstance(colors, str):
            colors = []
        colors_map = {}
        idx = 2 if len(columns) >= 3 else 0
        all_cats = vdf[columns[idx]].distinct(agg=f"MAX({columns[1]})")
        for idx, i in enumerate(all_cats):
            colors_map[i] = colors[idx % len(colors)]
        order_by_start_str, order_by_end_str = "", ""
        if order_by_start:
            order_by_start_str = f"AND {order_by} > '{order_by_start}'"
        if order_by_end:
            order_by_end_str = f" AND {order_by} < '{order_by_end}'"
        condition = " AND ".join([f"{column} IS NOT NULL" for column in columns])
        query_result = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('plotting._matplotlib.AnimatedBarChart.animated_bar')*/ * 
                FROM 
                    (SELECT 
                        {order_by},
                        {", ".join(columns)} 
                     FROM {vdf._genSQL()} 
                     WHERE {order_by} IS NOT NULL 
                       AND {condition}
                       {order_by_start_str}
                       {order_by_end_str} 
                     LIMIT {limit_over} 
                        OVER (PARTITION BY {order_by} 
                              ORDER BY {columns[1]} DESC)) x 
                ORDER BY {order_by} ASC, {columns[1]} ASC 
                LIMIT {limit}""",
            title="Selecting points to draw the animated bar chart.",
            method="fetchall",
        )
        order_by_values = [item[0] for item in query_result]
        column1 = [item[1] for item in query_result]
        column2 = [float(item[2]) for item in query_result]
        column3 = []
        if len(columns) >= 3:
            column3 = [item[3] for item in query_result]
            color = [colors_map[item[3]] for item in query_result]
        else:
            color = [colors_map[item[1]] for item in query_result]
        current_ts, ts_idx = order_by_values[0], 0
        bar_values = []
        n = len(order_by_values)
        for idx, elem in enumerate(order_by_values):
            if elem != current_ts or idx == n - 1:
                bar_values += [
                    {
                        "y": column1[ts_idx:idx],
                        "width": column2[ts_idx:idx],
                        "c": color[ts_idx:idx],
                        "x": column3[ts_idx:idx],
                        "date": current_ts,
                    }
                ]
                current_ts, ts_idx = elem, idx
        if not (ax):
            fig, ax = plt.subplots()
            if conf._get_import_success("jupyter"):
                if pie:
                    fig.set_size_inches(11, min(limit_over, 600))
                else:
                    fig.set_size_inches(9, 6)
            ax.xaxis.grid()
            ax.set_axisbelow(True)

        def animate(i):
            ax.clear()
            if not (pie):
                ax.xaxis.grid()
                ax.set_axisbelow(True)
                min_x, max_x = min(bar_values[i]["width"]), max(bar_values[i]["width"])
                delta_x = max_x - min_x
                ax.barh(
                    y=bar_values[i]["y"],
                    width=bar_values[i]["width"],
                    color=bar_values[i]["c"],
                    alpha=0.6,
                    **style_kwargs,
                )
                if bar_values[i]["width"][0] > 0:
                    ax.barh(
                        y=bar_values[i]["y"],
                        width=[-0.3 * delta_x for elem in bar_values[i]["y"]],
                        color=bar_values[i]["c"],
                        alpha=0.6,
                        **style_kwargs,
                    )
                if fixed_xy_lim:
                    ax.set_xlim(min(column2), max(column2))
                else:
                    ax.set_xlim(min_x - 0.3 * delta_x, max_x + 0.3 * delta_x)
                all_text = []
                for k in range(len(bar_values[i]["y"])):
                    tmp_txt = []
                    tmp_txt += [
                        ax.text(
                            bar_values[i]["width"][k],
                            k + 0.1,
                            bar_values[i]["y"][k],
                            ha="right",
                            fontweight="bold",
                            size=10,
                        )
                    ]
                    width_format = bar_values[i]["width"][k]
                    if width_format - int(width_format) == 0:
                        width_format = int(width_format)
                    width_format = f"{width_format:}"
                    tmp_txt += [
                        ax.text(
                            bar_values[i]["width"][k] + 0.005 * delta_x,
                            k - 0.15,
                            width_format,
                            ha="left",
                            size=10,
                        )
                    ]
                    if len(columns) >= 3:
                        tmp_txt += [
                            ax.text(
                                bar_values[i]["width"][k],
                                k - 0.3,
                                bar_values[i]["x"][k],
                                ha="right",
                                size=10,
                                color="#333333",
                            )
                        ]
                    all_text += [tmp_txt]
                if date_in_title:
                    ax.set_title(date_f(bar_values[i]["date"]))
                else:
                    my_text = ax.text(
                        max_x + 0.27 * delta_x,
                        int(limit_over / 2),
                        date_f(bar_values[i]["date"]),
                        **date_style_dict,
                    )
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position("top")
                ax.set_xlabel(columns[1])
                ax.set_yticks([])
            else:
                param = {
                    "wedgeprops": {"edgecolor": "white", "alpha": 0.5},
                    "textprops": {"fontsize": 10, "fontweight": "bold"},
                    "autopct": "%1.1f%%",
                }

                def autopct(val):
                    a = val / 100.0 * sum(bar_values[i]["width"])
                    return f"{a:}"

                pie_chart = ax.pie(
                    x=bar_values[i]["width"],
                    labels=bar_values[i]["y"],
                    colors=bar_values[i]["c"],
                    **self._update_dict(param, style_kwargs),
                )
                for elem in pie_chart[2]:
                    elem.set_fontweight("normal")
                if date_in_title:
                    ax.set_title(date_f(bar_values[i]["date"]))
                else:
                    my_text = ax.text(
                        1.8, 1, date_f(bar_values[i]["date"]), **date_style_dict
                    )
                all_categories = []
                custom_lines = []
                if len(columns) >= 3:
                    for idx, elem in enumerate(bar_values[i]["x"]):
                        if elem not in all_categories:
                            all_categories += [elem]
                            custom_lines += [
                                Line2D(
                                    [0],
                                    [0],
                                    color=bar_values[i]["c"][idx],
                                    lw=6,
                                    alpha=self._update_dict(param, style_kwargs)[
                                        "wedgeprops"
                                    ]["alpha"],
                                )
                            ]
                    leg = ax.legend(
                        custom_lines,
                        all_categories,
                        title=by,
                        loc="center left",
                        bbox_to_anchor=[1, 0.5],
                    )
            return (ax,)

        myAnimation = animation.FuncAnimation(
            fig,
            animate,
            frames=range(0, len(bar_values)),
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
