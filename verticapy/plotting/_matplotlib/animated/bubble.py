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
import numpy as np
from typing import Callable, Literal, Optional, TYPE_CHECKING

from matplotlib.axes import Axes
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from verticapy._config.colors import get_cmap, get_colors
import verticapy._config.config as conf
from verticapy._typing import SQLColumns
from verticapy._utils._sql._sys import _executeSQL

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame

if conf._get_import_success("jupyter"):
    from IPython.display import HTML

from verticapy.plotting._matplotlib.base import MatplotlibBase


class AnimatedBubblePlot(MatplotlibBase):
    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["bubble"]:
        return "animated_bubble"

    def draw(
        self,
        vdf: "vDataFrame",
        order_by: str,
        columns: SQLColumns,
        by: str = "",
        label_name: str = "",
        order_by_start: str = "",
        order_by_end: str = "",
        limit_over: int = 10,
        limit: int = 1000000,
        lim_labels: int = 6,
        fixed_xy_lim: bool = False,
        bbox: list = [],
        img: str = "",
        date_in_title: bool = False,
        date_f: Optional[Callable] = None,
        date_style_dict: dict = {},
        interval: int = 10,
        repeat: bool = True,
        return_html: bool = True,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> animation.Animation:
        """
        Draws an animated bubble plot using the Matplotlib API.
        """
        if isinstance(columns, str):
            columns = [columns]
        if not (date_style_dict):
            date_style_dict = {
                "fontsize": 100,
                "alpha": 0.6,
                "color": "gray",
                "ha": "center",
                "va": "center",
            }
        if date_f == None:

            def date_f(x):
                return str(x)

        if len(columns) == 2:
            columns += [1]
        if "color" in style_kwargs:
            colors = style_kwargs["color"]
        elif "colors" in style_kwargs:
            colors = style_kwargs["colors"]
        else:
            colors = get_colors()
        if isinstance(colors, str):
            colors = [colors]
        param = {
            "alpha": 0.8,
            "edgecolors": "black",
        }
        if by:
            if vdf[by].isnum():
                param = {
                    "alpha": 0.8,
                    "cmap": get_cmap()[0],
                    "edgecolors": "black",
                }
            else:
                colors_map = {}
                all_cats = vdf[by].distinct(agg=f"MAX({columns[2]})")
                for idx, elem in enumerate(all_cats):
                    colors_map[elem] = colors[idx % len(colors)]
        else:
            by = 1
        if label_name:
            columns += [label_name]
        count = vdf.shape()[0]
        if not (ax):
            fig, ax = plt.subplots()
            if conf._get_import_success("jupyter"):
                fig.set_size_inches(12, 8)
            ax.grid()
            ax.set_axisbelow(True)
        else:
            fig = ax.get_figure()
        count = vdf.shape()[0]
        if columns[2] != 1:
            max_size = float(vdf[columns[2]].max())
            min_size = float(vdf[columns[2]].min())
        where = f" AND {order_by} > '{order_by_start}'" if (order_by_start) else ""
        where += f" AND {order_by} < '{order_by_end}'" if (order_by_end) else ""
        query_result = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('plotting._matplotlib.AnimatedBubblePlot.animated_bubble_plot')*/ * 
                FROM 
                    (SELECT 
                        {order_by}, 
                        {", ".join([str(column) for column in columns])}, 
                        {by} 
                     FROM {vdf._genSQL(True)} 
                     WHERE  {columns[0]} IS NOT NULL 
                        AND {columns[1]} IS NOT NULL 
                        AND {columns[2]} IS NOT NULL
                        AND {order_by} IS NOT NULL
                        AND {by} IS NOT NULL{where} 
                     LIMIT {limit_over} OVER (PARTITION BY {order_by} 
                                    ORDER BY {order_by}, {columns[2]} DESC)) x 
                ORDER BY {order_by}, 4 DESC, 3 DESC, 2 DESC 
                LIMIT {limit}""",
            title="Selecting points to draw the animated bubble plotting._matplotlib.",
            method="fetchall",
        )
        size = 50
        order_by_values = [item[0] for item in query_result]
        if columns[2] != 1:
            size = [
                1000 * (float(item[3]) - min_size) / max((max_size - min_size), 1e-50)
                for item in query_result
            ]
        column1, column2 = (
            [float(item[1]) for item in query_result],
            [float(item[2]) for item in query_result],
        )
        if label_name:
            label_columns = [item[-2] for item in query_result]
        if "cmap" in param:
            c = [float(item[4]) for item in query_result]
        elif by == 1:
            c = colors[0]
        else:
            custom_lines = []
            all_categories = []
            c = []
            for item in query_result:
                if item[-1] not in all_categories:
                    all_categories += [item[-1]]
                    custom_lines += [Line2D([0], [0], color=colors_map[item[-1]], lw=6)]
                c += [colors_map[item[-1]]]
        current_ts, ts_idx = order_by_values[0], 0
        scatter_values = []
        n = len(order_by_values)
        for idx, elem in enumerate(order_by_values):
            if elem != current_ts or idx == n - 1:
                scatter_values += [
                    {
                        "x": column1[ts_idx:idx],
                        "y": column2[ts_idx:idx],
                        "c": c[ts_idx:idx] if isinstance(c, list) else c,
                        "s": size
                        if isinstance(size, (float, int))
                        else size[ts_idx:idx],
                        "date": current_ts,
                    }
                ]
                if label_name:
                    scatter_values[-1]["label"] = label_columns[ts_idx:idx]
                current_ts, ts_idx = elem, idx
        im = ax.scatter(
            scatter_values[0]["x"],
            scatter_values[0]["y"],
            c=scatter_values[0]["c"],
            s=scatter_values[0]["s"],
            **self._update_dict(param, style_kwargs),
        )
        if label_name:
            text_plots = []
            for idx in range(lim_labels):
                text_plots += [
                    ax.text(
                        scatter_values[0]["x"][idx],
                        scatter_values[0]["y"][idx],
                        scatter_values[0]["label"][idx],
                        ha="right",
                        va="bottom",
                    )
                ]
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        if bbox:
            ax.set_xlim(bbox[0], bbox[1])
            ax.set_ylim(bbox[2], bbox[3])
            if not (date_in_title):
                my_text = ax.text(
                    (bbox[0] + bbox[1]) / 2,
                    (bbox[2] + bbox[3]) / 2,
                    date_f(scatter_values[0]["date"]),
                    **date_style_dict,
                )
        elif fixed_xy_lim:
            min_x, max_x = min(column1), max(column1)
            min_y, max_y = min(column2), max(column2)
            delta_x, delta_y = max_x - min_x, max_y - min_y
            ax.set_xlim(min_x - 0.02 * delta_x, max_x + 0.02 * delta_x)
            ax.set_ylim(min_y - 0.02 * delta_y, max_y + 0.02 * delta_y)
            if not (date_in_title):
                my_text = ax.text(
                    (max_x + min_x) / 2,
                    (max_y + min_y) / 2,
                    date_f(scatter_values[0]["date"]),
                    **date_style_dict,
                )
        if img:
            bim = plt.imread(img)
            if not (bbox):
                bbox = (min(column1), max(column1), min(column2), max(column2))
                ax.set_xlim(bbox[0], bbox[1])
                ax.set_ylim(bbox[2], bbox[3])
            ax.imshow(bim, extent=bbox)
        elif not (date_in_title):
            my_text = ax.text(
                (max(scatter_values[0]["x"]) + min(scatter_values[0]["x"])) / 2,
                (max(scatter_values[0]["y"]) + min(scatter_values[0]["y"])) / 2,
                date_f(scatter_values[0]["date"]),
                **date_style_dict,
            )
        if "cmap" in param:
            fig.colorbar(im, ax=ax).set_label(by)
        elif label_name:
            leg = ax.legend(
                custom_lines,
                all_categories,
                title=by,
                loc="center left",
                bbox_to_anchor=[1, 0.5],
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        def animate(i):
            array = np.array(
                [
                    (scatter_values[i]["x"][j], scatter_values[i]["y"][j])
                    for j in range(len(scatter_values[i]["x"]))
                ]
            )
            im.set_offsets(array)
            if columns[2] != 1:
                im.set_sizes(np.array(scatter_values[i]["s"]))
            if "cmap" in param:
                im.set_array(np.array(scatter_values[i]["c"]))
            elif label_name:
                im.set_color(np.array(scatter_values[i]["c"]))
            if "edgecolors" in self._update_dict(param, style_kwargs):
                im.set_edgecolor(self._update_dict(param, style_kwargs)["edgecolors"])
            if label_name:
                for k in range(lim_labels):
                    text_plots[k].set_position(
                        (scatter_values[i]["x"][k], scatter_values[i]["y"][k])
                    )
                    text_plots[k].set_text(scatter_values[i]["label"][k])
            min_x, max_x = min(scatter_values[i]["x"]), max(scatter_values[i]["x"])
            min_y, max_y = min(scatter_values[i]["y"]), max(scatter_values[i]["y"])
            delta_x, delta_y = max_x - min_x, max_y - min_y
            if not (fixed_xy_lim):
                ax.set_xlim(min_x - 0.02 * delta_x, max_x + 0.02 * delta_x)
                ax.set_ylim(min_y - 0.02 * delta_y, max_y + 0.02 * delta_y)
                if not (date_in_title):
                    my_text.set_position([(max_x + min_x) / 2, (max_y + min_y) / 2])
            if not (date_in_title):
                my_text.set_text(date_f(scatter_values[i]["date"]))
            else:
                ax.set_title(date_f(scatter_values[i]["date"]))
            return (ax,)

        myAnimation = animation.FuncAnimation(
            fig,
            animate,
            frames=range(1, len(scatter_values)),
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
