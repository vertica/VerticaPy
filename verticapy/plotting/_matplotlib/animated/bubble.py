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
from typing import Any, Callable, Literal, Optional, Union, TYPE_CHECKING

import numpy as np

from matplotlib.axes import Axes
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

import verticapy._config.config as conf
from verticapy._typing import NoneType
from verticapy._utils._sql._format import format_type

from verticapy.plotting._matplotlib.animated.base import AnimatedBase

if conf.get_import_success("IPython"):
    from IPython.display import HTML

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame


class AnimatedBubblePlot(AnimatedBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["animated_scatter"]:
        return "animated_scatter"

    @property
    def _compute_method(self) -> Literal["line_bubble"]:
        return "line_bubble"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (2, 4)

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "alpha": 0.8,
            "edgecolors": "black",
        }
        self.init_date_style_dict = {
            "fontsize": 100,
            "alpha": 0.6,
            "color": "gray",
            "ha": "center",
            "va": "center",
        }

    def _get_final_color(self, style_kwargs: dict) -> list:
        if "color" in style_kwargs:
            colors = style_kwargs["color"]
        elif "colors" in style_kwargs:
            colors = style_kwargs["colors"]
        else:
            colors = self.get_colors()
        if isinstance(colors, str):
            colors = [colors]
        return colors

    # Draw.

    def _animate(
        self,
        scatter_plot_init: plt.scatter,
        scatter_values: list,
        lim_labels: int,
        set_size: bool,
        date_in_title: bool,
        fixed_xy_lim: bool,
        text_plots: list,
        anim_text: plt.Text,
        date_f: Callable,
        kwargs: dict,
        style_kwargs: dict,
        ax: Axes,
    ) -> Callable:
        def animate(i: int) -> tuple[Axes]:
            array = np.array(
                [
                    (scatter_values[i]["x"][j], scatter_values[i]["y"][j])
                    for j in range(len(scatter_values[i]["x"]))
                ]
            )
            scatter_plot_init.set_offsets(array)
            if set_size:
                scatter_plot_init.set_sizes(np.array(scatter_values[i]["s"]))
            if "cmap" in kwargs:
                scatter_plot_init.set_array(np.array(scatter_values[i]["c"]))
            elif not isinstance(self.layout["catcol"], NoneType):
                scatter_plot_init.set_color(np.array(scatter_values[i]["c"]))
            if "edgecolors" in self._update_dict(kwargs, style_kwargs):
                scatter_plot_init.set_edgecolor(
                    self._update_dict(kwargs, style_kwargs)["edgecolors"]
                )
            if not isinstance(self.layout["catcol"], NoneType):
                for k in range(lim_labels):
                    text_plots[k].set_position(
                        (scatter_values[i]["x"][k], scatter_values[i]["y"][k])
                    )
                    text_plots[k].set_text(scatter_values[i]["label"][k])
            min_x, max_x = min(scatter_values[i]["x"]), max(scatter_values[i]["x"])
            min_y, max_y = min(scatter_values[i]["y"]), max(scatter_values[i]["y"])
            delta_x, delta_y = max_x - min_x, max_y - min_y
            if not fixed_xy_lim:
                ax.set_xlim(min_x - 0.02 * delta_x, max_x + 0.02 * delta_x)
                ax.set_ylim(min_y - 0.02 * delta_y, max_y + 0.02 * delta_y)
                if not date_in_title:
                    anim_text.set_position([(max_x + min_x) / 2, (max_y + min_y) / 2])
            if not date_in_title:
                anim_text.set_text(date_f(scatter_values[i]["date"]))
            else:
                ax.set_title(date_f(scatter_values[i]["date"]))
            return (ax,)

        return animate

    def _compute_anim_params(
        self, date_f: Optional[Callable] = None, **style_kwargs
    ) -> tuple:
        if isinstance(date_f, NoneType):

            def new_date_f(x: Any) -> str:
                return str(x)

            date_f = new_date_f

        colors = self._get_final_color(style_kwargs=style_kwargs)
        kwargs = self.init_style
        if self.layout["by"]:
            if self.layout["by_is_num"]:
                kwargs = {
                    **kwargs,
                    "cmap": self.get_cmap(idx=0),
                }
            else:
                colors_map = {}
                all_cats = np.unique(self.data["Y"][:, -1])
                for idx, cat in enumerate(all_cats):
                    colors_map[cat] = colors[idx % len(colors)]
        size = 50
        if len(self.layout["columns"]) > 2:
            Y2 = self.data["Y"][:, 2].astype(float)
            min_size = np.nanmin(Y2)
            max_size = np.nanmax(Y2)
            size = 1000 * (Y2 - min_size) / max((max_size - min_size), 1e-50)
        custom_lines, all_categories, c = [], [], []
        if "cmap" in kwargs:
            c = list(self.data["Y"][:, 2].astype(float))
        elif isinstance(self.layout["by"], NoneType):
            c = [colors[0]] * len(self.data["x"])
        else:
            for cat in self.data["Y"][:, -1]:
                if cat not in all_categories:
                    all_categories += [cat]
                    custom_lines += [Line2D([0], [0], color=colors_map[cat], lw=6)]
                c += [colors_map[cat]]
        current_ts, ts_idx = self.data["x"][0], 0
        scatter_values = []
        n = len(self.data["x"])
        Y0 = self.data["Y"][:, 0].astype(float)
        Y1 = self.data["Y"][:, 1].astype(float)
        for idx, x in enumerate(self.data["x"]):
            if x != current_ts or idx == n - 1:
                scatter_values += [
                    {
                        "x": Y0[ts_idx:idx],
                        "y": Y1[ts_idx:idx],
                        "c": c
                        if isinstance(c, str) or isinstance(c, NoneType)
                        else c[ts_idx:idx],
                        "s": size
                        if isinstance(size, (float, int))
                        else size[ts_idx:idx],
                        "date": current_ts,
                    }
                ]
                if self.layout["catcol"]:
                    scatter_values[-1]["label"] = self.data["Y"][:, -2][ts_idx:idx]
                current_ts, ts_idx = x, idx
        return kwargs, date_f, scatter_values, custom_lines, all_categories

    def draw(
        self,
        lim_labels: int = 6,
        fixed_xy_lim: bool = False,
        bbox: Optional[list] = None,
        img: Optional[str] = None,
        date_in_title: bool = False,
        date_f: Optional[Callable] = None,
        date_style_dict: Optional[dict] = None,
        interval: int = 10,
        repeat: bool = True,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Union["HTML", animation.Animation]:
        """
        Draws an animated bubble plot using the Matplotlib API.
        """
        date_style_dict = format_type(date_style_dict, dtype=dict)
        bbox = format_type(bbox, dtype=list)
        ax, fig, style_kwargs = self._get_ax_fig(
            ax, size=(12, 8), set_axis_below=True, grid=True, style_kwargs=style_kwargs
        )
        (
            kwargs,
            date_f,
            scatter_values,
            custom_lines,
            all_cats,
        ) = self._compute_anim_params(
            date_f=date_f,
            style_kwargs=style_kwargs,
        )
        sc = ax.scatter(
            scatter_values[0]["x"],
            scatter_values[0]["y"],
            c=scatter_values[0]["c"],
            s=scatter_values[0]["s"],
            **self._update_dict(kwargs, style_kwargs),
        )
        text_plots = []
        if not isinstance(self.layout["catcol"], NoneType):
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
        ax.set_xlabel(self.layout["columns"][0])
        ax.set_ylabel(self.layout["columns"][1])
        Y0 = self.data["Y"][:, 0].astype(float)
        Y1 = self.data["Y"][:, 1].astype(float)
        min_x = np.nanmin(Y0)
        max_x = np.nanmax(Y0)
        min_y = np.nanmin(Y1)
        max_y = np.nanmax(Y1)
        if bbox:
            ax.set_xlim(bbox[0], bbox[1])
            ax.set_ylim(bbox[2], bbox[3])
            if not date_in_title:
                anim_text = ax.text(
                    (bbox[0] + bbox[1]) / 2,
                    (bbox[2] + bbox[3]) / 2,
                    date_f(scatter_values[0]["date"]),
                    **self.init_date_style_dict,
                    **date_style_dict,
                )
        elif fixed_xy_lim:
            delta_x, delta_y = max_x - min_x, max_y - min_y
            ax.set_xlim(min_x - 0.02 * delta_x, max_x + 0.02 * delta_x)
            ax.set_ylim(min_y - 0.02 * delta_y, max_y + 0.02 * delta_y)
            if not date_in_title:
                anim_text = ax.text(
                    (max_x + min_x) / 2,
                    (max_y + min_y) / 2,
                    date_f(scatter_values[0]["date"]),
                    **self.init_date_style_dict,
                    **date_style_dict,
                )
        if img:
            bim = plt.imread(img)
            if not bbox:
                bbox = (min_x, max_x, min_y, max_y)
                ax.set_xlim(bbox[0], bbox[1])
                ax.set_ylim(bbox[2], bbox[3])
            ax.imshow(bim, extent=bbox)
        elif not date_in_title:
            anim_text = ax.text(
                (max(scatter_values[0]["x"]) + min(scatter_values[0]["x"])) / 2,
                (max(scatter_values[0]["y"]) + min(scatter_values[0]["y"])) / 2,
                date_f(scatter_values[0]["date"]),
                **self.init_date_style_dict,
                **date_style_dict,
            )
        if "cmap" in kwargs:
            fig.colorbar(sc, ax=ax).set_label(self.layout["by"])
        elif not isinstance(self.layout["catcol"], NoneType):
            ax.legend(
                custom_lines,
                all_cats,
                title=self.layout["by"],
                loc="center left",
                bbox_to_anchor=[1, 0.5],
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        animate = self._animate(
            scatter_plot_init=sc,
            scatter_values=scatter_values,
            lim_labels=lim_labels,
            set_size=(len(self.layout["columns"]) > 2),
            date_in_title=date_in_title,
            fixed_xy_lim=fixed_xy_lim,
            anim_text=anim_text,
            text_plots=text_plots,
            date_f=date_f,
            kwargs=kwargs,
            style_kwargs=style_kwargs,
            ax=ax,
        )
        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=range(1, len(scatter_values)),
            interval=interval,
            blit=False,
            repeat=repeat,
        )
        return self._return_animation(a=anim)
