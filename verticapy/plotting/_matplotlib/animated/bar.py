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
from typing import Any, Callable, Literal, Optional

import numpy as np

from matplotlib.axes import Axes
import matplotlib.animation as animation

from verticapy._typing import NoneType
from verticapy._utils._sql._format import format_type

from verticapy.plotting._matplotlib.animated.base import AnimatedBase


class AnimatedBarChart(AnimatedBase):
    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["animated_bar"]:
        return "animated_bar"

    @property
    def _compute_method(self) -> Literal["line"]:
        return "line"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_date_style_dict = {
            "fontsize": 50,
            "alpha": 0.6,
            "color": "gray",
            "ha": "right",
            "va": "center",
        }

    def _get_style_color(self, style_kwargs: dict) -> list:
        colors = self.get_colors()
        for c in ["color", "colors"]:
            if c in style_kwargs:
                colors = style_kwargs[c]
                del style_kwargs[c]
                break
        if isinstance(colors, str):
            colors = [colors]
        return colors

    # Draw.

    def _animate(
        self,
        bar_values: dict,
        m: int,
        date_f: Callable,
        fixed_xy_lim: bool,
        date_in_title: bool,
        date_style_dict: dict,
        style_kwargs: dict,
        ax: Axes,
    ) -> Callable:
        def animate(i: int) -> tuple[Axes]:
            ax.clear()
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
                n = len(bar_values[i]["y"])
                ax.barh(
                    y=bar_values[i]["y"],
                    width=[-0.3 * delta_x for i in range(n)],
                    color=bar_values[i]["c"],
                    alpha=0.6,
                    **style_kwargs,
                )
            if fixed_xy_lim:
                ax.set_xlim(min(self.data["Y"][:, 1]), max(self.data["Y"][:, 2]))
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
                if m >= 3:
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
                ax.text(
                    max_x + 0.27 * delta_x,
                    int(self.layout["limit_over"] / 2),
                    date_f(bar_values[i]["date"]),
                    **self.init_date_style_dict,
                    **date_style_dict,
                )
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            ax.set_xlabel(self.layout["columns"][1])
            ax.set_yticks([])
            return (ax,)

        return animate

    def _compute_anim_params(self, date_f: Optional[Callable], **style_kwargs) -> tuple:
        if isinstance(date_f, NoneType):

            def new_date_f(x: Any) -> str:
                return str(x)

            date_f = new_date_f

        colors = self._get_style_color(style_kwargs=style_kwargs)
        n, m = self.data["Y"].shape
        all_cats = np.unique(self.data["Y"][:, 0])
        colors_map = {}
        for j, i in enumerate(all_cats):
            colors_map[i] = colors[j % len(colors)]
        idx = 2 if m >= 3 else 0
        color = [colors_map[i] for i in self.data["Y"][:, idx]]
        bar_values = []
        current_ts, ts_idx = self.data["x"][0], 0
        for idx, x in enumerate(self.data["x"]):
            if x != current_ts or idx == n - 1:
                bar_values += [
                    {
                        "y": self.data["Y"][:, 0][ts_idx:idx],
                        "width": self.data["Y"][:, 1][ts_idx:idx].astype(float),
                        "c": color[ts_idx:idx],
                        "x": self.data["Y"][:, 2][ts_idx:idx] if m >= 3 else [],
                        "date": current_ts,
                    }
                ]
                current_ts, ts_idx = x, idx
        return m, date_f, bar_values

    def draw(
        self,
        fixed_xy_lim: bool = False,
        date_in_title: bool = False,
        date_f: Optional[Callable] = None,
        date_style_dict: Optional[dict] = None,
        interval: int = 10,
        repeat: bool = True,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> animation.Animation:
        """
        Draws an animated bar chart using the Matplotlib API.
        """
        date_style_dict = format_type(date_style_dict, dtype=dict)
        ax, fig, style_kwargs = self._get_ax_fig(
            ax, size=(9, 6), set_axis_below=True, grid=True, style_kwargs=style_kwargs
        )
        m, date_f, bar_values = self._compute_anim_params(
            date_f=date_f, style_kwargs=style_kwargs
        )
        anim = animation.FuncAnimation(
            fig,
            self._animate(
                bar_values=bar_values,
                m=m,
                date_f=date_f,
                fixed_xy_lim=fixed_xy_lim,
                date_in_title=date_in_title,
                date_style_dict=date_style_dict,
                style_kwargs=style_kwargs,
                ax=ax,
            ),
            frames=range(0, len(bar_values)),
            interval=interval,
            blit=False,
            repeat=repeat,
        )
        return self._return_animation(anim)
