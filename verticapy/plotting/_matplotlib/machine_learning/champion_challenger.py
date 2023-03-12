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
from typing import Literal, Optional

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from verticapy.plotting._matplotlib.base import MatplotlibBase


class ChampionChallengerPlot(MatplotlibBase):
    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["champion"]:
        return "champion"

    def draw(
        self,
        x: list,
        y: list,
        s: list = None,
        z: list = [],
        x_label: str = "time",
        y_label: str = "score",
        title: str = "Model Type",
        reverse: tuple = (True, True),
        plt_text: bool = True,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a Machine Learning Bubble Plot using the Matplotlib API.
        """
        if s:
            s = [min(250 + 5000 * elem, 1200) if elem != 0 else 1000 for elem in s]
        if z and s:
            data = [(x[i], y[i], s[i], z[i]) for i in range(len(x))]
            data.sort(key=lambda tup: str(tup[3]))
            x = [elem[0] for elem in data]
            y = [elem[1] for elem in data]
            s = [elem[2] for elem in data]
            z = [elem[3] for elem in data]
        elif z:
            data = [(x[i], y[i], z[i]) for i in range(len(x))]
            data.sort(key=lambda tup: str(tup[2]))
            x = [elem[0] for elem in data]
            y = [elem[1] for elem in data]
            z = [elem[2] for elem in data]
        colors = self.get_colors()
        ax, fig = self._get_ax_fig(ax, size=(8, 6), set_axis_below=True, grid="y")
        if z:
            current_cat = z[0]
            idx = 0
            i = 0
            j = 1
            all_scatter = []
            all_categories = [current_cat]
            tmp_colors = []
            while j != len(z):
                while j < len(z) and z[j] == current_cat:
                    j += 1
                param = {
                    "alpha": 0.8,
                    "marker": "o",
                    "color": colors[idx],
                    "edgecolors": "black",
                }
                if s:
                    size = s[i:j]
                else:
                    size = 50
                all_scatter += [
                    ax.scatter(
                        x[i:j],
                        y[i:j],
                        s=size,
                        **self._update_dict(param, style_kwargs, idx),
                    )
                ]
                tmp_colors += [self._update_dict(param, style_kwargs, idx)["color"]]
                if j < len(z):
                    all_categories += [z[j]]
                    current_cat = z[j]
                    i = j
                    idx += 1
            ax.legend(
                [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="black",
                        markerfacecolor=color,
                        markersize=8,
                    )
                    for color in tmp_colors
                ],
                all_categories,
                bbox_to_anchor=[1, 0.5],
                loc="center left",
                title=title,
                labelspacing=1,
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        else:
            param = {
                "alpha": 0.8,
                "marker": "o",
                "color": colors[0],
                "edgecolors": "black",
            }
            if s:
                size = s
            else:
                size = 300
            ax.scatter(x, y, s=size, **self._update_dict(param, style_kwargs, 0))
        if reverse[0]:
            ax.set_xlim(
                max(x) + 0.1 * (1 + max(x) - min(x)),
                min(x) - 0.1 - 0.1 * (1 + max(x) - min(x)),
            )
        if reverse[1]:
            ax.set_ylim(
                max(y) + 0.1 * (1 + max(y) - min(y)),
                min(y) - 0.1 * (1 + max(y) - min(y)),
            )
        if plt_text:
            ax.set_xlabel(x_label, loc="right")
            ax.set_ylabel(y_label, loc="top")
            ax.spines["left"].set_position("center")
            ax.spines["bottom"].set_position("center")
            ax.spines["right"].set_color("none")
            ax.spines["top"].set_color("none")
            delta_x = (max(x) - min(x)) * 0.1
            delta_y = (max(y) - min(y)) * 0.1
            plt.text(
                max(x) + delta_x if reverse[0] else min(x) - delta_x,
                max(y) + delta_y if reverse[1] else min(y) - delta_y,
                "Modest",
                size=15,
                rotation=130.0,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round",
                    ec=self.get_colors(idx=0),
                    fc=self.get_colors(idx=0),
                    alpha=0.3,
                ),
            )
            plt.text(
                max(x) + delta_x if reverse[0] else min(x) - delta_x,
                min(y) - delta_y if reverse[1] else max(y) + delta_y,
                "Efficient",
                size=15,
                rotation=30.0,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round",
                    ec=self.get_colors(idx=1),
                    fc=self.get_colors(idx=1),
                    alpha=0.3,
                ),
            )
            plt.text(
                min(x) - delta_x if reverse[0] else max(x) + delta_x,
                max(y) + delta_y if reverse[1] else min(y) - delta_y,
                "Performant",
                size=15,
                rotation=-130.0,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round",
                    ec=self.get_colors(idx=2),
                    fc=self.get_colors(idx=2),
                    alpha=0.3,
                ),
            )
            plt.text(
                min(x) - delta_x if reverse[0] else max(x) + delta_x,
                min(y) - delta_y if reverse[1] else max(y) + delta_y,
                "Performant & Efficient",
                size=15,
                rotation=-30.0,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round",
                    ec=self.get_colors(idx=3),
                    fc=self.get_colors(idx=3),
                    alpha=0.3,
                ),
            )
        else:
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
        return ax
