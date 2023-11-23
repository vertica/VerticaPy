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
from typing import Literal, Optional
import numpy as np

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from verticapy.plotting._matplotlib.base import MatplotlibBase


class ChampionChallengerPlot(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["champion"]:
        return "champion"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "alpha": 0.8,
            "marker": "o",
            "color": self.get_colors(idx=0),
            "edgecolors": "black",
        }
        self.init_style_line = {
            "marker": "o",
            "color": "black",
            "markersize": 8,
        }
        self.layout = {
            "x_label": "x",
            "y_label": "y",
            "reverse": (True, True),
            "title": "",
            **self.layout,
        }
        self.data = {"z": [], "s": [], **self.data}

    # Draw.

    def draw(
        self,
        plt_text: bool = True,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a machine learning bubble plot using the Matplotlib API.
        """
        n = len(self.data["x"])
        if len(self.data["s"]) > 0:
            s = np.array(
                [
                    min(250 + 5000 * si, 3000) if si != 0 else 250
                    for si in self.data["s"]
                ]
            )
        if len(self.data["c"]) > 0 and len(self.data["s"]) > 0:
            X = [
                [self.data["x"][i], self.data["y"][i], s[i], self.data["c"][i]]
                for i in range(n)
            ]
            X.sort(key=lambda tup: str(tup[3]))
        elif len(self.data["c"]) > 0:
            X = [
                [self.data["x"][i], self.data["y"][i], self.data["c"][i]]
                for i in range(n)
            ]
            X.sort(key=lambda tup: str(tup[2]))
        else:
            X = [
                [
                    self.data["x"][i],
                    self.data["y"][i],
                ]
                for i in range(n)
            ]
        X = np.array(X)
        x, y = X[:, 0].astype(float), X[:, 1].astype(float)
        if len(self.data["c"]) > 0 and len(self.data["s"]) > 0:
            s, c = X[:, 2].astype(float), X[:, 3]
        elif len(self.data["c"]) > 0:
            c = X[:, 2]
        colors = self.get_colors()
        ax, fig, style_kwargs = self._get_ax_fig(
            ax, size=(8, 6), set_axis_below=True, grid="y", style_kwargs=style_kwargs
        )
        if len(self.data["c"]) > 0:
            current_cat = c[0]
            idx, i, j = 0, 0, 1
            all_scatter, all_categories, tmp_colors = [], [current_cat], []
            while j != len(c):
                while j < len(c) and c[j] == current_cat:
                    j += 1
                kwargs = {
                    **self.init_style,
                    "color": colors[idx],
                }
                all_scatter += [
                    ax.scatter(
                        x[i:j],
                        y[i:j],
                        s=s[i:j] if len(self.data["s"]) > 0 else 50,
                        **self._update_dict(kwargs, style_kwargs, idx),
                    )
                ]
                tmp_colors += [self._update_dict(kwargs, style_kwargs, idx)["color"]]
                if j < len(c):
                    all_categories += [c[j]]
                    current_cat = c[j]
                    i = j
                    idx += 1
            ax.legend(
                [
                    Line2D(
                        [0],
                        [0],
                        markerfacecolor=color,
                        **self.init_style_line,
                    )
                    for color in tmp_colors
                ],
                all_categories,
                bbox_to_anchor=[1, 0.5],
                loc="center left",
                title=self.layout["title"],
                labelspacing=1,
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        else:
            ax.scatter(
                x,
                y,
                s=s if len(self.data["s"]) > 0 else 300,
                **self._update_dict(self.init_style, style_kwargs, 0),
            )
        if self.layout["reverse"][0]:
            ax.set_xlim(
                max(x) + 0.1 * (1 + max(x) - min(x)),
                min(x) - 0.1 - 0.1 * (1 + max(x) - min(x)),
            )
        if self.layout["reverse"][1]:
            ax.set_ylim(
                max(y) + 0.1 * (1 + max(y) - min(y)),
                min(y) - 0.1 * (1 + max(y) - min(y)),
            )
        if plt_text:
            ax.set_xlabel(self.layout["x_label"], loc="right")
            ax.set_ylabel(self.layout["y_label"], loc="top")
            ax.spines["left"].set_position("center")
            ax.spines["bottom"].set_position("center")
            ax.spines["right"].set_color("none")
            ax.spines["top"].set_color("none")
            delta_x = (max(x) - min(x)) * 0.1
            delta_y = (max(y) - min(y)) * 0.1
            plt.text(
                max(x) + delta_x if self.layout["reverse"][0] else min(x) - delta_x,
                max(y) + delta_y if self.layout["reverse"][1] else min(y) - delta_y,
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
                max(x) + delta_x if self.layout["reverse"][0] else min(x) - delta_x,
                min(y) - delta_y if self.layout["reverse"][1] else max(y) + delta_y,
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
                min(x) - delta_x if self.layout["reverse"][0] else max(x) + delta_x,
                max(y) + delta_y if self.layout["reverse"][1] else min(y) - delta_y,
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
                min(x) - delta_x if self.layout["reverse"][0] else max(x) + delta_x,
                min(y) - delta_y if self.layout["reverse"][1] else max(y) + delta_y,
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
            ax.set_xlabel(self.layout["x_label"])
            ax.set_ylabel(self.layout["y_label"])
        return ax
