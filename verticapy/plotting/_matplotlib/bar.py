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

from verticapy.plotting._matplotlib.base import MatplotlibBase


class BarChart(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["bar"]:
        return "bar"

    @property
    def _compute_method(self) -> Literal["1D"]:
        return "1D"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {"color": self.get_colors(idx=0), "alpha": 0.86}

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a histogram using the Matplotlib API.
        """
        style_kwargs = self._fix_color_style_kwargs(style_kwargs)
        ax, fig, style_kwargs = self._get_ax_fig(
            ax,
            size=(min(int(len(self.data["x"]) / 1.8) + 1, 600), 6),
            set_axis_below=True,
            grid="y",
            style_kwargs=style_kwargs,
        )
        ax.bar(
            self.data["x"],
            self.data["y"],
            self.data["adj_width"],
            **self._update_dict(self.init_style, style_kwargs),
        )
        ax.set_xlabel(self.layout["column"])
        if self.data["is_categorical"]:
            xticks = self.data["x"]
            xticks_label = self._format_string(self.layout["labels"])
        else:
            xticks = [li[0] for li in self.layout["labels"]] + [
                self.layout["labels"][-1][-1]
            ]
            xticks_label = xticks
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks_label, rotation=90)
        ax.set_ylabel(self.layout["method"])
        return ax


class BarChart2D(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["bar"]:
        return "bar"

    @property
    def _compute_method(self) -> Literal["2D"]:
        return "2D"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {"width": 0.5, "alpha": 0.86}

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a 2D BarChart using the Matplotlib API.
        """
        style_kwargs = self._fix_color_style_kwargs(style_kwargs)
        colors = self.get_colors()
        m, n = self.data["X"].shape
        ax, fig, style_kwargs = self._get_ax_fig(
            ax,
            size=(min(600, 3 * m) / 2 + 1, 6),
            set_axis_below=True,
            grid="y",
            style_kwargs=style_kwargs,
        )
        for i in range(0, n):
            params = {
                "x": [j for j in range(m)],
                "height": self.data["X"][:, i],
                "label": self.layout["y_labels"][i],
                "color": colors[i % len(colors)],
                **self.init_style,
            }
            params = self._update_dict(params, style_kwargs, i)
            if self.layout["kind"] == "stacked":
                if i == 0:
                    bottom = np.array([0.0 for j in range(m)])
                else:
                    bottom += self.data["X"][:, i - 1].astype(float)
                params["bottom"] = bottom
            else:
                params["x"] = [j + i * self.init_style["width"] / n for j in range(m)]
                params["width"] = self.init_style["width"] / n
            ax.bar(**params)
        if self.layout["kind"] == "stacked":
            xticks = [j for j in range(m)]
        else:
            xticks = [
                j + self.init_style["width"] / 2 - self.init_style["width"] / 2 / n
                for j in range(m)
            ]
        ax.set_xticks(xticks)
        ax.set_xticklabels(self.layout["x_labels"], rotation=90)
        ax.set_xlabel(self.layout["columns"][0])
        ax.set_ylabel(self.layout["method"])
        ax.legend(
            title=self.layout["columns"][1], loc="center left", bbox_to_anchor=[1, 0.5]
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax
