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
import copy
from typing import Literal, Optional

import numpy as np

from matplotlib.axes import Axes
import matplotlib.ticker as mticker

from verticapy.plotting._matplotlib.base import MatplotlibBase


class HorizontalBarChart(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["barh"]:
        return "barh"

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
        Draws a bar chart using the Matplotlib API.
        """
        style_kwargs = self._fix_color_style_kwargs(style_kwargs)
        ax, fig, style_kwargs = self._get_ax_fig(
            ax,
            size=(10, min(int(len(self.data["x"]) / 1.8) + 1, 600)),
            grid="x",
            style_kwargs=style_kwargs,
        )
        ax.barh(
            self.data["x"],
            self.data["y"],
            self.data["adj_width"],
            **self._update_dict(self.init_style, style_kwargs, 0),
        )
        ax.set_ylabel(self.layout["column"])
        if self.data["is_categorical"]:
            ax.set_yticks(self.data["x"])
            ax.set_yticklabels(self._format_string(self.layout["labels"]), rotation=0)
        else:
            ax.set_yticks(
                [x - round(self.data["width"] / 2, 10) for x in self.data["x"]]
            )
        ax.set_xlabel(self.layout["method"])
        return ax


class HorizontalBarChart2D(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["barh"]:
        return "barh"

    @property
    def _compute_method(self) -> Literal["2D"]:
        return "2D"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {"height": 0.5, "alpha": 0.86}

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a 2D bar chart using the Matplotlib API.
        """
        style_kwargs = self._fix_color_style_kwargs(style_kwargs)
        colors = self.get_colors()
        n, m = self.data["X"].shape
        if self.layout["kind"] == "fully_stacked":
            if self.layout["method"] != "density":
                raise ValueError(
                    "Fully Stacked Bar works only with the 'density' method."
                )
        if self.layout["kind"] == "density":
            if self.layout["method"] != "density":
                raise ValueError("Pyramid Bar works only with the 'density' method.")
            if n != 2 and m != 2:
                raise ValueError(
                    "One of the 2 columns must have 2 categories to draw a Pyramid Bar."
                )
            if m != 2:
                self.data["X"] = np.transpose(self.data["X"])
                y_labels = self.layout["x_labels"]
                self.layout["x_labels"] = self.layout["y_labels"]
                self.layout["y_labels"] = y_labels
                self.layout["columns"] = [
                    self.layout["columns"][1],
                    self.layout["columns"][0],
                ]
        matrix = copy.deepcopy(self.data["X"])
        m, n = matrix.shape
        yticks = [j for j in range(m)]
        if self.layout["kind"] == "density":
            ax, fig, style_kwargs = self._get_ax_fig(
                ax,
                size=(10, min(m * 3, 600) / 8 + 1),
                grid="x",
                style_kwargs=style_kwargs,
            )
        else:
            ax, fig, style_kwargs = self._get_ax_fig(
                ax,
                size=(10, min(m * 3, 600) / 2 + 1),
                grid="x",
                style_kwargs=style_kwargs,
            )
        if self.layout["kind"] == "fully_stacked":
            for i in range(0, m):
                matrix[i] /= sum(matrix[i])
        for i in range(0, n):
            current_column = matrix[:, i]
            params = {
                "y": [j for j in range(m)],
                "width": matrix[:, i],
                "label": self.layout["y_labels"][i],
                "color": colors[i % len(colors)],
                **self.init_style,
            }
            params = self._update_dict(params, style_kwargs, i)
            if self.layout["kind"] in ("stacked", "fully_stacked"):
                if i == 0:
                    last_column = np.array([0.0 for j in range(m)])
                else:
                    last_column += matrix[:, i - 1].astype(float)
                params["left"] = last_column
            elif self.layout["kind"] == "density":
                if i == 1:
                    current_column = [-j for j in current_column]
                params["width"] = current_column
                params["height"] = self.init_style["height"] / 1.5
            else:
                params["y"] = [j + i * self.init_style["height"] / n for j in range(m)]
                params["height"] = self.init_style["height"] / n
            ax.barh(**params)
        if self.layout["kind"] not in ("stacked", "fully_stacked"):
            yticks = [
                j + self.init_style["height"] / 2 - self.init_style["height"] / 2 / n
                for j in range(m)
            ]
        ax.set_yticks(yticks)
        ax.set_yticklabels(self.layout["x_labels"])
        ax.set_ylabel(self.layout["columns"][0])
        ax.set_xlabel(self.layout["method"])
        if self.layout["kind"] in ("density", "fully_stacked"):
            vals = ax.get_xticks()
            max_val = max(abs(x) for x in vals)
            ax.xaxis.set_major_locator(mticker.FixedLocator(vals))
            ax.set_xticklabels(["{:,.2%}".format(abs(x)) for x in vals])
        ax.legend(
            title=self.layout["columns"][1], loc="center left", bbox_to_anchor=[1, 0.5]
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax
