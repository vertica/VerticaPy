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
import copy
from typing import Literal, Optional, TYPE_CHECKING
import numpy as np

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from verticapy._config.colors import get_colors
from verticapy._typing import SQLColumns
from verticapy.errors import ParameterError

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame, vDataColumn

from verticapy.plotting._matplotlib.base import MatplotlibBase


class HorizontalBarChart(MatplotlibBase):
    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["barh"]:
        return "barh"

    @property
    def _compute_method(self) -> Literal["1D"]:
        return "1D"

    def draw(self, ax: Optional[Axes] = None, **style_kwargs,) -> Axes:
        """
        Draws a bar chart using the Matplotlib API.
        """
        ax, fig = self._get_ax_fig(
            ax, size=(10, min(int(len(self.data["x"]) / 1.8) + 1, 600)), grid="x"
        )
        params = {"color": get_colors()[0], "alpha": 0.86}
        params = self._update_dict(params, style_kwargs, 0)
        ax.barh(self.data["x"], self.data["y"], self.data["adj_width"], **params)
        ax.set_ylabel(self.layout["x"])
        if self.data["is_categorical"]:
            ax.set_yticks(self.data["x"])
            ax.set_yticklabels(self._format_string(self.data["labels"]), rotation=0)
        else:
            ax.set_yticks(
                [x - round(self.data["width"] / 2, 10) for x in self.data["x"]]
            )
        ax.set_xlabel(self.layout["method"])
        return ax


class HorizontalBarChart2D(MatplotlibBase):
    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["barh"]:
        return "barh"

    @property
    def _compute_method(self) -> Literal["2D"]:
        return "2D"

    def draw(
        self,
        bar_type: Literal["auto", "fully_stacked", "stacked", "density"],
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a 2D bar chart using the Matplotlib API.
        """
        colors = get_colors()
        n, m = self.data["matrix"].shape
        if bar_type == "fully_stacked":
            if self.layout["method"] != "density":
                raise ValueError(
                    "Fully Stacked Bar works only with the 'density' method."
                )
        if bar_type == "density":
            if self.layout["method"] != "density":
                raise ValueError("Pyramid Bar works only with the 'density' method.")
            if n != 2 and m != 2:
                raise ValueError(
                    "One of the 2 columns must have 2 categories to draw a Pyramid Bar."
                )
            if m != 2:
                self.data["matrix"] = np.transpose(self.data["matrix"])
                y_labels = self.data["x_labels"]
                self.data["x_labels"] = self.data["y_labels"]
                self.data["y_labels"] = y_labels
                self.layout["columns"] = [
                    self.layout["columns"][1],
                    self.layout["columns"][0],
                ]
        matrix = copy.deepcopy(self.data["matrix"])
        m, n = matrix.shape
        yticks = [j for j in range(m)]
        bar_height = 0.5
        if bar_type == "density":
            ax, fig = self._get_ax_fig(ax, size=(10, min(m * 3, 600) / 8 + 1), grid="x")
        else:
            ax, fig = self._get_ax_fig(ax, size=(10, min(m * 3, 600) / 2 + 1), grid="x")
        if bar_type == "fully_stacked":
            for i in range(0, m):
                matrix[i] /= sum(matrix[i])
        for i in range(0, n):
            current_column = matrix[:, i]
            params = {
                "y": [j for j in range(m)],
                "width": matrix[:, i],
                "height": bar_height,
                "label": self.data["y_labels"][i],
                "alpha": 0.86,
                "color": colors[i % len(colors)],
            }
            params = self._update_dict(params, style_kwargs, i)
            if bar_type in ("stacked", "fully_stacked"):
                if i == 0:
                    last_column = np.array([0.0 for j in range(m)])
                else:
                    last_column += matrix[:, i - 1].astype(float)
                params["left"] = last_column
            elif bar_type == "density":
                if i == 1:
                    current_column = [-j for j in current_column]
                params["width"] = current_column
                params["height"] = bar_height / 1.5
            else:
                params["y"] = [j + i * bar_height / n for j in range(m)]
                params["height"] = bar_height / n
            ax.barh(**params)
        if bar_type != "stacked":
            yticks = [j + bar_height / 2 - bar_height / 2 / n for j in range(m)]
        ax.set_yticks(yticks)
        ax.set_yticklabels(self.data["x_labels"])
        ax.set_ylabel(self.layout["columns"][0])
        ax.set_xlabel(self.layout["method"])
        if bar_type in ("density", "fully_stacked"):
            vals = ax.get_xticks()
            max_val = max([abs(x) for x in vals])
            ax.xaxis.set_major_locator(mticker.FixedLocator(vals))
            ax.set_xticklabels(["{:,.2%}".format(abs(x)) for x in vals])
        ax.legend(
            title=self.layout["columns"][1], loc="center left", bbox_to_anchor=[1, 0.5]
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax
