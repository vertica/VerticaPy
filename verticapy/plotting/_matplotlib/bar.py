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
from typing import Literal, Optional, TYPE_CHECKING
import numpy as np

from matplotlib.axes import Axes

from verticapy._config.colors import get_colors
from verticapy._typing import SQLColumns

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame, vDataColumn

from verticapy.plotting._matplotlib.base import MatplotlibBase


class BarChart(MatplotlibBase):
    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["bar"]:
        return "bar"

    @property
    def _compute_method(self) -> Literal["1D"]:
        return "1D"

    def draw(self, ax: Optional[Axes] = None, **style_kwargs,) -> Axes:
        """
        Draws a histogram using the Matplotlib API.
        """
        ax, fig = self._get_ax_fig(
            ax,
            size=(min(int(len(self.data["x"]) / 1.8) + 1, 600), 6),
            set_axis_below=True,
            grid="y",
        )
        params = {"color": get_colors()[0], "alpha": 0.86}
        params = self._update_dict(params, style_kwargs)
        ax.bar(self.data["x"], self.data["y"], self.data["adj_width"], **params)
        ax.set_xlabel(self.layout["x"])
        if self.data["is_categorical"]:
            xticks = self.data["x"]
            xticks_label = self._format_string(self.data["labels"])
        else:
            xticks = [x - round(self.data["width"] / 2, 10) for x in self.data["x"]]
            xticks_label = xticks
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks_label, rotation=90)
        ax.set_ylabel(self.layout["method"])
        return ax


class BarChart2D(MatplotlibBase):
    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["bar"]:
        return "bar"

    @property
    def _compute_method(self) -> Literal["2D"]:
        return "2D"

    def draw(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        method: str = "density",
        of: str = "",
        max_cardinality: tuple[int, int] = (6, 6),
        h: tuple[Optional[float], Optional[float]] = (None, None),
        stacked: bool = False,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a 2D BarChart using the Matplotlib API.
        """
        if isinstance(columns, str):
            columns = [columns]
        colors = get_colors()
        self._compute_pivot_table(
            vdf, columns, method=method, of=of, h=h, max_cardinality=max_cardinality,
        )
        m, n = self.data["matrix"].shape
        bar_width = 0.5
        ax, fig = self._get_ax_fig(
            ax, size=(min(600, 3 * m) / 2 + 1, 6), set_axis_below=True, grid="y"
        )
        for i in range(0, n):
            params = {
                "x": [j for j in range(m)],
                "height": self.data["matrix"][:, i],
                "width": bar_width,
                "label": self.data["y_labels"][i],
                "alpha": 0.86,
                "color": colors[i % len(colors)],
            }
            params = self._update_dict(params, style_kwargs, i)
            if stacked:
                if i == 0:
                    bottom = np.array([0.0 for j in range(m)])
                else:
                    bottom += self.data["matrix"][:, i - 1].astype(float)
                params["bottom"] = bottom
            else:
                params["x"] = [j + (i - 1) * bar_width / (n - 1) for j in range(m)]
                params["width"] = bar_width / (n - 1)
            ax.bar(**params)
        if stacked:
            xticks = [j for j in range(m)]
        else:
            xticks = [j + bar_width / 2 - bar_width / 2 / (n - 1) for j in range(m)]
        ax.set_xticks(xticks)
        ax.set_xticklabels(self.data["x_labels"], rotation=90)
        ax.set_xlabel(columns[0])
        ax.set_ylabel(self._map_method(method, of)[0])
        ax.legend(title=columns[1], loc="center left", bbox_to_anchor=[1, 0.5])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax
