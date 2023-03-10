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
import warnings
from typing import Optional, TYPE_CHECKING
import numpy as np

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from verticapy._config.colors import get_colors
import verticapy._config.config as conf
from verticapy._typing import SQLColumns
from verticapy.errors import ParameterError

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame, vDataColumn

from verticapy.plotting._matplotlib.base import MatplotlibBase


class BarChart(MatplotlibBase):
    def bar(
        self,
        vdc: "vDataColumn",
        method: str = "density",
        of: Optional[str] = None,
        max_cardinality: int = 6,
        nbins: int = 0,
        h: float = 0.0,
        ax: Optional[Axes] = None,
        **style_kwds,
    ) -> Axes:
        """
        Draws a histogram using the Matplotlib API.
        """
        self._compute_plot_params(vdc, method, of, max_cardinality, nbins, h)
        is_numeric = vdc.isnum()
        ax, fig = self._get_ax_fig(
            ax,
            size=(min(int(len(self.data["x"]) / 1.8) + 1, 600), 6),
            set_axis_below=True,
            grid="y",
        )
        params = {"color": get_colors()[0], "alpha": 0.86}
        params = self.updated_dict(params, style_kwds)
        ax.bar(self.data["x"], self.data["y"], self.data["adj_width"], **params)
        ax.set_xlabel(vdc._alias)
        if self.data["is_categorical"]:
            xticks = self.data["x"]
            xticks_label = self._format_string(self.data["labels"])
        else:
            xticks = [x - round(self.data["width"] / 2, 10) for x in self.data["x"]]
            xticks_label = xticks
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks_label, rotation=90)
        ax.set_ylabel(self._map_method(method, of)[0])
        return ax

    def bar2D(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        method: str = "density",
        of: str = "",
        max_cardinality: tuple[int, int] = (6, 6),
        h: tuple[Optional[float], Optional[float]] = (None, None),
        stacked: bool = False,
        ax: Optional[Axes] = None,
        **style_kwds,
    ) -> Axes:
        """
        Draws a 2D BarChart using the Matplotlib API.
        """
        if isinstance(columns, str):
            columns = [columns]
        colors = get_colors()
        matrix, x_labels, y_labels = self._compute_pivot_table(
            vdf, columns, method=method, of=of, h=h, max_cardinality=max_cardinality,
        )[0:3]
        m, n = matrix.shape
        bar_width = 0.5
        ax, fig = self._get_ax_fig(
            ax, size=(min(600, 3 * m) / 2 + 1, 6), set_axis_below=True, grid="y"
        )
        for i in range(0, n):
            params = {
                "x": [j for j in range(m)],
                "height": matrix[:, i],
                "width": bar_width,
                "label": y_labels[i],
                "alpha": 0.86,
                "color": colors[i % len(colors)],
            }
            params = self.updated_dict(params, style_kwds, i)
            if stacked:
                if i == 0:
                    bottom = np.array([0.0 for j in range(m)])
                else:
                    bottom += matrix[:, i - 1].astype(float)
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
        ax.set_xticklabels(x_labels, rotation=90)
        ax.set_xlabel(columns[0])
        ax.set_ylabel(self._map_method(method, of)[0])
        ax.legend(title=columns[1], loc="center left", bbox_to_anchor=[1, 0.5])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax

    def multiple_hist(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        method: str = "density",
        of: str = "",
        h: float = 0.0,
        ax: Optional[Axes] = None,
        **style_kwds,
    ) -> Axes:
        """
        Draws a muli-histogram chart using the Matplotlib API.
        """
        if isinstance(columns, str):
            columns = [columns]
        colors = get_colors()
        if len(columns) > 5:
            raise ParameterError(
                "The number of column must be <= 5 to use 'multiple_hist' method"
            )
        else:
            ax, fig = self._get_ax_fig(ax, size=(8, 6), set_axis_below=True, grid="y")
            alpha, all_columns, all_h = 1, [], []
            if h <= 0:
                for idx, column in enumerate(columns):
                    all_h += [vdf[column].numh()]
                h = min(all_h)
            for idx, column in enumerate(columns):
                if vdf[column].isnum():
                    [x, y, z, h, is_categorical] = self._compute_plot_params(
                        vdf[column], method=method, of=of, max_cardinality=1, h=h
                    )
                    h = h / 0.94
                    param = {"color": colors[idx % len(colors)]}
                    plt.bar(
                        x,
                        y,
                        h,
                        label=column,
                        alpha=alpha,
                        **self.updated_dict(param, style_kwds, idx),
                    )
                    alpha -= 0.2
                    all_columns += [columns[idx]]
                else:
                    if vdf._vars["display"]["print_info"]:
                        warning_message = (
                            f"The Virtual Column {column} is not numerical."
                            " Its histogram will not be drawn."
                        )
                        warnings.warn(warning_message, Warning)
            ax.set_xlabel(", ".join(all_columns))
            ax.set_ylabel(self._map_method(method, of)[0])
            ax.legend(title="columns", loc="center left", bbox_to_anchor=[1, 0.5])
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            return ax
