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

from verticapy.plotting._matplotlib.base import MatplotlibBase


class PCAPlot(MatplotlibBase):
    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["pca"]:
        return "pca"

    def draw_circle(
        self,
        x: list,
        y: list,
        variable_names: list = [],
        explained_variance: tuple[Optional[float], Optional[float]] = (None, None),
        dimensions: tuple[int, int] = (1, 2),
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a PCA circle plot using the Matplotlib API.
        """
        colors = self.get_colors()
        if "color" in style_kwargs:
            colors[0] = style_kwargs["color"]
        circle1 = plt.Circle((0, 0), 1, edgecolor=colors[0], facecolor="none")
        ax, fig = self._get_ax_fig(ax, size=(6, 6), set_axis_below=True, grid=False)
        n = len(x)
        ax.add_patch(circle1)
        for i in range(n):
            ax.arrow(
                0,
                0,
                x[i],
                y[i],
                head_width=0.05,
                color="black",
                length_includes_head=True,
            )
            ax.text(x[i], y[i], variable_names[i])
        ax.plot([-1.1, 1.1], [0.0, 0.0], linestyle="--", color="black")
        ax.plot([0.0, 0.0], [-1.1, 1.1], linestyle="--", color="black")
        if explained_variance[0]:
            dim1 = f"({round(explained_variance[0] * 100, 1)}%)"
        else:
            dim1 = ""
        ax.set_xlabel(f"Dim{dimensions[0]} {dim1}")
        if explained_variance[1]:
            dim1 = f"({round(explained_variance[1] * 100, 1)}%)"
        else:
            dim1 = ""
        ax.set_ylabel(f"Dim{dimensions[1]} {dim1}")
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        return ax

    def draw_var(
        self,
        x: list,
        y: list,
        variable_names: list = [],
        explained_variance: tuple[Optional[float], Optional[float]] = (None, None),
        dimensions: tuple[int, int] = (1, 2),
        bar_name: str = "",
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a PCA Variance Plot using the Matplotlib API.
        """
        colors = self.get_colors()
        if "color" in style_kwargs:
            colors[0] = style_kwargs["color"]
        ax, fig = self._get_ax_fig(ax, size=(6, 6), set_axis_below=True, grid=True)
        n = len(x)
        delta_y = (max(y) - min(y)) * 0.04
        delta_x = (max(x) - min(x)) * 0.04
        for i in range(n):
            ax.text(
                x[i], y[i] + delta_y, variable_names[i], horizontalalignment="center"
            )
        param = {"marker": "^", "s": 100, "edgecolors": "black"}
        if "c" not in style_kwargs:
            param["color"] = colors[0]
        img = ax.scatter(x, y, **self._update_dict(param, style_kwargs, 0))
        ax.plot(
            [min(x) - 5 * delta_x, max(x) + 5 * delta_x],
            [0.0, 0.0],
            linestyle="--",
            color="black",
        )
        ax.plot(
            [0.0, 0.0],
            [min(y) - 5 * delta_y, max(y) + 5 * delta_y],
            linestyle="--",
            color="black",
        )
        ax.set_xlim(min(x) - 5 * delta_x, max(x) + 5 * delta_x)
        ax.set_ylim(min(y) - 5 * delta_y, max(y) + 5 * delta_y)
        if explained_variance[0]:
            dim1 = f"({round(explained_variance[0] * 100, 1)}%)"
        else:
            dim1 = ""
        ax.set_xlabel(f"Dim{dimensions[0]} {dim1}")
        if explained_variance[1]:
            dim1 = f"({round(explained_variance[1] * 100, 1)}%)"
        else:
            dim1 = ""
        ax.set_ylabel(f"Dim{dimensions[1]} {dim1}")
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        if "c" in style_kwargs:
            fig.colorbar(img).set_label(bar_name)
        return ax
