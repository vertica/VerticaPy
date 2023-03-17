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


class PCACirclePlot(MatplotlibBase):

    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["pca"]:
        return "pca"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "head_width": 0.05,
            "color": "black",
            "length_includes_head": True,
        }
        self.init_style_circle = {
            "edgecolor": self.get_colors(idx=0),
            "facecolor": "none",
        }
        self.init_style_plot = {"linestyle": "--", "color": "black"}
        return None

    # Draw.

    def draw(self, ax: Optional[Axes] = None, **style_kwargs,) -> Axes:
        """
        Draws a PCA circle plot using the Matplotlib API.
        """
        colors = self.get_colors()
        if "color" in style_kwargs:
            colors[0] = style_kwargs["color"]
        circle1 = plt.Circle((0, 0), 1, **self.init_style_circle)
        ax, fig = self._get_ax_fig(ax, size=(6, 6), set_axis_below=True, grid=False)
        n = len(self.data["x"])
        ax.add_patch(circle1)
        for i in range(n):
            ax.arrow(
                0, 0, self.data["x"][i], self.data["y"][i], **self.init_style,
            )
            ax.text(self.data["x"][i], self.data["y"][i], self.layout["columns"][i])
        ax.plot([-1.1, 1.1], [0.0, 0.0], **self.init_style_plot)
        ax.plot([0.0, 0.0], [-1.1, 1.1], **self.init_style_plot)
        if self.data["explained_variance"][0]:
            dim1 = f"({round(self.data['explained_variance'][0] * 100, 1)}%)"
        else:
            dim1 = ""
        ax.set_xlabel(f"Dim{self.data['dim'][0]} {dim1}")
        if self.data["explained_variance"][1]:
            dim1 = f"({round(self.data['explained_variance'][1] * 100, 1)}%)"
        else:
            dim1 = ""
        ax.set_ylabel(f"Dim{self.data['dim'][1]} {dim1}")
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        return ax


class PCAVarPlot(MatplotlibBase):

    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["pca"]:
        return "pca"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "marker": "^",
            "s": 100,
            "edgecolors": "black",
            "color": self.get_colors(idx=0),
        }
        self.init_style_plot = {
            "linestyle": "--",
            "color": "black",
        }
        return None

    # Draw.

    def draw(self, ax: Optional[Axes] = None, **style_kwargs,) -> Axes:
        """
        Draws a PCA Variance Plot using the Matplotlib API.
        """
        ax, fig = self._get_ax_fig(ax, size=(6, 6), set_axis_below=True, grid=True)
        n = len(self.data["x"])
        min_x, max_x = min(self.data["x"]), max(self.data["x"])
        min_y, max_y = min(self.data["y"]), max(self.data["y"])
        delta_x = (max_x - min_x) * 0.04
        delta_y = (max_y - min_y) * 0.04
        for i in range(n):
            ax.text(
                self.data["x"][i],
                self.data["y"][i] + delta_y,
                self.layout["columns"][i],
                horizontalalignment="center",
            )
        img = ax.scatter(
            self.data["x"],
            self.data["y"],
            **self._update_dict(self.init_style, style_kwargs, 0),
        )
        ax.plot(
            [min_x - 5 * delta_x, max_x + 5 * delta_x],
            [0.0, 0.0],
            **self.init_style_plot,
        )
        ax.plot(
            [0.0, 0.0],
            [min_y - 5 * delta_y, max_y + 5 * delta_y],
            **self.init_style_plot,
        )
        ax.set_xlim(min_x - 5 * delta_x, max_x + 5 * delta_x)
        ax.set_ylim(min_y - 5 * delta_y, max_y + 5 * delta_y)
        if self.data["explained_variance"][0]:
            dim1 = f"({round(self.data['explained_variance'][0] * 100, 1)}%)"
        else:
            dim1 = ""
        ax.set_xlabel(f"Dim{self.data['dim'][0]} {dim1}")
        if self.data["explained_variance"][1]:
            dim1 = f"({round(self.data['explained_variance'][1] * 100, 1)}%)"
        else:
            dim1 = ""
        ax.set_ylabel(f"Dim{self.data['dim'][1]} {dim1}")
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        if "c" in style_kwargs:
            fig.colorbar(img).set_label(self.layout["method"])
        return ax
