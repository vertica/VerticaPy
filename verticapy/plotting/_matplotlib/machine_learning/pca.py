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
import matplotlib.pyplot as plt

from verticapy._typing import NoneType

from verticapy.plotting._matplotlib.base import MatplotlibBase


class PCACirclePlot(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["pca_circle"]:
        return "pca_circle"

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
        self.layout["columns"] = self._clean_quotes(self.layout["columns"])

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a PCA circle plot using the Matplotlib API.
        """
        colors = self.get_colors()
        if "color" in style_kwargs:
            colors[0] = style_kwargs["color"]
        circle1 = plt.Circle((0, 0), 1, **self.init_style_circle)
        ax, fig, style_kwargs = self._get_ax_fig(
            ax, size=(6, 6), set_axis_below=True, grid=False, style_kwargs=style_kwargs
        )
        n = len(self.data["x"])
        ax.add_patch(circle1)
        for i in range(n):
            ax.arrow(
                0,
                0,
                self.data["x"][i],
                self.data["y"][i],
                **self.init_style,
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


class PCAScreePlot(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["pca_scree"]:
        return "pca_scree"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style_bar = {"color": self.get_colors(idx=0), "alpha": 0.86}
        self.init_style_scree = {
            "color": "black",
            "linewidth": 2,
            "marker": "o",
            "markevery": 0.05,
            "markersize": 7,
            "markeredgecolor": "black",
            "markerfacecolor": "white",
        }
        self.init_style_line = {
            "c": "r",
            "linestyle": "--",
        }

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a PCA Scree plot using the Matplotlib API.
        """
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
            **self._update_dict(self.init_style_bar, style_kwargs),
        )
        n, dt = len(self.data["x"]), 0.6
        if self.layout["plot_scree"]:
            dt = 1.0
            ax.plot(
                self.data["x"],
                self.data["y"],
                self.data["adj_width"],
                **self.init_style_scree,
            )
        if self.layout["plot_line"]:
            ax.plot([0.5, n + 0.5], [1 / n * 100, 1 / n * 100], **self.init_style_line)
        ax.set_xlabel(self.layout["x_label"])
        ax.set_ylabel(self.layout["y_label"])
        ax.set_xticks([i + 1 for i in range(n)])
        ax.set_xticklabels(self.layout["labels"], rotation=90)
        for i in range(n):
            text_str = f"{round(self.data['y'][i], 1)}%"
            ax.text(
                i + dt,
                self.data["y"][i] + 1,
                text_str,
            )
        ax.set_xlim(0.5, n + 0.5)
        ax.set_title(self.layout["title"])
        return ax


class PCAVarPlot(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["pca_var"]:
        return "pca_var"

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

    def _get_final_style(self, style_kwargs: dict) -> dict:
        style_kwargs_updated = copy.deepcopy(
            self._update_dict(self.init_style, style_kwargs, 0)
        )
        style_kwargs_updated["c"] = self.data["c"]
        if "color" in style_kwargs_updated and not (
            isinstance(style_kwargs_updated, NoneType)
        ):
            del style_kwargs_updated["color"]
        if "colors" in style_kwargs_updated and not (
            isinstance(style_kwargs_updated, NoneType)
        ):
            del style_kwargs_updated["colors"]
        if "cmap" not in style_kwargs_updated:
            style_kwargs_updated["cmap"] = self.get_cmap(
                color=[
                    self.get_colors(idx=0),
                    self.get_colors(idx=1),
                    self.get_colors(idx=2),
                ]
            )
        return style_kwargs_updated

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a PCA variance plot using the Matplotlib API.
        """
        ax, fig, style_kwargs = self._get_ax_fig(
            ax, size=(6, 6), set_axis_below=True, grid=True, style_kwargs=style_kwargs
        )
        n = len(self.data["x"])
        min_x, max_x = np.nanmin(self.data["x"]), np.nanmax(self.data["x"])
        min_y, max_y = np.nanmin(self.data["y"]), np.nanmax(self.data["y"])
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
            **self._get_final_style(style_kwargs=style_kwargs),
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
        if self.layout["method"] in ("cos2", "contrib"):
            fig.colorbar(img).set_label(self.layout["method"])
        return ax
