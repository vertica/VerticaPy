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
import numpy as np

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from verticapy.plotting._matplotlib.base import MatplotlibBase


class ScatterMatrix(MatplotlibBase):

    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["scatter"]:
        return "scatter_matrix"

    @property
    def _compute_method(self) -> Literal["matrix"]:
        return "matrix"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "edgecolor": "black",
            "alpha": 0.9,
            "s": 40,
            "marker": "o",
        }
        return None

    # Draw.

    def draw(self, **style_kwargs,) -> Axes:
        """
        Draws a scatter matrix using the Matplotlib API.
        """
        n = len(self.layout["columns"])
        if conf._get_import_success("jupyter"):
            figsize = min(1.5 * (n + 1), 500), min(1.5 * (n + 1), 500)
            fig, axes = plt.subplots(nrows=n, ncols=n, figsize=figsize,)
        else:
            figsize = min(int((n + 1) / 1.1), 500), min(int((n + 1) / 1.1), 500)
            fig, axes = plt.subplots(nrows=n, ncols=n, figsize=figsize,)
        for i in range(n):
            axes[-1][i].set_xlabel(self.layout["columns"][i], rotation=90)
            axes[i][0].set_ylabel(self.layout["columns"][i], rotation=0)
            axes[i][0].yaxis.get_label().set_ha("right")
            for j in range(n):
                axes[i][j].get_xaxis().set_ticks([])
                axes[i][j].get_yaxis().set_ticks([])
                if self.layout["columns"][i] == self.layout["columns"][j]:
                    kwargs = {
                        "color": self.get_colors(d=style_kwargs, idx=0),
                        "edgecolor": "black",
                    }
                    if "edgecolor" in style_kwargs:
                        kwargs["edgecolor"] = style_kwargs["edgecolor"]
                    axes[i, j].bar(
                        self.data["hist"][self.layout["columns"][i]]["x"],
                        self.data["hist"][self.layout["columns"][i]]["y"],
                        self.data["hist"][self.layout["columns"][i]]["width"],
                        **kwargs,
                    )
                else:
                    kwargs = {
                        "color": self.get_colors(d=style_kwargs, idx=1),
                        **self.init_style,
                    }
                    axes[i, j].scatter(
                        self.data["scatter"]["X"][:, j],
                        self.data["scatter"]["X"][:, i],
                        **self._update_dict(kwargs, style_kwargs, 1),
                    )
        return axes


class ScatterPlot(MatplotlibBase):

    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["scatter"]:
        return "scatter"

    @property
    def _compute_method(self) -> Literal["sample"]:
        return "sample"

    # Styling Methods.

    def _init_style(self) -> None:
        """Must be overridden in child class"""
        self.init_style = {
            "s": 50,
            "edgecolors": "black",
            "marker": "o",
        }
        return None

    # Draw.

    def draw(
        self,
        bbox: Optional[tuple] = None,
        img: Optional[str] = None,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a scatter plot using the Matplotlib API.
        """
        n, m = self.data["X"].shape
        if not (ax):
            if m == 2:
                ax, fig = self._get_ax_fig(
                    ax, size=(8, 6), set_axis_below=True, grid=True
                )
            else:
                if conf._get_import_success("jupyter"):
                    plt.figure(figsize=(8, 6))
                ax = plt.axes(projection="3d")
        args = [self.data["X"][:, i] for i in range(m)]
        kwargs = self._update_dict(self.init_style, style_kwargs, 0)
        if self.layout["has_size"]:
            s_min, s_max = min(self.data["s"]), max(self.data["s"])
            if s_max != s_min:
                kwargs["s"] = 1000 * (self.data["s"] - s_min) / (s_max - s_min) + 1e-50
        if self.layout["has_category"]:
            uniques = self._format_string(np.unique(self.data["c"]), th=20)
            colors = np.array(["#FFFFFF" for c in self.data["c"]], dtype=object)
            marker = style_kwargs["marker"] if "marker" in style_kwargs else "o"
            legend = []
            for i, c in enumerate(uniques):
                color = self.get_colors(idx=i)
                colors[self.data["c"] == c] = color
                legend += [
                    Line2D(
                        [0],
                        [0],
                        marker=marker,
                        color="w",
                        markerfacecolor=color,
                        label=c,
                        markersize=8,
                    )
                ]
            kwargs["color"] = colors
        elif self.layout["has_cmap"]:
            kwargs["color"] = None
            kwargs["c"] = self.data["c"]
            if "cmap" not in kwargs:
                kwargs["cmap"] = self.get_cmap(idx=0)
        sc = ax.scatter(*args, **kwargs)
        ax.set_xlabel(self.layout["columns"][0])
        bbox_to_anchor = [1, 0.5]
        if m > 1:
            ax.set_ylabel(self.layout["columns"][1])
        if m > 2:
            ax.set_zlabel(self.layout["columns"][2])
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            bbox_to_anchor = [1.1, 0.5]
        if m == 2:
            if bbox:
                ax.set_xlim(bbox[0], bbox[1])
                ax.set_ylim(bbox[2], bbox[3])
            if img:
                ax.imshow(im, extent=bbox)
        if self.layout["has_category"]:
            ax.legend(
                handles=legend,
                loc="center left",
                title=self.layout["c"],
                bbox_to_anchor=bbox_to_anchor,
            )
        elif self.layout["has_cmap"]:
            fig.colorbar(sc).set_label(self.layout["c"])
        return ax
