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
from matplotlib.lines import Line2D

from verticapy.plotting._matplotlib.scatter import ScatterPlot


class OutliersPlot(ScatterPlot):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["outliers"]:
        return "outliers"

    @property
    def _compute_method(self) -> Literal["outliers"]:
        return "outliers"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (1, 2)

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "edgecolor": "black",
            "alpha": 0.9,
            "s": 40,
            "marker": "o",
        }
        self.init_style_line = {
            "marker": "o",
            "color": "black",
            "label": "Scatter",
            "markersize": 8,
        }
        self.init_style_line_inlier = {
            "lw": 4,
        }
        self.init_style_linewidth = {
            "linewidths": 2,
        }

    # Draw.

    def draw(
        self,
        cmap: str = None,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws an outliers contour plot using the Matplotlib API.
        """
        style_kwargs = self._fix_color_style_kwargs(style_kwargs)
        if not cmap:
            cmap = self.get_cmap(color=self.get_colors(idx=2))
        ax, fig, style_kwargs = self._get_ax_fig(
            ax, size=(8, 6), set_axis_below=False, grid=False, style_kwargs=style_kwargs
        )
        th = self.data["th"]
        X = self.data["map"]["X"]
        Y = self.data["map"]["Y"]
        Z = self.data["map"]["Z"]
        zvals = self.data["map"]["zvals"]
        if len(self.layout["columns"]) == 1:
            if Z.max() > th:
                cp = ax.contourf(X, Y, Z, cmap=cmap, levels=np.linspace(th, Z.max(), 8))
                fig.colorbar(cp).set_label("ZSCORE")
            ax.fill_between(zvals, [-1, -1], [1, 1], facecolor=self.layout["color"])
            for x0 in zvals:
                ax.plot([x0, x0], [-1, 1], color=self.layout["inliers_border_color"])
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
            ax.set_xlabel(self.layout["columns"][0])
            ax.set_yticks([], [])
            ax.set_ylim(-1, 1)
        elif len(self.layout["columns"]) == 2:
            ax.contourf(X, Y, Z, colors=self.layout["color"])
            ax.contour(
                X,
                Y,
                Z,
                levels=[th],
                colors=self.layout["inliers_border_color"],
                **self.init_style_linewidth,
            )
            cp = ax.contourf(X, Y, Z, cmap=cmap, levels=np.linspace(th, Z.max(), 8))
            fig.colorbar(cp).set_label("ZSCORE")
            ax.set_xlabel(self.layout["columns"][0])
            ax.set_ylabel(self.layout["columns"][1])
        for x, c in [
            (self.data["inliers"], self.layout["inliers_color"]),
            (self.data["outliers"], self.layout["outliers_color"]),
        ]:
            ax.scatter(
                x[:, 0],
                x[:, 1],
                color=c,
                **{
                    **self.init_style,
                    **style_kwargs,
                },
            )
        args = [[0], [0]]
        ax.legend(
            [
                Line2D(
                    *args,
                    color=self.layout["inliers_border_color"],
                    **self.init_style_line_inlier,
                ),
                Line2D(
                    *args,
                    **self.init_style_line,
                    markerfacecolor=self.layout["inliers_color"],
                ),
                Line2D(
                    *args,
                    **self.init_style_line,
                    markerfacecolor=self.layout["outliers_color"],
                ),
            ],
            ["threshold", "inliers", "outliers"],
            loc="center left",
            bbox_to_anchor=[1, 0.5],
            labelspacing=1,
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax
