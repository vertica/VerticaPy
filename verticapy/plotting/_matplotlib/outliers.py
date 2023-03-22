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
import random
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
        return None

    # Draw.

    def draw(
        self,
        color: str = "orange",
        outliers_color: str = "black",
        inliers_color: str = "white",
        inliers_border_color: str = "red",
        cmap: str = None,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws an outliers contour plot using the Matplotlib API.
        """
        min0, max0 = self.data["min"][0], self.data["max"][0]
        avg0, std0 = self.data["avg"][0], self.data["std"][0]
        th = self.data["th"]
        if not (cmap):
            cmap = self.get_cmap(color=self.get_colors(idx=2))
        x_grid = np.linspace(min0, max0, 1000)
        ax, fig = self._get_ax_fig(ax, size=(8, 6), set_axis_below=False, grid=False)
        if len(self.layout["columns"]) == 1:
            y_grid = np.linspace(-1, 1, 1000)
            X, Y = np.meshgrid(x_grid, y_grid)
            Z = (X - avg0) / std0
            cp = ax.contourf(X, Y, Z, cmap=cmap, levels=np.linspace(th, Z.max(), 8))
            zvals = [-th * std0 + avg0, th * std0 + avg0]
            ax.fill_between(zvals, [-1, -1], [1, 1], facecolor=color)
            for x0 in zvals:
                ax.plot([x0, x0], [-1, 1], color=inliers_border_color)
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
            ax.set_xlabel(self.layout["columns"][0])
            ax.set_yticks([], [])
            ax.set_ylim(-1, 1)
            x = self.data["X"][:, 0]
            zs = abs(x - avg0) / std0
            for x0, c in [
                (x[abs(zs) <= th], inliers_color),
                (x[abs(zs) > th], outliers_color),
            ]:
                y0 = np.array([2 * (random.random() - 0.5) for i in range(len(x0))])
                ax.scatter(
                    x0, y0, color=c, **{**self.init_style, **style_kwargs,},
                )
        elif len(self.layout["columns"]) == 2:
            min1, max1 = self.data["min"][1], self.data["max"][1]
            avg1, std1 = self.data["avg"][1], self.data["std"][1]
            y_grid = np.linspace(min1, max1, 1000)
            X, Y = np.meshgrid(x_grid, y_grid)
            Z = np.sqrt(((X - avg0) / std0) ** 2 + ((Y - avg1) / std1) ** 2)
            ax.contourf(X, Y, Z, colors=color)
            ax.contour(
                X,
                Y,
                Z,
                levels=[th],
                colors=inliers_border_color,
                **self.init_style_linewidth,
            )
            cp = ax.contourf(X, Y, Z, cmap=cmap, levels=np.linspace(th, Z.max(), 8))
            ax.set_xlabel(self.layout["columns"][0])
            ax.set_ylabel(self.layout["columns"][1])
            x = self.data["X"][:, 0]
            y = self.data["X"][:, 1]
            X = [
                self.data["X"][
                    (abs(x - avg0) / std0 <= th) & (abs(y - avg1) / std1 <= th)
                ]
            ]
            X += [
                self.data["X"][
                    (abs(x - avg0) / std0 > th) | (abs(y - avg1) / std1 > th)
                ]
            ]
            for i, c in enumerate([inliers_color, outliers_color]):
                ax.scatter(
                    X[i][:, 0],
                    X[i][:, 1],
                    color=c,
                    **{**self.init_style, **style_kwargs,},
                )
        fig.colorbar(cp).set_label("ZSCORE")
        args = [[0], [0]]
        ax.legend(
            [
                Line2D(
                    *args, color=inliers_border_color, **self.init_style_line_inlier
                ),
                Line2D(*args, **self.init_style_line, markerfacecolor=inliers_color),
                Line2D(*args, **self.init_style_line, markerfacecolor=outliers_color),
            ],
            ["threshold", "inliers", "outliers"],
            loc="center left",
            bbox_to_anchor=[1, 0.5],
            labelspacing=1,
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax
