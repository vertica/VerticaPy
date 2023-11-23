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
import scipy.spatial as scipy_st

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from verticapy.plotting._matplotlib.base import MatplotlibBase


class VoronoiPlot(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["voronoi"]:
        return "voronoi"

    @property
    def _compute_method(self) -> Literal["sample"]:
        return "sample"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (2, 2)

    # Styling Methods.

    def _init_style(self) -> None:
        """Must be overridden in child class"""
        self.init_style = {"show_vertices": False}
        self.init_style_scatter = {
            "color": "black",
            "s": 10,
            "alpha": 1,
            "zorder": 3,
        }
        self.init_style_crosses = {
            "color": "white",
            "s": 200,
            "linewidths": 5,
            "alpha": 1,
            "zorder": 4,
            "marker": "x",
        }

    # Draw.

    def draw(
        self,
        plot_crosses: bool = True,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a KMeans Voronoi plot using the Matplotlib API.
        """
        min_x = min(self.data["clusters"][:, 0])
        max_x = max(self.data["clusters"][:, 0])
        min_y = min(self.data["clusters"][:, 1])
        max_y = max(self.data["clusters"][:, 1])
        dummies_point = np.array(
            [
                [min_x - 999, min_y - 999],
                [min_x - 999, max_y + 999],
                [max_x + 999, min_y - 999],
                [max_x + 999, max_y + 999],
            ]
        )
        v = scipy_st.Voronoi(np.concatenate((self.data["clusters"], dummies_point)))
        scipy_st.voronoi_plot_2d(
            v, ax=ax, **self._update_dict(self.init_style, style_kwargs)
        )
        if not ax:
            ax = plt
            ax.xlabel(self.layout["columns"][0])
            ax.ylabel(self.layout["columns"][1])
        for idx, region in enumerate(v.regions):
            if not -1 in region:
                polygon = [v.vertices[i] for i in region]
                if "color" in style_kwargs:
                    if isinstance(style_kwargs["color"], str):
                        color = style_kwargs["color"]
                    else:
                        color = style_kwargs["color"][idx % len(style_kwargs["color"])]
                else:
                    color = self.get_colors(idx=idx)
                ax.fill(*zip(*polygon), alpha=0.4, color=color)
        ax.plot(self.data["clusters"][:, 0], self.data["clusters"][:, 1], "ko")
        ax.xlim(min_x - 0.05 * (max_x - min_x), max_x + 0.05 * (max_x - min_x))
        ax.ylim(min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y))
        if len(self.data["X"]) > 0:
            ax.scatter(
                self.data["X"][:, 0],
                self.data["X"][:, 1],
                **self.init_style_scatter,
            )
            if plot_crosses:
                ax.scatter(
                    self.data["clusters"][:, 0],
                    self.data["clusters"][:, 1],
                    **self.init_style_crosses,
                )
        return ax
