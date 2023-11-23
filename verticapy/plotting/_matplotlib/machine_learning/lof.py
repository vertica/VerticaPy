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

from verticapy._typing import NoneType

from verticapy.plotting._matplotlib.base import MatplotlibBase


class LOFPlot(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["lof"]:
        return "lof"

    @property
    def _compute_method(self) -> Literal["sample"]:
        return "sample"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (2, 4)

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "s": 50,
            "edgecolors": "black",
        }

    def _get_colors(self, style_kwargs: dict) -> list:
        colors = []
        if "color" in style_kwargs:
            if isinstance(style_kwargs["color"], str):
                colors = [style_kwargs["color"]]
            else:
                colors = style_kwargs["color"]
            del style_kwargs["color"]
        elif "colors" in style_kwargs:
            if isinstance(style_kwargs["colors"], str):
                colors = [style_kwargs["colors"]]
            else:
                colors = style_kwargs["colors"]
            del style_kwargs["colors"]
        colors += self.get_colors()
        return colors

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a local outlier plot using the Matplotlib API.
        """
        colors = self._get_colors(style_kwargs=style_kwargs)
        param = {
            **self.init_style,
            "color": colors[0],
        }
        min_lof = np.nanmin(self.data["X"][:, -1])
        max_lof = np.nanmax(self.data["X"][:, -1])
        radius = 1000 * (self.data["X"][:, -1] - min_lof) / (max_lof - min_lof)
        x_label = self.layout["columns"][0]
        if 2 <= len(self.layout["columns"]) <= 3:
            X = self.data["X"][:, 0]
            if len(self.layout["columns"]) == 2:
                Y = np.array([0 for x in range(len(X))])
                size = (8, 2)
                y_label = None
            else:
                Y = self.data["X"][:, 1]
                size = (8, 6)
                y_label = self.layout["columns"][1]
            ax, fig, style_kwargs = self._get_ax_fig(
                ax, size=size, set_axis_below=True, grid=True, style_kwargs=style_kwargs
            )
            if isinstance(y_label, NoneType):
                ax.set_yticks([])
            else:
                ax.set_ylabel(y_label)
            kwargs = [
                {"label": "Data points", **self._update_dict(param, style_kwargs, 0)},
                {
                    "s": radius,
                    "label": "Outlier scores",
                    "facecolors": "none",
                    "color": colors[1],
                },
            ]
            for kwds in kwargs:
                ax.scatter(X, Y, **kwds)
        elif len(self.layout["columns"]) == 4:
            ax, fig, style_kwargs = self._get_ax_fig(
                ax, size=(8, 6), dim=3, style_kwargs=style_kwargs
            )
            ax.set_ylabel(self.layout["columns"][1])
            ax.set_zlabel(self.layout["columns"][2])
            kwargs = [
                {"label": "Data points", **self._update_dict(param, style_kwargs, 0)},
                {"s": radius, "facecolors": "none", "color": colors[1]},
            ]
            for kwds in kwargs:
                ax.scatter(
                    self.data["X"][:, 0],
                    self.data["X"][:, 1],
                    self.data["X"][:, 2],
                    **kwds,
                )
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        else:
            raise Exception(
                "LocalOutlierFactor Plot is available for a maximum of 3 columns."
            )
        ax.set_xlabel(x_label)
        return ax
