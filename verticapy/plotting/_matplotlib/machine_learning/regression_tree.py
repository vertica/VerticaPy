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

from matplotlib.axes import Axes

from verticapy.plotting._matplotlib.base import MatplotlibBase


class RegressionTreePlot(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["regression_tree"]:
        return "regression_tree"

    @property
    def _compute_method(self) -> Literal["sample"]:
        return "sample"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (3, 3)

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "marker": "o",
            "color": self.get_colors(idx=0),
            "s": 50,
            "edgecolors": "black",
        }

    @staticmethod
    def _get_final_color(style_kwargs: dict) -> str:
        color = "black"
        if "color" in style_kwargs:
            if (
                not isinstance(style_kwargs["color"], str)
                and len(style_kwargs["color"]) > 1
            ):
                color = style_kwargs["color"][1]
        return color

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a regression tree plot using the Matplotlib API.
        """
        ax, fig, style_kwargs = self._get_ax_fig(
            ax, size=(8, 6), set_axis_below=True, grid=True, style_kwargs=style_kwargs
        )
        X = self.data["X"][self.data["X"][:, 0].argsort()]
        x0 = X[:, 0]
        x1 = X[:, 0]
        y0 = X[:, 2]
        y1 = X[:, 1]
        x0, y0 = zip(*sorted(zip(x0, y0)))
        x1, y1 = zip(*sorted(zip(x1, y1)))
        ax.step(x1, y1, color=self._get_final_color(style_kwargs=style_kwargs))
        ax.scatter(x0, y0, **self._update_dict(self.init_style, style_kwargs))
        ax.set_xlabel(self.layout["columns"][0])
        ax.set_ylabel(self.layout["columns"][1])
        return ax
