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

from verticapy.plotting._matplotlib.base import MatplotlibBase


class RegressionPlot(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["regression"]:
        return "regression"

    @property
    def _compute_method(self) -> Literal["sample"]:
        return "sample"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (2, 3)

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "marker": "o",
            "color": self.get_colors(idx=0),
            "s": 50,
            "edgecolors": "black",
        }

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a regression plot using the Matplotlib API.
        """
        style_kwargs = self._fix_color_style_kwargs(style_kwargs)
        x0 = self.data["X"][:, 0]
        y0 = self.data["X"][:, 1]
        min_reg_x, max_reg_x = min(x0), max(x0)
        if len(self.layout["columns"]) == 2:
            ax, fig, style_kwargs = self._get_ax_fig(
                ax,
                size=(8, 6),
                set_axis_below=True,
                grid=True,
                style_kwargs=style_kwargs,
            )
            x_reg = [min_reg_x, max_reg_x]
            y_reg = [
                self.data["coef"][0] + self.data["coef"][1] * item for item in x_reg
            ]
            ax.plot(x_reg, y_reg, alpha=1, color="black")
            ax.scatter(
                x0,
                y0,
                **self._update_dict(self.init_style, style_kwargs, 0),
            )
            ax.set_xlabel(self.layout["columns"][0])
            ax.set_ylabel(self.layout["columns"][1])
        elif len(self.layout["columns"]) == 3:
            z0 = self.data["X"][:, 2]
            step_x = (max_reg_x - min_reg_x) / 40.0
            min_reg_y, max_reg_y = min(y0), max(y0)
            step_y = (max_reg_y - min_reg_y) / 40.0
            X_reg = (
                np.arange(min_reg_x - 5 * step_x, max_reg_x + 5 * step_x, step_x)
                if (step_x > 0)
                else [max_reg_x]
            )
            Y_reg = (
                np.arange(min_reg_y - 5 * step_y, max_reg_y + 5 * step_y, step_y)
                if (step_y > 0)
                else [max_reg_y]
            )
            X_reg, Y_reg = np.meshgrid(X_reg, Y_reg)
            Z_reg = (
                self.data["coef"][0]
                + self.data["coef"][1] * X_reg
                + self.data["coef"][2] * Y_reg
            )
            ax, fig, style_kwargs = self._get_ax_fig(
                ax, size=(8, 6), dim=3, style_kwargs=style_kwargs
            )
            ax.plot_surface(
                X_reg, Y_reg, Z_reg, rstride=1, cstride=1, alpha=0.5, color="gray"
            )
            ax.scatter(
                x0,
                y0,
                z0,
                **self._update_dict(self.init_style, style_kwargs, 0),
            )
            ax.set_xlabel(self.layout["columns"][0])
            ax.set_ylabel(self.layout["columns"][1])
            ax.set_zlabel(
                self.layout["columns"][2]
                + " = f("
                + self.layout["columns"][0]
                + ", "
                + self.layout["columns"][1]
                + ")"
            )
        else:
            raise ValueError("The number of predictors is too big to draw the plot.")
        return ax
