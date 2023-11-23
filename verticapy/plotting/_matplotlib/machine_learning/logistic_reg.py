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


class LogisticRegressionPlot(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["logit"]:
        return "logit"

    @property
    def _compute_method(self) -> Literal["sample"]:
        return "sample"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (2, 3)

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style_0 = {
            "marker": "o",
            "s": 50,
            "color": self.get_colors(idx=0),
            "edgecolors": "black",
            "alpha": 0.8,
        }
        self.init_style_1 = {
            "marker": "o",
            "s": 50,
            "color": self.get_colors(idx=1),
            "edgecolors": "black",
        }
        self.init_style_Z = {
            "rstride": 1,
            "cstride": 1,
            "alpha": 0.5,
            "color": "gray",
        }

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a logistic regression plot using the Matplotlib API.
        """

        def logit(x: float) -> float:
            return 1 / (1 + np.exp(-x))

        x, z = self.data["X"][:, 0], self.data["X"][:, -1]
        x0, x1 = x[z == 0], x[z == 1]
        min_logit_x, max_logit_x = min(self.data["X"][:, 0]), max(self.data["X"][:, 0])
        step_x = (max_logit_x - min_logit_x) / 40.0
        if len(self.layout["columns"]) == 2:
            ax, fig, style_kwargs = self._get_ax_fig(
                ax,
                size=(8, 6),
                set_axis_below=True,
                grid=True,
                style_kwargs=style_kwargs,
            )
            x_logit = (
                np.arange(min_logit_x - 5 * step_x, max_logit_x + 5 * step_x, step_x)
                if (step_x > 0)
                else np.array([max_logit_x])
            )
            y_logit = logit(self.data["coef"][0] + self.data["coef"][1] * x_logit)
            ax.plot(x_logit, y_logit, alpha=1, color="black")
            all_scatter = []
            for i, x, s in [(0, x0, self.init_style_0), (1, x1, self.init_style_1)]:
                all_scatter += [
                    ax.scatter(
                        x,
                        logit(self.data["coef"][0] + self.data["coef"][1] * x),
                        **self._update_dict(s, style_kwargs, i),
                    )
                ]
            ax.set_xlabel(self.layout["columns"][0])
            ax.set_ylabel(self.layout["columns"][-1])
            ax.legend(
                all_scatter,
                [0, 1],
                scatterpoints=1,
                loc="center left",
                bbox_to_anchor=[1, 0.5],
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        elif len(self.layout["columns"]) == 3:
            y = self.data["X"][:, 1]
            y0, y1 = y[z == 0], y[z == 1]
            min_logit_y, max_logit_y = (
                min(self.data["X"][:, 1]),
                max(self.data["X"][:, 1]),
            )
            step_y = (max_logit_y - min_logit_y) / 40.0
            X_logit = (
                np.arange(min_logit_x - 5 * step_x, max_logit_x + 5 * step_x, step_x)
                if (step_x > 0)
                else np.array([max_logit_x])
            )
            Y_logit = (
                np.arange(min_logit_y - 5 * step_y, max_logit_y + 5 * step_y, step_y)
                if (step_y > 0)
                else np.array([max_logit_y])
            )
            X_logit, Y_logit = np.meshgrid(X_logit, Y_logit)
            Z_logit = 1 / (
                1
                + np.exp(
                    -(
                        self.data["coef"][0]
                        + self.data["coef"][1] * X_logit
                        + self.data["coef"][2] * Y_logit
                    )
                )
            )
            ax, fig, style_kwargs = self._get_ax_fig(
                ax, size=(8, 6), dim=3, style_kwargs=style_kwargs
            )
            ax.plot_surface(
                X_logit,
                Y_logit,
                Z_logit,
                **self.init_style_Z,
            )
            all_scatter = []
            for i, x, y, s in [
                (0, x0, y0, self.init_style_0),
                (1, x1, y1, self.init_style_1),
            ]:
                all_scatter += [
                    ax.scatter(
                        x,
                        y,
                        logit(
                            self.data["coef"][0]
                            + self.data["coef"][1] * x
                            + self.data["coef"][2] * y
                        ),
                        **self._update_dict(s, style_kwargs, i),
                    )
                ]
            ax.set_xlabel(self.layout["columns"][0])
            ax.set_ylabel(self.layout["columns"][1])
            ax.set_zlabel(self.layout["columns"][2])
            ax.legend(
                all_scatter,
                [0, 1],
                scatterpoints=1,
                loc="center left",
                bbox_to_anchor=[1.15, 0.5],
                title=self.layout["columns"][2],
                ncol=2,
                fontsize=8,
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        else:
            raise ValueError("The number of predictors is too big.")
        return ax
