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
import random
from typing import Literal, Optional
import numpy as np

from matplotlib.axes import Axes

from verticapy.plotting._matplotlib.base import MatplotlibBase


class SVMClassifierPlot(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["svm"]:
        return "svm"

    @property
    def _compute_method(self) -> Literal["sample"]:
        return "sample"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (2, 4)

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style_0 = {
            "marker": "o",
            "color": self.get_colors(idx=0),
            "s": 50,
            "edgecolors": "black",
        }
        self.init_style_1 = {
            "marker": "o",
            "color": self.get_colors(idx=1),
            "s": 50,
            "edgecolors": "black",
        }
        self.init_style_svm = {
            "alpha": 1.0,
            "color": "black",
        }
        self.init_style_svm_3d = {
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
        Draws an SVM classifier plot using the Matplotlib API.
        """
        style_kwargs = self._fix_color_style_kwargs(style_kwargs)
        x, w = self.data["X"][:, 0], self.data["X"][:, -1]
        x0, x1 = x[w == 0], x[w == 1]
        if len(self.layout["columns"]) == 2:
            ax, fig, style_kwargs = self._get_ax_fig(
                ax,
                size=(8, 6),
                set_axis_below=True,
                grid=True,
                style_kwargs=style_kwargs,
            )
            x_svm = [-self.data["coef"][0] / self.data["coef"][1] for i in range(2)]
            y_svm = [-1, 1]
            ax.plot(x_svm, y_svm, **self.init_style_svm)
            all_scatter = []
            for i, x, s in [(0, x0, self.init_style_0), (1, x1, self.init_style_1)]:
                all_scatter += [
                    ax.scatter(
                        x,
                        [2 * (random.random() - 0.5) for xi in x],
                        **self._update_dict(s, style_kwargs, i),
                    )
                ]
            ax.set_yticks([])
        else:
            y = self.data["X"][:, 1]
            y0, y1 = y[w == 0], y[w == 1]
            min_svm_x, max_svm_x = np.nanmin(x), np.nanmax(x)
            if len(self.layout["columns"]) == 3:
                ax, fig, style_kwargs = self._get_ax_fig(
                    ax,
                    size=(8, 6),
                    set_axis_below=True,
                    grid=True,
                    style_kwargs=style_kwargs,
                )
                x_svm = [min_svm_x, max_svm_x]
                y_svm = [
                    -(self.data["coef"][0] + self.data["coef"][1] * x)
                    / self.data["coef"][2]
                    for x in x_svm
                ]
                ax.plot(x_svm, y_svm, **self.init_style_svm)
                all_scatter = []
                for i, x, y, s in [
                    (0, x0, y0, self.init_style_0),
                    (1, x1, y1, self.init_style_1),
                ]:
                    all_scatter += [
                        ax.scatter(x, y, **self._update_dict(s, style_kwargs, i))
                    ]
            elif len(self.layout["columns"]) == 4:
                z = self.data["X"][:, 2]
                z0, z1 = z[w == 0], z[w == 1]
                step_x = (max_svm_x - min_svm_x) / 40.0
                min_svm_y, max_svm_y = np.nanmin(y), np.nanmax(y)
                step_y = (max_svm_y - min_svm_y) / 40.0
                X_svm = (
                    np.arange(min_svm_x - 5 * step_x, max_svm_x + 5 * step_x, step_x)
                    if (step_x > 0)
                    else [max_svm_x]
                )
                Y_svm = (
                    np.arange(min_svm_y - 5 * step_y, max_svm_y + 5 * step_y, step_y)
                    if (step_y > 0)
                    else [max_svm_y]
                )
                X_svm, Y_svm = np.meshgrid(X_svm, Y_svm)
                Z_svm = (
                    -(
                        self.data["coef"][0]
                        + self.data["coef"][1] * X_svm
                        + self.data["coef"][2] * Y_svm
                    )
                    / self.data["coef"][3]
                )
                ax, fig, style_kwargs = self._get_ax_fig(
                    ax, size=(8, 6), dim=3, style_kwargs=style_kwargs
                )
                ax.plot_surface(
                    X_svm,
                    Y_svm,
                    Z_svm,
                    **self.init_style_svm_3d,
                )
                all_scatter = []
                alpha = 1.0
                for i, x, y, z, s in [
                    (0, x0, y0, z0, self.init_style_0),
                    (1, x1, y1, z1, self.init_style_1),
                ]:
                    all_scatter += [
                        ax.scatter(
                            x,
                            y,
                            z,
                            alpha=alpha,
                            **self._update_dict(s, style_kwargs, i),
                        )
                    ]
                    alpha -= 0.2
            else:
                raise ValueError("The number of predictors is too big.")
        ax.set_xlabel(self.layout["columns"][0])
        bbox_to_anchor = [1, 0.5]
        kwargs = {}
        if len(self.layout["columns"]) > 2:
            ax.set_ylabel(self.layout["columns"][1])
        if len(self.layout["columns"]) > 3:
            ax.set_zlabel(self.layout["columns"][2])
            bbox_to_anchor = [1.15, 0.5]
            kwargs = {"ncol": 1, "fontsize": 8}
        ax.legend(
            all_scatter,
            [0, 1],
            scatterpoints=1,
            loc="center left",
            title=self.layout["columns"][-1],
            bbox_to_anchor=bbox_to_anchor,
            **kwargs,
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax
