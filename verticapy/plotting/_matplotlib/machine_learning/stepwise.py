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
from typing import Optional

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from verticapy._config.colors import get_colors
import verticapy._config.config as conf

from verticapy.plotting._matplotlib.base import MatplotlibBase


class StepwisePlot(MatplotlibBase):
    def plot_stepwise_ml(
        self,
        x: list,
        y: list,
        z: list = [],
        w: list = [],
        var: list = [],
        x_label: str = "n_features",
        y_label: str = "score",
        direction="forward",
        ax: Optional[Axes] = None,
        **style_kwds,
    ) -> Axes:
        """
        Draws a stepwise plot using the Matplotlib API.
        """
        colors = get_colors()
        ax, fig = self._get_ax_fig(ax, size=(8, 6), set_axis_below=True, grid="y")
        sign = "+" if direction == "forward" else "-"
        x_new, y_new, z_new = [], [], []
        for idx in range(len(x)):
            if idx == 0 or w[idx][0] == sign:
                x_new += [x[idx]]
                y_new += [y[idx]]
                z_new += [z[idx]]
        if len(var[0]) > 3:
            var0 = var[0][0:2] + ["..."] + var[0][-1:]
        else:
            var0 = var[0]
        if len(var[1]) > 3:
            var1 = var[1][0:2] + ["..."] + var[1][-1:]
        else:
            var1 = var[1]
        if "color" in style_kwds:
            if isinstance(style_kwds["color"], str):
                c0, c1 = style_kwds["color"], colors[1]
            else:
                c0, c1 = style_kwds["color"][0], style_kwds["color"][1]
        else:
            c0, c1 = colors[0], colors[1]
        if "color" in style_kwds:
            del style_kwds["color"]
        if direction == "forward":
            delta_ini, delta_final = 0.1, -0.15
            rot_ini, rot_final = -90, 90
            verticalalignment_init, verticalalignment_final = "top", "bottom"
            horizontalalignment = "center"
        else:
            delta_ini, delta_final = 0.35, -0.3
            rot_ini, rot_final = 90, -90
            verticalalignment_init, verticalalignment_final = "top", "bottom"
            horizontalalignment = "left"
        param = {"marker": "s", "alpha": 0.5, "edgecolors": "black", "s": 400}
        ax.scatter(
            x_new[1:-1], y_new[1:-1], c=c0, **self.updated_dict(param, style_kwds)
        )
        ax.scatter(
            [x_new[0], x_new[-1]],
            [y_new[0], y_new[-1]],
            c=c1,
            **self.updated_dict(param, style_kwds),
        )
        ax.text(
            x_new[0] + delta_ini,
            y_new[0],
            f"Initial Variables: [{', '.join(var0)}]",
            rotation=rot_ini,
            verticalalignment=verticalalignment_init,
        )
        for idx in range(1, len(x_new)):
            dx, dy = x_new[idx] - x_new[idx - 1], y_new[idx] - y_new[idx - 1]
            ax.arrow(x_new[idx - 1], y_new[idx - 1], dx, dy, fc="k", ec="k", alpha=0.2)
            ax.text(
                (x_new[idx] + x_new[idx - 1]) / 2,
                (y_new[idx] + y_new[idx - 1]) / 2,
                sign + " " + z_new[idx],
                rotation=rot_ini,
            )
        if direction == "backward":
            ax.set_xlim(
                max(x) + 0.1 * (1 + max(x) - min(x)),
                min(x) - 0.1 - 0.1 * (1 + max(x) - min(x)),
            )
        ax.text(
            x_new[-1] + delta_final,
            y_new[-1],
            f"Final Variables: [{', '.join(var1)}]",
            rotation=rot_final,
            verticalalignment=verticalalignment_final,
            horizontalalignment=horizontalalignment,
        )
        ax.set_xticks(x_new)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        return ax
