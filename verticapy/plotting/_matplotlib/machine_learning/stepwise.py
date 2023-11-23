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


class StepwisePlot(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["stepwise"]:
        return "stepwise"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {"marker": "s", "alpha": 0.5, "edgecolors": "black", "s": 400}

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a stepwise plot using the Matplotlib API.
        """
        colors = self.get_colors()
        ax, fig, style_kwargs = self._get_ax_fig(
            ax, size=(8, 6), set_axis_below=True, grid="y", style_kwargs=style_kwargs
        )
        sign = "+" if self.layout["direction"] == "forward" else "-"
        x_new, y_new, c_new = [], [], []
        for idx in range(len(self.data["x"])):
            if idx == 0 or self.data["sign"][idx][0] == sign:
                x_new += [self.data["x"][idx]]
                y_new += [self.data["y"][idx]]
                c_new += [self.data["c"][idx]]
        if len(self.layout["in_variables"]) > 3:
            var0 = (
                self.layout["in_variables"][0:2]
                + ["..."]
                + self.layout["in_variables"][-1:]
            )
        else:
            var0 = self.layout["in_variables"]
        if len(self.layout["out_variables"]) > 3:
            var1 = (
                self.layout["out_variables"][0:2]
                + ["..."]
                + self.layout["out_variables"][-1:]
            )
        else:
            var1 = self.layout["out_variables"]
        if "color" in style_kwargs:
            if isinstance(style_kwargs["color"], str):
                c0, c1 = style_kwargs["color"], colors[1]
            else:
                c0, c1 = style_kwargs["color"][0], style_kwargs["color"][1]
        else:
            c0, c1 = colors[0], colors[1]
        if "color" in style_kwargs:
            del style_kwargs["color"]
        if self.layout["direction"] == "forward":
            delta_ini, delta_final = 0.1, -0.15
            rot_ini, rot_final = -90, 90
            verticalalignment_init, verticalalignment_final = "top", "bottom"
            horizontalalignment = "center"
        else:
            delta_ini, delta_final = 0.35, -0.3
            rot_ini, rot_final = 90, -90
            verticalalignment_init, verticalalignment_final = "top", "bottom"
            horizontalalignment = "left"
        ax.scatter(
            x_new[1:-1],
            y_new[1:-1],
            c=c0,
            **self._update_dict(self.init_style, style_kwargs),
        )
        ax.scatter(
            [x_new[0], x_new[-1]],
            [y_new[0], y_new[-1]],
            c=c1,
            **self._update_dict(self.init_style, style_kwargs),
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
                sign + " " + c_new[idx],
                rotation=rot_ini,
            )
        if self.layout["direction"] == "backward":
            max_x, min_x = min(self.data["x"]), max(self.data["x"])
            ax.set_xlim(
                max_x + 0.1 * (1 + max_x - min_x),
                min_x - 0.1 - 0.1 * (1 + max_x - min_x),
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
        ax.set_xlabel(self.layout["x_label"])
        ax.set_ylabel(self.layout["y_label"])
        return ax
