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
from typing import Any, Callable, Literal, Optional

import numpy as np

from matplotlib.axes import Axes
import matplotlib.animation as animation

from verticapy.plotting._matplotlib.animated.base import AnimatedBase


class AnimatedLinePlot(AnimatedBase):
    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["animated_line"]:
        return "animated_line"

    @property
    def _compute_method(self) -> Literal["line"]:
        return "line"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "linewidth": 1,
            "linewidth": 2,
        }

    def _get_style(self, idx: int = 0) -> dict[str, Any]:
        colors = self.get_colors()
        return {**self.init_style, "color": colors[idx % len(colors)]}

    # Draw.

    def _animate(
        self, all_plots: list, window_size: int, fixed_xy_lim: bool, ax: Axes
    ) -> Callable:
        def animate(i: int) -> tuple[Axes]:
            k = max(i - window_size, 0)
            min_y, max_y = np.inf, -np.inf
            n = len(self.layout["columns"])
            for m in range(0, n):
                all_plots[m].set_xdata(self.data["x"][0:i])
                all_plots[m].set_ydata(self.data["Y"][:, m][0:i])
                if len(self.data["Y"][:, m][0:i]) > 0:
                    min_y = min(np.nanmin(self.data["Y"][:, m][0:i]), min_y)
                    max_y = max(np.nanmax(self.data["Y"][:, m][0:i]), max_y)
            if not fixed_xy_lim:
                if i > 0:
                    ax.set_ylim(min_y, max_y)
                if i > window_size:
                    ax.set_xlim(self.data["x"][k], self.data["x"][i])
                else:
                    ax.set_xlim(self.data["x"][0], self.data["x"][window_size])
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
            return (ax,)

        return animate

    def draw(
        self,
        fixed_xy_lim: bool = False,
        window_size: int = 100,
        step: int = 10,
        interval: int = 5,
        repeat: bool = True,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> animation.Animation:
        """
        Draws an animated time series plot using the Matplotlib API.
        """
        ax, fig, style_kwargs = self._get_ax_fig(
            ax, size=(8, 6), set_axis_below=True, grid="y", style_kwargs=style_kwargs
        )
        all_plots = []
        n = len(self.layout["columns"])
        for i in range(0, n):
            all_plots += [
                ax.plot(
                    [],
                    [],
                    label=self.layout["columns"][i],
                    **self._update_dict(self._get_style(idx=i), style_kwargs, i),
                )[0]
            ]
        if len(self.layout["columns"]) > 1:
            ax.legend(loc="center left", bbox_to_anchor=[1, 0.5])
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.set_xlabel(self.layout["order_by"])
        if fixed_xy_lim:
            ax.set_xlim(self.data["x"][0], self.data["x"][-1])
            ax.set_ylim(np.nanmin(self.data["Y"]), np.nanmax(self.data["Y"]))
        anim = animation.FuncAnimation(
            fig,
            self._animate(
                all_plots=all_plots,
                window_size=window_size,
                fixed_xy_lim=fixed_xy_lim,
                ax=ax,
            ),
            frames=range(0, len(self.data["x"]) - 1, step),
            interval=interval,
            blit=False,
            repeat=repeat,
        )
        return self._return_animation(anim)
