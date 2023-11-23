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

from verticapy._typing import NoneType

from verticapy.plotting._matplotlib.base import MatplotlibBase


class RangeCurve(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["range"]:
        return "range"

    @property
    def _compute_method(self) -> Literal["range"]:
        return "range"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {"alpha1": 0.5, "alpha2": 0.9}
        self.init_style_scatter = {
            "c": "white",
            "marker": "o",
            "s": 60,
            "edgecolors": "black",
            "zorder": 3,
        }

    def _get_final_color(self, style_kwargs: dict, idx: int) -> dict:
        kwargs = {"color": self.get_colors(d=style_kwargs, idx=idx)}
        return self._update_dict(kwargs, style_kwargs, idx)

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        plot_scatter: bool = True,
        plot_median: bool = True,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a range curve using the Matplotlib API.
        """
        style_kwargs = self._fix_color_style_kwargs(style_kwargs)
        ax, fig, style_kwargs = self._get_ax_fig(
            ax, size=(8, 6), set_axis_below=True, grid=True, style_kwargs=style_kwargs
        )
        n, m = self.data["Y"].shape
        for i in range(0, m, 3):
            idx = int(i / 3)
            ax.fill_between(
                self.data["x"],
                self.data["Y"][:, i],
                self.data["Y"][:, i + 2],
                alpha=self.init_style["alpha1"],
                label=self.layout["columns"][idx],
                facecolor=self.get_colors(d=style_kwargs, idx=idx),
            )
            for j in [0, 2]:
                ax.plot(
                    self.data["x"],
                    self.data["Y"][:, i + j],
                    alpha=self.init_style["alpha2"],
                    **self._get_final_color(style_kwargs=style_kwargs, idx=idx),
                )
            if plot_median:
                ax.plot(
                    self.data["x"],
                    self.data["Y"][:, i + 1],
                    **self._get_final_color(style_kwargs=style_kwargs, idx=idx),
                )
            if ((plot_scatter) or n < 20) and plot_median:
                ax.scatter(
                    self.data["x"],
                    self.data["Y"][:, i + 1],
                    **self.init_style_scatter,
                )
        ax.set_xlabel(self.layout["order_by"])
        if len(self.layout["columns"]) == 1:
            ax.set_ylabel(self.layout["columns"][0])
        else:
            if ("y_label" in self.layout) and not (
                isinstance(self.layout["y_label"], NoneType)
            ):
                ax.set_ylabel(self.layout["y_label"])
            ax.legend(loc="center left", bbox_to_anchor=[1, 0.5])
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        ax.set_xlim(self.data["x"][0], self.data["x"][-1])
        return ax
