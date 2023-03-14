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
from typing import Literal, Optional
import numpy as np

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from verticapy._typing import ArrayLike

from verticapy.plotting._matplotlib.base import MatplotlibBase


class ACFPlot(MatplotlibBase):

    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["acf"]:
        return "acf"

    # Draw.

    def draw(
        self,
        bar_type: Literal["line", "bar"] = "bar",
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws an ACF Time Series plot using the Matplotlib API.
        """
        tmp_style = {}
        for elem in style_kwargs:
            if elem not in ("color", "colors"):
                tmp_style[elem] = style_kwargs[elem]
        if "color" in style_kwargs:
            color = style_kwargs["color"]
        else:
            color = self.get_colors(idx=0)
        ax, fig = self._get_ax_fig(ax, size=(10, 3), set_axis_below=False, grid=False)
        if bar_type == "bar":
            ax.bar(
                self.data["x"],
                self.data["y"],
                width=0.007 * len(self.data["x"]),
                color="#444444",
                zorder=1,
                linewidth=0,
            )
            param = {
                "s": 90,
                "marker": "o",
                "facecolors": color,
                "edgecolors": "black",
                "zorder": 2,
            }
            ax.scatter(
                self.data["x"], self.data["y"], **self._update_dict(param, tmp_style),
            )
            ax.plot(
                np.concatenate(([-1], self.data["x"], [self.data["x"][-1] + 1])),
                [0 for elem in range(len(self.data["x"]) + 2)],
                color=color,
                zorder=0,
            )
            ax.set_xlim(-1, self.data["x"][-1] + 1)
        else:
            ax.plot(
                self.data["x"], self.data["y"], color=color, **tmp_style,
            )
        ax.set_xticks(self.data["x"])
        ax.set_xticklabels(self.data["x"], rotation=90)
        if isinstance(self.data["confidence"], np.ndarray):
            ax.fill_between(
                self.data["x"],
                -self.data["confidence"],
                self.data["confidence"],
                color=color,
                alpha=0.1,
            )
        ax.set_xlabel("lag")
        return ax
