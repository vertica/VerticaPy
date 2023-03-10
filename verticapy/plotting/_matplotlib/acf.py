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
from verticapy._typing import ArrayLike

from verticapy.plotting._matplotlib.base import MatplotlibBase


class ACFPlot(MatplotlibBase):
    def acf_plot(
        self,
        x: ArrayLike,
        y: ArrayLike,
        title: str = "",
        confidence: ArrayLike = None,
        type_bar: bool = True,
        ax: Optional[Axes] = None,
        **style_kwds,
    ) -> Axes:
        """
        Draws an ACF Time Series plot using the Matplotlib API.
        """
        tmp_style = {}
        for elem in style_kwds:
            if elem not in ("color", "colors"):
                tmp_style[elem] = style_kwds[elem]
        if "color" in style_kwds:
            color = style_kwds["color"]
        else:
            color = get_colors()[0]
        ax, fig = self._get_ax_fig(ax, size=(10, 3), set_axis_below=False, grid=False)
        if type_bar:
            ax.bar(x, y, width=0.007 * len(x), color="#444444", zorder=1, linewidth=0)
            param = {
                "s": 90,
                "marker": "o",
                "facecolors": color,
                "edgecolors": "black",
                "zorder": 2,
            }
            ax.scatter(
                x, y, **self.updated_dict(param, tmp_style),
            )
            ax.plot(
                [-1] + x + [x[-1] + 1],
                [0 for elem in range(len(x) + 2)],
                color=color,
                zorder=0,
            )
            ax.set_xlim(-1, x[-1] + 1)
        else:
            ax.plot(
                x, y, color=color, **tmp_style,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(x, rotation=90)
        if confidence:
            ax.fill_between(
                x, [-c for c in confidence], confidence, color=color, alpha=0.1
            )
        ax.set_xlabel("lag")
        return ax
