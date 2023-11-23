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
import matplotlib.pyplot as plt

from verticapy.plotting._matplotlib.base import MatplotlibBase


class SpiderChart(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["spider"]:
        return "spider"

    @property
    def _compute_method(self) -> Literal["2D"]:
        return "2D"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {"linewidth": 1, "linestyle": "solid"}
        self.init_style_opacity = {"alpha": 0.1}

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a spider plot using the Matplotlib API.
        """
        style_kwargs = self._fix_color_style_kwargs(style_kwargs)
        m = self.data["X"].shape[0]
        angles = [i / float(m) * 2 * np.pi for i in range(m)]
        angles += angles[:1]
        _, fig, style_kwargs = self._get_ax_fig(
            ax,
            style_kwargs=style_kwargs,
        )
        if not ax:
            ax = fig.add_subplot(111, polar=True)
        spider_vals = np.array([])
        colors = self.get_colors()
        for i, category in enumerate(self.layout["y_labels"]):
            if len(self.data["X"].shape) == 1:
                values = np.concatenate((self.data["X"], self.data["X"][:1]))
            else:
                values = np.concatenate(
                    (self.data["X"][:, i], self.data["X"][:, i][:1])
                )
            spider_vals = np.concatenate((spider_vals, values))
            plt.xticks(angles[:-1], self.layout["x_labels"], color="grey", size=8)
            ax.set_rlabel_position(0)
            kwargs = {"color": colors[i], **self.init_style}
            kwargs = self._update_dict(kwargs, style_kwargs, i)
            args = [angles, values]
            ax.plot(*args, label=category, **kwargs)
            ax.fill(*args, color=kwargs["color"], **self.init_style_opacity)
        y_ticks = [
            min(spider_vals),
            (max(spider_vals) + min(spider_vals)) / 2,
            max(spider_vals),
        ]
        ax.set_yticks(y_ticks)
        ax.set_rgrids(y_ticks, angle=180.0, fmt="%0.1f")
        ax.set_xlabel(self.layout["columns"][0])
        ax.set_ylabel(self.layout["method"])
        if len(self.layout["columns"]) > 1:
            ax.legend(
                title=self.layout["columns"][1],
                loc="center left",
                bbox_to_anchor=[1.1, 0.5],
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax
