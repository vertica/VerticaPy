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
import math
from typing import Literal, Optional, TYPE_CHECKING
import numpy as np

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from verticapy._config.colors import get_colors
from verticapy._typing import SQLColumns
from verticapy.errors import ParameterError

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame

from verticapy.plotting._matplotlib.base import MatplotlibBase


class SpiderChart(MatplotlibBase):
    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["spider"]:
        return "spider"

    @property
    def _compute_method(self) -> Literal["2D"]:
        return "2D"

    def draw(self, ax: Optional[Axes] = None, **style_kwargs,) -> Axes:
        """
        Draws a spider plot using the Matplotlib API.
        """
        m = self.data["matrix"].shape[0]
        if m < 3:
            raise ParameterError(
                "The column used to draw the Spider Chart must "
                f"have at least 3 categories. Found {int(m)}."
            )
        angles = [i / float(m) * 2 * math.pi for i in range(m)]
        angles += angles[:1]
        fig = plt.figure()
        if not (ax):
            ax = fig.add_subplot(111, polar=True)
        spider_vals = np.array([])
        colors = get_colors()
        for i, category in enumerate(self.data["y_labels"]):
            if len(self.data["matrix"].shape) == 1:
                values = np.concatenate((self.data["matrix"], self.data["matrix"][:1]))
            else:
                values = np.concatenate(
                    (self.data["matrix"][:, i], self.data["matrix"][:, i][:1])
                )
            spider_vals = np.concatenate((spider_vals, values))
            plt.xticks(angles[:-1], self.data["x_labels"], color="grey", size=8)
            ax.set_rlabel_position(0)
            params = {"linewidth": 1, "linestyle": "solid", "color": colors[i]}
            params = self._update_dict(params, style_kwargs, i)
            args = [angles, values]
            ax.plot(*args, label=category, **params)
            ax.fill(*args, alpha=0.1, color=params["color"])
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
