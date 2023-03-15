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
import copy, warnings
from typing import Literal, Optional, TYPE_CHECKING

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from verticapy._typing import SQLColumns

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame

from verticapy.plotting._matplotlib.base import MatplotlibBase


class Histogram(MatplotlibBase):

    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["hist"]:
        return "hist"

    @property
    def _compute_method(self) -> Literal["hist"]:
        return "hist"

    # Draw.

    def draw(self, ax: Optional[Axes] = None, **style_kwargs,) -> Axes:
        """
        Draws an histogram using the Matplotlib API.
        """
        ax, fig = self._get_ax_fig(ax, size=(8, 6), set_axis_below=True, grid="y")
        alpha, colors = 1.0, self.get_colors()
        key = "categories" if self.layout["has_category"] else "columns"
        delta = alpha / len(self.layout[key]) * 0.8
        for i, column in enumerate(self.layout[key]):
            kwargs = {"color": colors[i % len(colors)]}
            kwargs = self._update_dict(kwargs, style_kwargs, i)
            plt.bar(
                self.data[column]["x"],
                self.data[column]["y"],
                self.data["width"],
                label=column,
                alpha=alpha,
                **kwargs,
            )
            alpha -= delta
        ax.set_ylabel(self.layout["method_of"])
        if len(self.layout["columns"]) == 1:
            ax.set_xlabel(self.layout["columns"][0])
        ax.legend(title=self.layout["by"], loc="center left", bbox_to_anchor=[1, 0.5])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax
