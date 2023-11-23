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


class ElbowCurve(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["elbow"]:
        return "elbow"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "color": self.get_colors(idx=0),
            "marker": "o",
            "markerfacecolor": "white",
            "markersize": 7,
            "markeredgecolor": "black",
        }

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a machine learning bubble plot using the Matplotlib API.
        """
        ax, fig, style_kwargs = self._get_ax_fig(
            ax, size=(8, 6), set_axis_below=False, grid="y", style_kwargs=style_kwargs
        )
        ax.plot(
            self.data["x"],
            self.data["y"],
            **self._update_dict(self.init_style, style_kwargs),
        )
        ax.set_title(self.layout["title"])
        ax.set_xlabel(self.layout["x_label"])
        ax.set_ylabel(self.layout["y_label"])
        return ax
