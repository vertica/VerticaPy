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

from matplotlib.axes import Axes

from verticapy.plotting._matplotlib.base import MatplotlibBase


class BoxPlot(MatplotlibBase):

    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["box"]:
        return "box"

    @property
    def _compute_method(self) -> Literal["describe"]:
        return "describe"

    # Draw.

    def draw(self, ax: Optional[Axes] = None, **style_kwargs,) -> Axes:
        """
        Draws a multi box plot using the Matplotlib API.
        """
        n, m = self.data["X"].shape
        if m == 1 and "vert" not in style_kwargs:
            style_kwargs["vert"] = False
        elif "vert" not in style_kwargs:
            style_kwargs["vert"] = True
        ax, fig = self._get_ax_fig(
            ax,
            size=(10, 6),
            set_axis_below=True,
            grid="y" if style_kwargs["vert"] else "x",
        )
        box = ax.boxplot(
            self.data["X"],
            notch=False,
            sym="",
            whis=float("Inf"),
            widths=0.5,
            labels=self.layout["labels"],
            patch_artist=True,
            **style_kwargs,
        )
        if not (style_kwargs["vert"]):
            ax.set_yticklabels(self.layout["labels"], rotation=90)
        else:
            ax.set_xticklabels(self.layout["labels"], rotation=90)
        for median in box["medians"]:
            median.set(
                color="black", linewidth=1,
            )
        for i, patch in enumerate(box["boxes"]):
            patch.set_facecolor(self.get_colors(idx=i))
        if self.layout["has_category"]:
            if not (style_kwargs["vert"]):
                x_label, y_label = self.layout["y_label"], self.layout["x_label"]
            else:
                x_label, y_label = self.layout["x_label"], self.layout["y_label"]
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
        return ax
