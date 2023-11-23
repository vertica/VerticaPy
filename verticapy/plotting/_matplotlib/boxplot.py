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

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {"widths": 0.3}

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a multi-box plot using the Matplotlib API.
        """
        style_kwargs = self._fix_color_style_kwargs(style_kwargs)
        m = self.data["X"].shape[1]
        if m == 1 and "vert" not in style_kwargs:
            style_kwargs["vert"] = False
        elif "vert" not in style_kwargs:
            style_kwargs["vert"] = True
        ax, fig, style_kwargs = self._get_ax_fig(
            ax,
            size=(10, 6),
            set_axis_below=True,
            grid="y" if style_kwargs["vert"] else "x",
            style_kwargs=style_kwargs,
        )
        if style_kwargs["vert"]:
            set_lim = ax.set_ylim
            set_tick = ax.set_xticklabels
        else:
            set_lim = ax.set_xlim
            set_tick = ax.set_yticklabels
        box = ax.boxplot(
            self.data["X"],
            notch=False,
            labels=self.layout["labels"],
            patch_artist=True,
            **self.init_style,
            **{key: value for key, value in style_kwargs.items() if key != "color"},
        )
        set_tick(self.layout["labels"], rotation=90)
        for median in box["medians"]:
            median.set(
                color="black",
                linewidth=1,
            )
        for i, patch in enumerate(box["boxes"]):
            patch.set_facecolor(self.get_colors(d=style_kwargs, idx=i))
        for i, flier in enumerate(box["fliers"]):
            xdata = [i + 1] * len(self.data["fliers"][i])
            ydata = self.data["fliers"][i]
            if style_kwargs["vert"]:
                kwargs = {"xdata": xdata, "ydata": ydata}
            else:
                kwargs = {"xdata": ydata, "ydata": xdata}
            flier.set(**kwargs)
        if self.layout["has_category"]:
            if not style_kwargs["vert"]:
                x_label, y_label = self.layout["y_label"], self.layout["x_label"]
            else:
                x_label, y_label = self.layout["x_label"], self.layout["y_label"]
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
        min_lim = min(
            min(min(f) if len(f) > 0 else np.inf for f in self.data["fliers"]),
            self.data["X"].min(),
        )
        max_lim = max(
            max(max(f) if len(f) > 0 else -np.inf for f in self.data["fliers"]),
            self.data["X"].max(),
        )
        h = (max_lim - min_lim) * 0.01
        set_lim(min_lim - h, max_lim + h)
        return ax
