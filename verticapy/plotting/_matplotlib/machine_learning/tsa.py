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
from typing import Any, Literal, Optional

import numpy as np

from matplotlib.axes import Axes

from verticapy.plotting._matplotlib.base import MatplotlibBase


class TSPlot(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["tsa"]:
        return "tsa"

    @property
    def _compute_method(self) -> Literal["tsa"]:
        return "tsa"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "color": self.get_colors(),
            "linewidth": 2,
        }
        self.init_style_fill = {
            "alpha": 0.2,
        }

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a time series plot using the Matplotlib API.
        """

        # Initialization
        style_kwargs = self._fix_color_style_kwargs(style_kwargs)
        colors = self.get_colors()
        color_kwargs = {"color": self.get_colors()}
        ax, fig, style_kwargs = self._get_ax_fig(
            ax, size=(8, 6), set_axis_below=True, grid="y", style_kwargs=style_kwargs
        )
        # Standard Error
        if self.layout["has_se"]:
            args = [self.data["se_x"], self.data["se_low"], self.data["se_high"]]
            kwargs = self._update_dict(
                self.init_style, {**color_kwargs, **style_kwargs}, color_idx=3
            )
            ax.fill_between(
                *args,
                facecolor=kwargs["color"],
                **self.init_style_fill,
                label="95% confidence interval",
            )
            args = [self.data["se_x"], self.data["se_low"]]
            ax.plot(*args, color=kwargs["color"])
            args = [self.data["se_x"], self.data["se_high"]]
            ax.plot(*args, color=kwargs["color"])
        # True Values
        args = [self.data["x"], self.data["y"]]
        kwargs = self._update_dict(
            self.init_style, {**color_kwargs, **style_kwargs}, color_idx=0
        )
        ax.plot(*args, **kwargs, label=self.layout["columns"])
        # One step ahead forecast
        if not (self.layout["is_forecast"]):
            args = [self.data["x_pred_one"], self.data["y_pred_one"]]
            kwargs = self._update_dict(
                self.init_style, {**color_kwargs, **style_kwargs}, color_idx=1
            )
            ax.plot(*args, **kwargs, label="one-sted-ahead-forecast")
        # Forecast
        args = [self.data["x_pred"], self.data["y_pred"]]
        kwargs = self._update_dict(
            self.init_style, {**color_kwargs, **style_kwargs}, color_idx=2
        )
        # kwargs = {**kwargs, **{"linestyle": "dashed"}}
        ax.plot(*args, **kwargs, label="forecast")
        # Labels
        min_x = min(min(self.data["x"]), min(self.data["x_pred"]))
        max_x = max(max(self.data["x"]), max(self.data["x_pred"]))
        ax.set_xlim(min_x, max_x)
        ax.set_xlabel(self.layout["order_by"])
        ax.set_ylabel(self.layout["columns"])
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        ax.legend(
            loc="center left",
            bbox_to_anchor=[1, 0.5],
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax
