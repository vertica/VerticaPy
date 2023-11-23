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


class ACFPlot(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["acf"]:
        return "acf"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "s": 90,
            "marker": "o",
            "facecolors": self.get_colors(idx=0),
            "edgecolors": "black",
            "zorder": 2,
        }
        self.init_style_bar = {
            "color": "#444444",
            "zorder": 1,
            "linewidth": 0,
        }
        self.init_style_alpha = {
            "alpha": 0.1,
        }

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws an ACF time series plot using the Matplotlib API.
        """
        style_kwargs = self._fix_color_style_kwargs(style_kwargs)
        ax, fig, style_kwargs = self._get_ax_fig(
            ax,
            size=(10, 3),
            set_axis_below=False,
            grid=False,
            style_kwargs=style_kwargs,
        )
        if "color" not in style_kwargs:
            style_kwargs["color"] = self.get_colors(idx=0)
        color = style_kwargs["color"]
        if self.layout["kind"] == "bar":
            ax.bar(
                self.data["x"],
                self.data["y"],
                width=0.007 * len(self.data["x"]),
                **self.init_style_bar,
            )
            ax.scatter(
                self.data["x"],
                self.data["y"],
                **self._update_dict(self.init_style, style_kwargs),
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
                self.data["x"],
                self.data["y"],
                **style_kwargs,
            )
        ax.set_xticks(self.data["x"])
        ax.set_xticklabels(self.data["x"], rotation=90)
        if isinstance(self.data["z"], np.ndarray):
            ax.fill_between(
                self.data["x"],
                -self.data["z"],
                self.data["z"],
                color=color,
                **self.init_style_alpha,
            )
        ax.set_xlabel("lag")
        return ax


class ACFPACFPlot(ACFPlot):
    # Properties.

    @property
    def _kind(self) -> Literal["acf_pacf"]:
        return "acf_pacf"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "s": 90,
            "marker": "o",
            "facecolors": self.get_colors(idx=0),
            "edgecolors": "black",
            "zorder": 2,
        }
        self.init_style_opacity = {
            "alpha": 0.1,
        }
        self.init_style_bar = {
            "color": "#444444",
            "zorder": 1,
            "linewidth": 0,
        }

    # Draw.

    def draw(
        self,
        **style_kwargs,
    ) -> plt.Figure:
        """
        Draws an ACF-PACF time series plot using the Matplotlib API.
        """
        style_kwargs = self._fix_color_style_kwargs(style_kwargs)
        fig = plt.figure(figsize=(10, 6))
        plt.rcParams["axes.facecolor"] = "#FCFCFC"
        color = self._get_final_color(style_kwargs=style_kwargs)
        width = 0.007 * len(self.data["x"])
        ax1 = fig.add_subplot(211)
        x1 = np.concatenate(([-1], self.data["x"], [self.data["x"][-1] + 1]))
        y1 = [0 for x in range(len(self.data["x"]) + 2)]
        ax1.bar(self.data["x"], self.data["y0"], width=width, **self.init_style_bar)
        ax1.scatter(
            self.data["x"],
            self.data["y0"],
            **self._update_dict(self.init_style, style_kwargs),
        )
        ax1.plot(x1, y1, color=color, zorder=0)
        ax1.fill_between(
            self.data["x"], self.data["z"], color=color, **self.init_style_opacity
        )
        ax1.fill_between(
            self.data["x"], -self.data["z"], color=color, **self.init_style_opacity
        )
        ax1.set_title(self.layout["y0_label"])
        ax1.set_xticks([])
        ax1.set_xlim(self.data["x"][0] - 0.15, self.data["x"][-1] + 0.15)
        ax1.set_ylim(-1.1, 1.1)
        ax2 = fig.add_subplot(212)
        ax2.bar(self.data["x"], self.data["y1"], width=width, **self.init_style_bar)
        ax2.scatter(
            self.data["x"],
            self.data["y1"],
            **self._update_dict(self.init_style, style_kwargs),
        )
        ax2.plot(x1, y1, color=color, zorder=0)
        ax2.fill_between(
            self.data["x"], self.data["z"], color=color, **self.init_style_opacity
        )
        ax2.fill_between(
            self.data["x"], -self.data["z"], color=color, **self.init_style_opacity
        )
        ax2.set_title(self.layout["y1_label"])
        ax2.set_xlim(self.data["x"][0] - 0.15, self.data["x"][-1] + 0.15)
        ax2.set_ylim(-1.1, 1.1)
        return fig
