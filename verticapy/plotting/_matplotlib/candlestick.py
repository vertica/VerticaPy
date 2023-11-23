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

import verticapy as vp
from verticapy.plotting._matplotlib.base import MatplotlibBase


class CandleStick(MatplotlibBase):
    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["candlestick"]:
        return "candlestick"

    @property
    def _compute_method(self) -> Literal["candle"]:
        return "candle"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "width": 700,
            "height": 500,
        }

    # Draw.

    def draw(
        self,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a candlestick plot using the Matplotlib API.
        """
        style_kwargs = self._fix_color_style_kwargs(style_kwargs)
        color_list = ["green", "red"]
        if "color" in style_kwargs:
            color_list = (
                style_kwargs["color"] + color_list
                if isinstance(style_kwargs["color"], list)
                else [style_kwargs["color"]] + color_list
            )
            style_kwargs.pop("color")
        ax, _, style_kwargs = self._get_ax_fig(
            ax,
            size=(min(int(len(self.data["x"]) / 1.8) + 1, 600), 6),
            set_axis_below=True,
            grid="y",
            style_kwargs=style_kwargs,
        )
        dataframe = vp.vDataFrame(
            {
                "index": self.data["x"],
                "open": self.data["Y"][:, 2],
                "close": self.data["Y"][:, 1],
                "high": self.data["Y"][:, 3],
                "low": self.data["Y"][:, 0],
            },
        )
        up = dataframe[dataframe.close >= dataframe.open]
        down = dataframe[dataframe.close < dataframe.open]
        col_up = color_list[0]
        col_down = color_list[1]
        if not isinstance(self.data["x"][0], str):
            differences = np.diff(self.data["x"])
            min_difference = np.min(differences)
            width = min_difference / 3
            width2 = min_difference / 20
        else:
            width = 0.5
            width2 = 0.08
        # open 1, close 2, high 3, low 4
        if up.shape()[0] > 0:
            up = up.to_numpy()
            ax.bar(
                up[:, 0],
                up[:, 2].astype("float") - up[:, 1].astype("float"),
                width,
                bottom=up[:, 1].astype("float"),
                color=col_up,
            )
            ax.bar(
                up[:, 0],
                up[:, 3].astype("float") - up[:, 2].astype("float"),
                width2,
                bottom=up[:, 2].astype("float"),
                color=col_up,
            )
            ax.bar(
                up[:, 0],
                up[:, 4].astype("float") - up[:, 1].astype("float"),
                width2,
                bottom=up[:, 1].astype("float"),
                color=col_up,
            )
        if down.shape()[0] > 0:
            down = down.to_numpy()
            ax.bar(
                down[:, 0],
                down[:, 2].astype("float") - down[:, 1].astype("float"),
                width,
                bottom=down[:, 1].astype("float"),
                color=col_down,
            )
            ax.bar(
                down[:, 0],
                down[:, 3].astype("float") - down[:, 1].astype("float"),
                width2,
                bottom=down[:, 1].astype("float"),
                color=col_down,
            )
            ax.bar(
                down[:, 0],
                down[:, 4].astype("float") - down[:, 2].astype("float"),
                width2,
                bottom=down[:, 2].astype("float"),
                color=col_down,
            )
        return ax
