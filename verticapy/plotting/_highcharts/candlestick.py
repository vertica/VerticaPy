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

from verticapy._typing import HChart
from verticapy.plotting._highcharts.line import LinePlot


class CandleStick(LinePlot):
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
            "rangeSelector": {"selected": 1},
            "title": {"text": ""},
            "yAxis": [
                {
                    "labels": {"align": "right", "x": -3},
                    "title": {"text": ""},
                    "height": "60%",
                    "lineWidth": 2,
                },
                {
                    "labels": {"align": "right", "x": -3},
                    "title": {"text": ""},
                    "top": "65%",
                    "height": "35%",
                    "offset": 0,
                    "lineWidth": 2,
                },
            ],
            "colors": self.get_colors(),
        }
        self.init_style_column = {"yAxis": 1}

    # Draw.

    def draw(
        self,
        chart: Optional[HChart] = None,
        **style_kwargs,
    ) -> HChart:
        """
        Draws a candlestick plot using the HC API.
        """
        chart, style_kwargs = self._get_chart(
            chart, stock=True, style_kwargs=style_kwargs
        )
        chart.set_dict_options(self.init_style)
        x = self._to_datetime(self.data["x"])
        Y = np.column_stack(
            (
                self.data["Y"][:, 2],
                self.data["Y"][:, 3],
                self.data["Y"][:, 0],
                self.data["Y"][:, 1],
            )
        )
        data = np.column_stack((x, Y)).tolist()
        chart.add_data_set(data, "candlestick", name=self.layout["column"])
        data = np.column_stack((x, self.data["z"])).tolist()
        chart.add_data_set(
            data, "column", name=self.layout["method_of"], **self.init_style_column
        )
        return chart
