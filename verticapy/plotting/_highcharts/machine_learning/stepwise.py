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

from verticapy._typing import HChart, NoneType
from verticapy.plotting._highcharts.base import HighchartsBase


class StepwisePlot(HighchartsBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["stepwise"]:
        return "stepwise"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "title": {"text": ""},
            "xAxis": {
                "reversed": False,
                "title": {"enabled": True, "text": self.layout["x_label"]},
                "startOnTick": True,
                "endOnTick": True,
                "showLastLabel": True,
            },
            "yAxis": {
                "reversed": True,
                "title": {"text": self.layout["y_label"]},
            },
            "legend": {"enabled": True},
            "plotOptions": {
                "scatter": {
                    "marker": {
                        "radius": 5,
                        "states": {
                            "hover": {"enabled": True, "lineColor": "rgb(100,100,100)"}
                        },
                    },
                    "states": {"hover": {"marker": {"enabled": False}}},
                }
            },
            "tooltip": {
                "headerFormat": '<span style="color:{series.color}">\u25CF</span> {series.name} <br/>',
                "pointFormat": "<b>"
                + str(self.layout["x_label"])
                + "</b>: {point.x}<br><b>"
                + str(self.layout["y_label"])
                + "</b>: {point.y}<br>"
                + str(self.layout["z_label"])
                + "</b>: {point.z}<br>",
            },
            "colors": self.get_colors(),
        }
        self.init_style_plus = {
            "color": self.get_colors(idx=0),
            "zIndex": 2,
        }
        self.init_style_minus = {
            "color": self.get_colors(idx=1),
            "zIndex": 2,
        }
        self.init_style_line = {
            "color": "#CFCFCF",
            "zIndex": 1,
            "dashStyle": "longdash",
            "enableMouseTracking": False,
        }
        self.init_style_start_end = {
            "color": self.get_colors(idx=2),
            "zIndex": 0,
            "tooltip": {
                "headerFormat": '<span style="color:{series.color}">\u25CF</span> {series.name} <br/>',
                "pointFormat": "<b>"
                + str(self.layout["x_label"])
                + "</b>: {point.x}<br><b>"
                + str(self.layout["y_label"])
                + "</b>: {point.y}<br>",
            },
        }

    # Draw.

    def draw(self, chart: Optional[HChart] = None, **style_kwargs) -> HChart:
        """
        Draws a stepwise plot using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        lookup_table = {
            "+": self.init_style_plus,
            "-": self.init_style_minus,
        }
        for i, c in enumerate(self.data["c"]):
            data = [
                [
                    int(self.data["x"][i]),
                    float(self.data["y"][i]),
                    float(self.data["s"][i]),
                ]
            ]
            if not isinstance(c, NoneType):
                sign = self.data["sign"][i]
                chart.add_data_set(data, "bubble", sign + c, **lookup_table[sign])
        i = -1
        if self.layout["direction"] == "forward":
            condition = self.data["sign"] != "-"
            while self.data["sign"][i] == "-":
                i -= 1
        else:
            condition = self.data["sign"] != "+"
            while self.data["sign"][i] == "+":
                i -= 1
        data = np.column_stack(
            (self.data["x"][condition], self.data["y"][condition])
        ).tolist()
        chart.add_data_set(data, "spline", "stepwise", **self.init_style_line)
        start = [[int(self.data["x"][0]), float(self.data["y"][0]), 100]]
        chart.add_data_set(start, "bubble", "start", **self.init_style_start_end)
        end = [[int(self.data["x"][i]), float(self.data["y"][i]), 100]]
        chart.add_data_set(end, "bubble", "end", **self.init_style_start_end)
        return chart
