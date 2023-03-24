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
from typing import Literal

from vertica_highcharts import Highchart

from verticapy.plotting._highcharts.base import HighchartsBase


class PieChart(HighchartsBase):

    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["pie"]:
        return "pie"

    @property
    def _compute_method(self) -> Literal["1D"]:
        return "1D"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "title": {"text": ""},
            "chart": {"inverted": True},
            "xAxis": {
                "reversed": False,
                "title": {"text": self.layout["column"], "enabled": True},
                "maxPadding": 0.05,
                "showLastLabel": True,
            },
            "yAxis": {"title": {"text": self.layout["method_of"], "enabled": True}},
            "plotOptions": {
                "pie": {
                    "allowPointSelect": True,
                    "cursor": "pointer",
                    "showInLegend": True,
                    "size": "110%",
                }
            },
            "tooltip": {
                "pointFormat": str(self.layout["method_of"]) + ": <b>{point.y}</b>"
            },
            "colors": self.get_colors(),
        }
        if self.layout["pie_type"] == "donut":
            self.init_style = {
                **self.init_style,
                "chart": {"type": "pie"},
                "plotOptions": {"pie": {"innerSize": 100, "depth": 45}},
            }
        elif self.layout["pie_type"] == "rose":
            self.init_style = {
                **self.init_style,
                "plotOptions": {"pie": {"startAngle": -90, "endAngle": 90}},
                "legend": {"enabled": False},
            }
        self.init_style_3d = {
            "chart": {"type": "pie", "options3d": {"enabled": True, "alpha": 45}}
        }
        return None

    # Draw.

    def draw(self, **style_kwargs,) -> Highchart:
        """
        Draws a pie chart using the HC API.
        """
        chart = Highchart(width=600, height=400)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        data = []
        for idx, y in enumerate(self.data["y"]):
            data += [{"name": self.layout["labels"][idx], "y": y}]
        data[-1] = {**data[-1], "sliced": True, "selected": True}
        chart.add_data_set(data, "pie")
        if self.layout["pie_type"] == "3d":
            chart.set_dict_options(self.init_style_3d)
            chart.add_JSsource("https://code.highcharts.com/6/highcharts-3d.js")
        return chart
