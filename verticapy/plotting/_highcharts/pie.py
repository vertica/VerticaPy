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
            "tooltip": {},
            "colors": self.get_colors(),
        }
        if self.layout["kind"] == "donut":
            self.init_style = {
                **self.init_style,
                "chart": {"type": "pie"},
                "plotOptions": {"pie": {"innerSize": 100, "depth": 45}},
            }
        elif self.layout["kind"] == "rose":
            self.init_style = {
                **self.init_style,
                "plotOptions": {"pie": {"startAngle": -90, "endAngle": 90}},
                "legend": {"enabled": False},
            }
        self.init_style_3d = {
            "chart": {"type": "pie", "options3d": {"enabled": True, "alpha": 45}}
        }

    # Draw.

    def draw(
        self,
        chart: Optional[HChart] = None,
        **style_kwargs,
    ) -> HChart:
        """
        Draws a pie chart using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        data = []
        for idx, y in enumerate(self.data["y"]):
            data += [
                {
                    "name": self.layout["column"]
                    + "="
                    + str(self.layout["labels"][idx]),
                    "y": float(y),
                }
            ]
        data[-1] = {**data[-1], "sliced": True, "selected": True}
        chart.add_data_set(data, "pie", self.layout["method_of"])
        if self.layout["kind"] == "3d":
            chart.set_dict_options(self.init_style_3d)
            chart.add_JSsource("https://code.highcharts.com/6/highcharts-3d.js")
        return chart


class NestedPieChart(HighchartsBase):
    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["pie"]:
        return "pie"

    @property
    def _compute_method(self) -> Literal["rollup"]:
        return "rollup"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (1, 2)

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "chart": {"type": "pie"},
            "title": {"text": ""},
            "yAxis": {"title": {"text": ""}},
            "plotOptions": {"pie": {"shadow": False, "center": ["50%", "50%"]}},
            "tooltip": {},
        }
        if len(self.layout["columns"]) == 1:
            self.init_style["plotOptions"]["pie"]["allowPointSelect"] = True
            self.init_style["plotOptions"]["pie"]["cursor"] = "pointer"
            self.init_style["plotOptions"]["pie"]["showInLegend"] = True

    # Draw.

    def draw(
        self,
        chart: Optional[HChart] = None,
        **style_kwargs,
    ) -> HChart:
        """
        Draws a nested pie chart using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        n = len(self.layout["columns"])
        data = []
        uniques = self.data["groups"][-1][0]
        y = self.data["groups"][-1][1].astype(float)
        if n == 1:
            for i, c in enumerate(uniques):
                data += [
                    {
                        "name": self.layout["columns"][0] + "=" + str(c),
                        "y": float(y[i]),
                        "color": self.get_colors(idx=i),
                    }
                ]
            data[-1] = {**data[-1], "sliced": True, "selected": True}
            chart.add_data_set(data, "pie", self.layout["method_of"])
        elif n == 2:
            group = np.column_stack(self.data["groups"][0])
            for i, c in enumerate(uniques):
                categories = group[group[:, 0] == c][:, -2].tolist()
                Y = group[group[:, 0] == c][:, -1].astype(float).tolist()
                data += [
                    {
                        "y": float(y[i]),
                        "color": self.get_colors(idx=i),
                        "drilldown": {
                            "name": c,
                            "categories": categories,
                            "data": Y,
                            "color": self.get_colors(idx=i),
                        },
                    }
                ]
            innerData = []
            outerData = []
            for i in range(len(data)):
                innerData += [
                    {
                        "name": self.layout["columns"][0] + "=" + str(uniques[i]),
                        "y": data[i]["y"],
                        "color": data[i]["color"],
                    }
                ]

                drillDataLen = len(data[i]["drilldown"]["data"])
                for j in range(drillDataLen):
                    brightness = 0.3 * (1 - (j / drillDataLen))
                    color = data[i]["color"]
                    if "#" in color and len(color) == 7:
                        c = "#"
                        for k in [1, 3, 5]:
                            val = int(
                                int("0x" + color[k : k + 2], 16) * (1 + brightness)
                            )
                            val = min(val, 255)
                            val = hex(val)[2:]
                            if len(val) == 1:
                                val = "0" + val
                            c += val
                        color = c
                    outerData += [
                        {
                            "name": self.layout["columns"][1]
                            + "="
                            + str(data[i]["drilldown"]["categories"][j]),
                            "y": data[i]["drilldown"]["data"][j],
                            "color": color,
                        }
                    ]
            chart.add_data_set(
                innerData,
                "pie",
                self.layout["method_of"],
                size="60%",
                dataLabels={
                    "color": "white",
                    "distance": -30,
                },
            )
            chart.add_data_set(
                outerData,
                "pie",
                self.layout["method_of"],
                size="80%",
                innerSize="60%",
            )
        else:
            raise ValueError("The number of columns is too big to draw the plot.")
        return chart
