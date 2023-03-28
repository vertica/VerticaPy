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
import random
from typing import Literal, Optional
import numpy as np

from verticapy._typing import HChart
from verticapy.plotting._highcharts.base import HighchartsBase


class OutliersPlot(HighchartsBase):

    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["outliers"]:
        return "outliers"

    @property
    def _compute_method(self) -> Literal["outliers"]:
        return "outliers"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (1, 2)

    # Styling Methods.

    def _init_style(self) -> None:
        columns = self._clean_quotes(self.layout["columns"])
        tooltip = {
            "headerFormat": '<span style="color:{series.color}">\u25CF</span> {series.name} <br/>',
            "pointFormat": "<b>" + str(columns[0]) + "</b>: {point.x}<br>",
        }
        if len(columns) > 1:
            tooltip["pointFormat"] += "<b>" + str(columns[1]) + "</b>: {point.y}<br>"
        self.init_style = {
            "title": {"text": ""},
            "xAxis": {
                "reversed": False,
                "title": {"enabled": True, "text": columns[0]},
                "startOnTick": True,
                "endOnTick": True,
                "showLastLabel": True,
            },
            "yAxis": {"title": {"text": ""}},
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
            "colors": [self.layout["outliers_color"], self.layout["inliers_color"]],
            "tooltip": tooltip,
        }
        if len(self.layout["columns"]) > 1:
            self.init_style["yAxis"]["title"]["text"] = columns[1]
        self.init_style_circle = {"color": self.layout["inliers_border_color"]}
        self.init_style_outliers = {
            "marker": {
                "fillColor": self.layout["outliers_color"],
                "lineWidth": 1,
                "lineColor": "black",
            },
        }
        self.init_style_inliers = {
            "marker": {
                "fillColor": self.layout["inliers_color"],
                "lineWidth": 1,
                "lineColor": "black",
            },
        }
        return None

    # Draw.

    def draw(self, chart: Optional[HChart] = None, **style_kwargs) -> HChart:
        """
        Draws an outliers plot using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        min0, max0 = self.data["min"][0], self.data["max"][0]
        avg0, std0 = self.data["avg"][0], self.data["std"][0]
        if len(self.layout["columns"]) == 1:
            avg1, std1 = 0, 1
        else:
            avg1, std1 = self.data["avg"][1], self.data["std"][1]
        th = self.data["th"]
        a = th * std0
        b = th * std1
        data_circle = [
            [
                avg0 + a * np.cos(2 * np.pi * x / 1000),
                avg1 + b * np.sin(2 * np.pi * x / 1000),
            ]
            for x in range(-1000, 1000, 1)
        ]
        x = self.data["X"][:, 0]
        if len(self.layout["columns"]) == 1:
            zs = abs(x - avg0) / std0
            data_0 = [[xi, 2 * (random.random() - 0.5)] for xi in x[abs(zs) <= th]]
            data_1 = [[xi, 2 * (random.random() - 0.5)] for xi in x[abs(zs) > th]]
        else:
            y = self.data["X"][:, 1]
            inliers_cond = (abs(x - avg0) / std0 <= th) & (abs(y - avg1) / std1 <= th)
            outliers_cond = (abs(x - avg0) / std0 > th) | (abs(y - avg1) / std1 > th)
            data_0 = np.column_stack((x[inliers_cond], y[inliers_cond])).tolist()
            data_1 = np.column_stack((x[outliers_cond], y[outliers_cond])).tolist()
        chart.add_data_set(
            data_circle, "spline", name="Outliers Border", **self.init_style_circle
        )
        chart.add_data_set(
            data_1, "scatter", name="Outliers", **self.init_style_outliers
        )
        chart.add_data_set(data_0, "scatter", name="Inliers", **self.init_style_inliers)
        return chart
