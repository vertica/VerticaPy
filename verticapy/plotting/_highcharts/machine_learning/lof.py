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


class LOFPlot(HighchartsBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["lof"]:
        return "lof"

    @property
    def _compute_method(self) -> Literal["sample"]:
        return "sample"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (2, 4)

    # Styling Methods.

    def _init_style(self) -> None:
        columns = self._clean_quotes(self.layout["columns"])
        tooltip = {
            "headerFormat": '<span style="color:{series.color}">\u25CF</span> {series.name} <br/>',
            "pointFormat": "<b>" + str(columns[0]) + "</b>: {point.x}<br>",
        }
        if len(columns) > 2:
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
            "colors": self.get_colors(),
        }
        if len(self.layout["columns"]) > 2:
            self.init_style["yAxis"]["title"]["text"] = columns[1]
        self.init_style_sc = {
            "zIndex": 1,
            "marker": {
                "fillColor": "white",
                "lineWidth": 1,
                "lineColor": "black",
            },
            "tooltip": tooltip,
        }
        tooltip["pointFormat"] += "<b>" + str(columns[-1]) + "</b>: {point.z}<br>"
        self.init_style_bubble = {
            "zIndex": 0,
            "tooltip": tooltip,
        }

    # Draw.

    def draw(self, chart: Optional[HChart] = None, **style_kwargs) -> HChart:
        """
        Draws a local outlier plot using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        s = self.data["X"][:, -1]
        x = self.data["X"][:, 0]
        if len(self.layout["columns"]) == 2:
            y = np.array([0 for xi in range(len(x))])
        else:
            y = self.data["X"][:, 1]
        if 2 <= len(self.layout["columns"]) <= 3:
            data_sc = np.column_stack((x, y)).tolist()
            data_bubble = np.column_stack((x, y, s)).tolist()
        else:
            raise Exception(
                "LocalOutlierFactor Plot is available for a maximum of 2 columns."
            )
        chart.add_data_set(data_bubble, "bubble", name="LOF", **self.init_style_bubble)
        chart.add_data_set(data_sc, "scatter", name="Data Points", **self.init_style_sc)
        return chart
