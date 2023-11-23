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
from datetime import date, datetime
from typing import Literal, Optional

import numpy as np

from verticapy._typing import HChart
from verticapy.plotting._highcharts.line import LinePlot


class TSPlot(LinePlot):
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
        colors = [c for c in self.get_colors()]
        if self.layout["is_forecast"]:
            del colors[1]
        self.init_style = {
            "title": {"text": ""},
            "xAxis": {
                "reversed": False,
                "title": {"enabled": True, "text": self.layout["order_by"]},
                "startOnTick": True,
                "endOnTick": True,
                "showLastLabel": True,
            },
            "yAxis": {"title": {"text": self.layout["columns"]}},
            "legend": {"enabled": True},
            "plotOptions": {
                "scatter": {
                    "marker": {
                        "radius": 5,
                        "states": {
                            "hover": {
                                "enabled": True,
                                "lineColor": "rgb(100,100,100)",
                            }
                        },
                    },
                    "states": {"hover": {"marker": {"enabled": False}}},
                    "tooltip": {
                        "headerFormat": '<span style="color:{series.color}">\u25CF</span> {series.name} <br/>',
                        "pointFormat": "<b>"
                        + self.layout["order_by"]
                        + "</b>: {point.x} <br/> <b>"
                        + self.layout["columns"],
                    },
                }
            },
            "colors": colors,
        }
        self.init_style_area_range = {
            "zIndex": 0,
            "lineWidth": 0,
            "fillOpacity": 0.3,
        }
        for x in self.data["x"]:
            if isinstance(x, (date, datetime)):
                self.init_style["xAxis"] = {
                    **self.init_style["xAxis"],
                    "type": "datetime",
                    "dateTimeLabelFormats": {},
                }
                break

    # Draw.

    def draw(
        self,
        chart: Optional[HChart] = None,
        **style_kwargs,
    ) -> HChart:
        """
        Draws a time series plot using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        # True Values
        x = self._to_datetime(self.data["x"])
        data = np.column_stack((x, self.data["y"])).tolist()
        chart.add_data_set(
            data,
            "line",
            self.layout["columns"],
        )
        # One step ahead forecast
        if not (self.layout["is_forecast"]):
            x = self._to_datetime(self.data["x_pred_one"])
            data = np.column_stack((x, self.data["y_pred_one"])).tolist()
            chart.add_data_set(
                data,
                "line",
                "one-sted-ahead-forecast",
            )
        # Forecast
        x = self._to_datetime(self.data["x_pred"])
        data = np.column_stack((x, self.data["y_pred"])).tolist()
        chart.add_data_set(
            data,
            "line",
            "forecast",
        )
        # Std Error
        if self.layout["has_se"]:
            x = self._to_datetime(self.data["se_x"])
            data_range = np.column_stack(
                (x, self.data["se_low"], self.data["se_high"])
            ).tolist()
            chart.add_data_set(
                data_range,
                "arearange",
                "95% confidence interval",
                **self.init_style_area_range,
            )
        return chart
