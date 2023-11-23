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


class ACFPlot(HighchartsBase):
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
            "title": {"text": ""},
            "chart": {"type": "column"},
            "legend": {"enabled": True},
            "colors": self.get_colors(),
            "xAxis": {
                "type": "category",
                "title": {"text": "lag"},
                "categories": self.data["x"].tolist(),
            },
            "yAxis": {"title": {"text": "value"}, "max": 1.2, "min": -1.2},
            "tooltip": {
                "headerFormat": '<span style="color:{series.color}">\u25CF</span> {series.name} <br/>',
                "pointFormat": "<b>lag</b>: {point.x} <br/> <b>value</b>: {point.y}",
            },
        }
        self.init_style_bar = {
            "pointPadding": 0.5,
            "zIndex": 1,
            "linkedTo": ":previous",
        }
        self.init_style_scatter = {
            "marker": {
                "fillColor": "white",
                "lineWidth": 1,
                "lineColor": self.get_colors(idx=0),
            },
            "zIndex": 2,
        }
        self.init_style_confidence = {
            "plotOptions": {
                "series": {
                    "opacity": 0.1,
                }
            },
            "tooltip": {
                "headerFormat": '<span style="color:{series.color}">\u25CF</span> {series.name} <br/>',
                "pointFormat": "<b>lag</b>: {point.x} <br/> <b>value</b>: {point.high}",
            },
            "zIndex": 0,
            "fillOpacity": 0.3,
            "marker": {
                "enabled": False,
            },
        }

    # Draw.

    def draw(
        self,
        chart: Optional[HChart] = None,
        **style_kwargs,
    ) -> HChart:
        """
        Draws an ACF time series plot using the HC API.
        """
        kind = "PACF" if self.layout["pacf"] else "ACF"
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        if "colors" in style_kwargs:
            self.init_style_confidence["fillColor"] = (
                style_kwargs["colors"][0]
                if isinstance(style_kwargs["colors"], list)
                else style_kwargs["colors"]
            )
            style_kwargs.pop("colors")
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        if self.layout["kind"] == "bar":
            chart.add_data_set(
                self.data["y"].tolist(), "scatter", kind, **self.init_style_scatter
            )
            chart.add_data_set(
                self.data["y"].tolist(), "bar", kind, **self.init_style_bar
            )
        else:
            chart.add_data_set(self.data["y"].tolist(), "line", kind)
        confidence = np.column_stack(
            (self.data["x"], -self.data["z"], self.data["z"])
        ).tolist()
        chart.add_data_set(
            confidence, "arearange", "confidence", **self.init_style_confidence
        )
        return chart


class ACFPACFPlot(ACFPlot):
    # Properties.

    @property
    def _kind(self) -> Literal["acf_pacf"]:
        return "acf_pacf"

    # Styling Methods.

    def _init_style(self) -> None:
        super()._init_style()
        self.init_style_bar["pointPadding"] = 0.3
        del self.init_style_bar["linkedTo"]

    # Draw.

    def draw(
        self,
        chart: Optional[HChart] = None,
        **style_kwargs,
    ) -> HChart:
        """
        Draws an ACF-PACF time series plot using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        for y, kind in [("y0", "ACF"), ("y1", "PACF")]:
            chart.add_data_set(
                self.data[y].tolist(), "bar", kind, **self.init_style_bar
            )
        confidence = np.column_stack(
            (self.data["x"], -self.data["z"], self.data["z"])
        ).tolist()
        chart.add_data_set(
            confidence, "arearange", "confidence", **self.init_style_confidence
        )
        return chart
