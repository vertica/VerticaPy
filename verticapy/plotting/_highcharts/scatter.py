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
import warnings
from typing import Literal, Optional

import numpy as np

from verticapy._typing import HChart
from verticapy.plotting._highcharts.base import HighchartsBase


class ScatterPlot(HighchartsBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["scatter"]:
        return "scatter"

    @property
    def _compute_method(self) -> Literal["sample"]:
        return "sample"

    # Styling Methods.

    def _init_style(self) -> None:
        tooltip = {
            "headerFormat": "",
            "pointFormat": "<b>"
            + str(self.layout["columns"][0])
            + "</b>: {point.x}<br><b>"
            + str(self.layout["columns"][1])
            + "</b>: {point.y}<br>",
        }
        if len(self.layout["columns"]) > 2:
            tooltip["pointFormat"] += (
                "<b>" + str(self.layout["columns"][2]) + "</b>: {point.z}"
            )
        elif self.layout["has_size"]:
            tooltip["pointFormat"] += (
                "<b>" + str(self.layout["size"]) + "</b>: {point.z}"
            )
        self.init_style = {
            "title": {"text": ""},
            "xAxis": {
                "reversed": False,
                "title": {"enabled": True, "text": self.layout["columns"][0]},
                "startOnTick": True,
                "endOnTick": True,
                "showLastLabel": True,
            },
            "yAxis": {"title": {"text": self.layout["columns"][1]}},
            "legend": {"enabled": False},
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
            "tooltip": tooltip,
            "colors": self.get_colors(),
        }
        if len(self.layout["columns"]) > 2:
            # BUG for 3D graphics, can not change default color.
            del self.init_style["colors"]
            self.init_style_3d = {
                "chart": {
                    "renderTo": "container",
                    "margin": 100,
                    "type": "scatter",
                    "options3d": {
                        "enabled": True,
                        "alpha": 10,
                        "beta": 30,
                        "depth": 400,
                        "viewDistance": 8,
                        "frame": {
                            "bottom": {"size": 1, "color": "rgba(0,0,0,0.02)"},
                            "back": {"size": 1, "color": "rgba(0,0,0,0.04)"},
                            "side": {"size": 1, "color": "rgba(0,0,0,0.06)"},
                        },
                    },
                },
                "zAxis": {"title": {"text": self.layout["columns"][2]}},
            }
        if self.layout["has_category"]:
            self.init_style["tooltip"][
                "headerFormat"
            ] = '<span style="color:{series.color}">\u25CF</span> {series.name} <br/>'
            self.init_style_cat = {
                "legend": {"enabled": True, "title": {"text": self.layout["c"]}}
            }

    # Draw.

    def draw(self, chart: Optional[HChart] = None, **style_kwargs) -> HChart:
        """
        Draws a scatter plot using the HC API.
        """
        has_cmap = self.layout["has_cmap"]
        if has_cmap:
            warning_message = f"The parameter {has_cmap} is not supported on the Highchart API. It is ignored."
            warnings.warn(warning_message, Warning)
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        kind = "bubble" if self.layout["has_size"] else "scatter"
        if len(self.layout["columns"]) > 2:
            chart.set_dict_options(self.init_style_3d)
            chart.add_3d_rotation()
            chart.add_JSsource("https://code.highcharts.com/6/highcharts-3d.js")
        if self.layout["has_category"]:
            chart.set_dict_options(self.init_style_cat)
            uniques = np.unique(self.data["c"])
            for c in uniques:
                data = self.data["X"][self.data["c"] == c]
                if self.layout["has_size"]:
                    data = np.column_stack((data, self.data["s"][self.data["c"] == c]))
                data = data.tolist()
                chart.add_data_set(data, kind, str(c))
        else:
            data = self.data["X"]
            if self.layout["has_size"]:
                data = np.column_stack((data, self.data["s"]))
            data = data.tolist()
            chart.add_data_set(data, kind)
        return chart
