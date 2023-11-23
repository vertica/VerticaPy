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

from verticapy._typing import HChart
from verticapy.plotting._highcharts.base import HighchartsBase


class SpiderChart(HighchartsBase):
    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["spider"]:
        return "spider"

    @property
    def _compute_method(self) -> Literal["2D"]:
        return "2D"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "chart": {"polar": True, "type": "line", "renderTo": "test"},
            "title": {"text": "", "x": -80},
            "pane": {"size": "80%"},
            "xAxis": {
                # "title": {"text": self.layout["columns"][0]},
                "categories": self.layout["x_labels"],
                "tickmarkPlacement": "on",
                "lineWidth": 0,
            },
            "yAxis": {
                "gridLineInterpolation": "polygon",
                "lineWidth": 0,
                "min": 0,
            },
            "tooltip": {
                "shared": True,
                "pointFormat": '<span style="color:{series.color}">{series.name}: <b>{point.y}</b><br/>',
            },
            "legend": {
                "enabled": False,
            },
            "colors": self.get_colors(),
        }
        if len(self.layout["columns"]) > 1:
            self.init_style["legend"] = {
                "title": {"text": self.layout["columns"][1]},
                "align": "right",
                "verticalAlign": "top",
                "y": 70,
                "layout": "vertical",
            }

    # Draw.

    def draw(
        self,
        chart: Optional[HChart] = None,
        **style_kwargs,
    ) -> HChart:
        """
        Draws a spider plot using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        for idx, label in enumerate(self.layout["y_labels"]):
            chart.add_data_set(
                list(self.data["X"][:, idx]), name=str(label), pointPlacement="on"
            )
        return chart
