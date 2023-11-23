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


class ContourPlot(HighchartsBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["contour"]:
        return "contour"

    @property
    def _compute_method(self) -> Literal["contour"]:
        return "contour"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (2, 2)

    @property
    def _max_nbins(self) -> int:
        return 23

    # Formatting Methods.

    def _get_narrow_vars(self) -> tuple:
        n, m = self.data["X"].shape
        data = []
        for i in range(n):
            for j in range(m):
                data += [[j, i, self.data["Z"][i][j]]]
        return data

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "chart": {
                "type": "heatmap",
                "marginTop": 40,
                "marginBottom": 80,
                "plotBorderWidth": 1,
            },
            "title": {"text": ""},
            "legend": {},
            "tooltip": {
                "formatter": (
                    "function () {return '<b>[' + this.series.xAxis."
                    "categories[this.point.x] + ', ' + this.series.yAxis"
                    ".categories[this.point.y] + ']</b>: ' + this.point"
                    ".value + '</b>';}"
                )
            },
            "xAxis": {
                "categories": np.round(self.data["X"][0], 3).tolist(),
                "title": {"text": self.layout["columns"][0]},
            },
            "yAxis": {
                "categories": np.round(self.data["Y"][:, 0], 3).tolist(),
                "title": {"text": self.layout["columns"][1]},
            },
            "legend": {
                "align": "right",
                "layout": "vertical",
                "margin": 0,
                "verticalAlign": "top",
                "y": 25,
                "symbolHeight": 300,
            },
            "colors": ["#EFEFEF"],
            "colorAxis": {
                "stops": [
                    [0, self.get_colors(idx=2)],
                    [0.45, "#FFFFFF"],
                    [0.55, "#FFFFFF"],
                    [1, self.get_colors(idx=0)],
                ],
            },
        }
        self.init_style_matrix = {
            "series_type": "heatmap",
            "dataLabels": {
                "enabled": False,
            },
        }

    # Draw.

    def draw(self, chart: Optional[HChart] = None, **style_kwargs) -> HChart:
        """
        Draws a contour plot using the HC API.
        """
        data = self._get_narrow_vars()
        chart, style_kwargs = self._get_chart(
            chart, width=400, height=400, style_kwargs=style_kwargs
        )
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        chart.add_data_set(data, **self.init_style_matrix)
        return chart
