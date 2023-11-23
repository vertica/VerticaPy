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
import random
from typing import Literal, Optional

import numpy as np

from verticapy._typing import HChart
from verticapy.plotting._highcharts.base import HighchartsBase


class SVMClassifierPlot(HighchartsBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["svm"]:
        return "svm"

    @property
    def _compute_method(self) -> Literal["sample"]:
        return "sample"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (2, 3)

    # Styling Methods.

    def _init_style(self) -> None:
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
                }
            },
            "tooltip": {
                "headerFormat": '<span style="color:{series.color}">\u25CF</span> {series.name} <br/>',
                "pointFormat": "<b>"
                + self.layout["columns"][0]
                + "</b>: {point.x} <br/> <b>"
                + self.layout["columns"][1]
                + "</b>: {point.y}",
            },
            "colors": self.get_colors(),
        }
        self.init_style_scatter = {
            "marker": {"lineWidth": 1},
            "zIndex": 0,
        }
        self.init_style_line = {
            "zIndex": 1,
            "color": "black",
        }

    # Draw.

    def draw(self, chart: Optional[HChart] = None, **style_kwargs) -> HChart:
        """
        Draws a SVM classifier plot using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        x, w = self.data["X"][:, 0], self.data["X"][:, -1]
        x0, x1 = x[w == 0], x[w == 1]
        if len(self.layout["columns"]) == 2:
            data_svm = [
                [-self.data["coef"][0] / self.data["coef"][1], i / 100]
                for i in range(-100, 101)
            ]
            data_1 = [[xi, 2 * (random.random() - 0.5)] for xi in x1]
            data_0 = [[xi, 2 * (random.random() - 0.5)] for xi in x0]
        elif len(self.layout["columns"]) == 3:
            y = self.data["X"][:, 1]
            y0, y1 = y[w == 0], y[w == 1]
            min_svm_x, max_svm_x = np.nanmin(x), np.nanmax(x)
            data_svm = [
                [
                    (min_svm_x + (max_svm_x - min_svm_x) * (i / 100)),
                    -(
                        self.data["coef"][0]
                        + self.data["coef"][1]
                        * (min_svm_x + (max_svm_x - min_svm_x) * (i / 100))
                    )
                    / self.data["coef"][2],
                ]
                for i in range(101)
            ]
            data_1 = np.column_stack((x1, y1)).tolist()
            data_0 = np.column_stack((x0, y0)).tolist()
        else:
            raise ValueError("The number of predictors is too big to draw the plot.")
        chart.add_data_set(data_svm, "line", name="SVM", **self.init_style_line)
        chart.add_data_set(data_1, "scatter", name="+1", **self.init_style_scatter)
        chart.add_data_set(data_0, "scatter", name="-1", **self.init_style_scatter)
        return chart
