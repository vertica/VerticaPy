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


class RegressionPlot(HighchartsBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["regression"]:
        return "regression"

    @property
    def _compute_method(self) -> Literal["sample"]:
        return "sample"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (2, 2)

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
                "headerFormat": "<b>{series.name}</b> <br/>",
                "pointFormat": "<b>"
                + self.layout["columns"][0]
                + "</b>: {point.x} <br/> <b>"
                + self.layout["columns"][1]
                + "</b>: {point.y}",
            },
            "colors": ["black"],
        }
        self.init_style_scatter = {
            "marker": {
                "fillColor": "white",
                "lineWidth": 1,
                "lineColor": self.get_colors(idx=0),
            },
            "zIndex": 0,
        }
        self.init_style_line = {
            "zIndex": 1,
        }

    # Draw.

    def draw(self, chart: Optional[HChart] = None, **style_kwargs) -> HChart:
        """
        Draws a regression plot using the HC API.
        """
        if len(self.layout["columns"]) == 2:
            chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
            chart.set_dict_options(self.init_style)
            chart.set_dict_options(style_kwargs)
            min_reg_x = np.nanmin(self.data["X"][:, 0])
            max_reg_x = np.nanmax(self.data["X"][:, 0])
            x_reg = [min_reg_x + (max_reg_x - min_reg_x) * i / 100 for i in range(101)]
            data = [[x, self.data["coef"][0] + self.data["coef"][1] * x] for x in x_reg]
            chart.add_data_set(data, "line", name="Prediction", **self.init_style_line)
            data = self.data["X"].tolist()
            chart.add_data_set(
                data, "scatter", name="Observations", **self.init_style_scatter
            )
            return chart
        else:
            raise ValueError("The number of predictors is too big to draw the plot.")
