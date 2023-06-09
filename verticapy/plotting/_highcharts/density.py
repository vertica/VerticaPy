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
from typing import Literal, Optional

import numpy as np

from verticapy._typing import HChart
from verticapy.plotting._highcharts.base import HighchartsBase


class DensityPlot(HighchartsBase):
    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["density"]:
        return "density"

    # Styling Methods.

    def _init_style(self) -> None:
        x_label = self._clean_quotes(self.layout["x_label"])
        y_label = self._clean_quotes(self.layout["y_label"])
        X = self.data["x"] if "x" in self.data else self.data["X"]
        self.init_style = {
            "title": {"text": ""},
            "xAxis": {
                "reversed": False,
                "title": {"enabled": True, "text": x_label},
                "startOnTick": True,
                "endOnTick": True,
                "showLastLabel": True,
                "min": np.nanmin(X),
                "max": np.nanmax(X),
            },
            "yAxis": {"title": {"text": y_label}},
            "legend": {"enabled": False},
            "tooltip": {
                "headerFormat": "",
                "pointFormat": "<b>"
                + x_label
                + "</b>: {point.x} <br/> <b>"
                + y_label
                + "</b>: {point.y}",
            },
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
            "colors": self.get_colors(),
        }

    # Draw.

    def draw(
        self,
        chart: Optional[HChart] = None,
        **style_kwargs,
    ) -> HChart:
        """
        Draws a density plot using the HC API.
        """
        x_label = self._clean_quotes(self.layout["x_label"])
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        data = np.column_stack((self.data["x"], self.data["y"])).tolist()
        chart.add_data_set(data, "area", x_label)
        return chart


class MultiDensityPlot(DensityPlot):
    # Styling Methods.

    def _init_style(self) -> None:
        super()._init_style()
        self.init_style["legend"]["enabled"] = True
        self.init_style["legend"]["title"] = self._clean_quotes(
            self.layout["labels_title"]
        )
        self.init_style["tooltip"][
            "headerFormat"
        ] = '<span style="color:{series.color}">\u25CF</span> {series.name} <br/>'

    # Draw.

    def draw(
        self,
        chart: Optional[HChart] = None,
        **style_kwargs,
    ) -> HChart:
        """
        Draws a multi-density plot using the HC API.
        """
        labels = self._clean_quotes(self.layout["labels"])
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        m = self.data["X"].shape[1]
        for i in range(m):
            data = np.column_stack(
                (self.data["X"][:, i], self.data["Y"][:, i])
            ).tolist()
            chart.add_data_set(data, "area", labels[i])
        return chart
