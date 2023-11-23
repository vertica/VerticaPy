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

from verticapy._typing import HChart, NoneType
from verticapy.plotting._highcharts.base import HighchartsBase


class BoxPlot(HighchartsBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["box"]:
        return "box"

    @property
    def _compute_method(self) -> Literal["describe"]:
        return "describe"

    # Styling Methods.

    def _init_style(self) -> None:
        labels = self.layout["labels"]
        y_label = self.layout["y_label"]
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        if isinstance(y_label, NoneType):
            pointFormat = ""
        else:
            pointFormat = f"{y_label}: "
        self.init_style = {
            "chart": {"type": "boxplot"},
            "title": {"text": ""},
            "legend": {"enabled": False},
            "xAxis": {
                "categories": labels,
                "title": {"text": self.layout["x_label"]},
            },
            "yAxis": {"title": {"text": y_label}},
            "colors": self.get_colors(),
        }
        self.init_style_boxplot = {
            "tooltip": {"headerFormat": "<em>{point.key}</em><br/>"},
            "colorByPoint": True,
            "fillColor": "#FDFDFD",
        }
        self.init_style_scatter = {
            "color": "#444444",
            "marker": {
                "fillColor": "white",
                "lineWidth": 1,
                "lineColor": "#444444",
            },
            "tooltip": {"pointFormat": pointFormat + "{point.y}"},
        }
        if len(labels) == 1:
            self.init_style["chart"]["inverted"] = True

    # Draw.

    def draw(
        self,
        chart: Optional[HChart] = None,
        **style_kwargs,
    ) -> HChart:
        """
        Draws a multi-box plot using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        chart.add_data_set(
            np.transpose(self.data["X"]).tolist(),
            "boxplot",
            "Quantiles",
            **self.init_style_boxplot,
        )
        fliers = []
        for i, fli in enumerate(self.data["fliers"]):
            for flj in fli:
                fliers += [[i, flj]]
        chart.add_data_set(fliers, "scatter", "Outliers", **self.init_style_scatter)
        return chart
