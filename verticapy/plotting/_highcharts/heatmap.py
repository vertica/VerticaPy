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
import copy
from typing import Literal, Optional

import numpy as np

from verticapy._typing import HChart, NoneType
from verticapy.plotting._highcharts.base import HighchartsBase


class HeatMap(HighchartsBase):
    # Properties.

    @property
    def _category(self) -> Literal["map"]:
        return "map"

    @property
    def _kind(self) -> Literal["heatmap"]:
        return "heatmap"

    @property
    def _compute_method(self) -> Literal["2D"]:
        return "2D"

    # Styling Methods.

    def _init_style(self) -> None:
        if len(self.layout["columns"]) > 1:
            y_label = self.layout["columns"][1]
        elif "method_of" in self.layout:
            y_label = self.layout["method_of"]
        else:
            y_label = ""
        if isinstance(self.layout["y_labels"], list):
            y_labels = copy.deepcopy(self.layout["y_labels"])
        else:
            y_labels = self.layout["y_labels"]
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
                "categories": self.layout["x_labels"],
                "title": {"text": self.layout["columns"][0]},
            },
            "yAxis": {
                "categories": y_labels,
                "title": {"text": y_label},
            },
            "legend": {
                "align": "right",
                "layout": "vertical",
                "margin": 0,
                "verticalAlign": "top",
                "y": 25,
                "symbolHeight": max(self.data["X"].shape[1] * 60, 220) * 0.7 - 25,
            },
            "colors": ["#EFEFEF"],
        }
        self.init_style_matrix = {
            "series_type": "heatmap",
            "borderWidth": 1,
            "dataLabels": {
                "enabled": (
                    "with_numbers" in self.layout and self.layout["with_numbers"]
                ),
                "color": "#000000",
            },
        }

    def _get_cmap_style(self, style_kwargs: dict) -> dict:
        if (
            "colorAxis" not in style_kwargs
            and "method" in self.layout
            and (
                self.layout["method"]
                in (
                    "pearson",
                    "spearman",
                    "spearmand",
                    "kendall",
                    "biserial",
                )
            )
        ):
            d = {
                "stops": [
                    [0, self.get_colors(idx=1)],
                    [0.45, "#FFFFFF"],
                    [0.55, "#FFFFFF"],
                    [1, self.get_colors(idx=0)],
                ],
                "min": -1,
                "max": 1,
            }
        elif (
            "colorAxis" not in style_kwargs
            and "method" in self.layout
            and self.layout["method"] == "cramer"
        ):
            d = {
                "stops": [
                    [0, "#FFFFFF"],
                    [0.2, "#FFFFFF"],
                    [1, self.get_colors(idx=0)],
                ],
                "min": 0,
                "max": 1,
            }
        elif "colorAxis" not in style_kwargs:
            d = {"minColor": "#FFFFFF", "maxColor": self.get_colors(idx=0)}
        else:
            d = {}
        for vm in ["vmin", "vmax"]:
            if vm in self.layout and not isinstance(self.layout[vm], NoneType):
                d[vm[1:]] = self.layout[vm]
        return d

    # Draw.

    def draw(self, chart: Optional[HChart] = None, **style_kwargs) -> HChart:
        """
        Draws a heatmap using the HC API.
        """
        n, m = self.data["X"].shape
        chart, style_kwargs = self._get_chart(
            chart,
            width=max(n * 80, 400),
            height=max(m * 60, 220),
            style_kwargs=style_kwargs,
        )
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        X = np.flip(self.data["X"], axis=1)
        data = []
        for i in range(len(self.layout["x_labels"])):
            for j in range(len(self.layout["y_labels"])):
                if "mround" in self.layout:
                    Xij = round(X[i, j], self.layout["mround"])
                else:
                    Xij = X[i, j]
                data += [[i, j, Xij]]
        chart.add_data_set(data, **self.init_style_matrix)
        chart.set_dict_options(
            {"colorAxis": self._get_cmap_style(style_kwargs=style_kwargs)}
        )
        return chart
