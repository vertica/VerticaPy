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

from verticapy._typing import HChart
from verticapy.plotting._highcharts.line import LinePlot


class RangeCurve(LinePlot):
    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["range"]:
        return "range"

    @property
    def _compute_method(self) -> Literal["range"]:
        return "range"

    # Styling Methods.

    def _init_style(self) -> None:
        super()._init_style()
        self.init_style = {
            **self.init_style,
            "tooltip": {"crosshairs": True, "shared": True},
        }
        self.init_style_line = {
            "zIndex": 1,
            "marker": {"fillColor": "white", "lineWidth": 2},
        }
        self.init_style_area_range = {
            "zIndex": 0,
            "lineWidth": 0,
            "linkedTo": ":previous",
            "fillOpacity": 0.3,
        }

    # Draw.

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
        colors = {**self.init_style, **style_kwargs}["colors"]
        x = self._to_datetime(self.data["x"].tolist())
        for idx, col in enumerate(self.layout["columns"]):
            y_min = self.data["Y"][:, 3 * idx]
            y = self.data["Y"][:, 3 * idx + 1]
            y_max = self.data["Y"][:, 3 * idx + 2]
            data_line = np.column_stack((x, y)).tolist()
            kwargs_line = copy.deepcopy(self.init_style_line)
            kwargs_line["marker"]["lineColor"] = colors[idx % len(colors)]
            chart.add_data_set(data_line, "line", col, **kwargs_line)
            data_range = np.column_stack((x, y_min, y_max)).tolist()
            kwargs_area_range = copy.deepcopy(self.init_style_area_range)
            kwargs_area_range["color"] = colors[idx % len(colors)]
            chart.add_data_set(data_range, "arearange", "Range", **kwargs_area_range)
        return chart
