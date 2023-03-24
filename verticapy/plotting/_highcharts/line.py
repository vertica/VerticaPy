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
import copy
from datetime import date, datetime
from typing import Literal, Union
import numpy as np

from vertica_highcharts import Highchart, Highstock

from verticapy.plotting._highcharts.base import HighchartsBase


class LinePlot(HighchartsBase):

    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["line"]:
        return "line"

    @property
    def _compute_method(self) -> Literal["line"]:
        return "line"

    # Formatting Methods.

    @staticmethod
    def _to_datetime(x: list) -> list:
        if len(x) > 0 and not (isinstance(x[0], datetime)) and isinstance(x[0], date):
            return [datetime.combine(d, datetime.min.time()) for d in x]
        else:
            return copy.deepcopy(x)

    # Styling Methods.

    def _init_style(self) -> None:
        if "stock" in self.layout and self.layout["stock"]:
            self.init_style = {
                "rangeSelector": {"selected": 0},
                "title": {"text": ""},
                "tooltip": {
                    "style": {"width": "200px"},
                    "valueDecimals": 4,
                    "shared": True,
                },
                "yAxis": {"title": {"text": ""}},
            }
        else:
            self.init_style = {
                "title": {"text": ""},
                "xAxis": {
                    "reversed": False,
                    "title": {"enabled": True, "text": self.layout["order_by"]},
                    "startOnTick": True,
                    "endOnTick": True,
                    "showLastLabel": True,
                },
                "yAxis": {
                    "title": {
                        "text": self.layout["columns"][0]
                        if len(self.layout["columns"]) == 1
                        else ""
                    }
                },
                "legend": {"enabled": False},
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
                        "tooltip": {
                            "headerFormat": "",
                            "pointFormat": "[{point.x}, {point.y}]",
                        },
                    }
                },
                "colors": self.get_colors(),
            }
        if self.layout["order_by_cat"] == "date":
            self.init_style["xAxis"] = {
                **self.init_style["xAxis"],
                "type": "datetime",
                "dateTimeLabelFormats": {},
            }
        if self.layout["has_category"]:
            self.init_style["legend"] = {
                "enabled": True,
                "title": {"text": self.layout["columns"][1]},
            }
        elif len(self.layout["columns"]) > 1:
            self.init_style["legend"] = {"enabled": True}
        return None

    # Draw.

    def draw(self, **style_kwargs,) -> Union[Highchart, Highstock]:
        """
        Draws a time series plot using the Matplotlib API.
        """
        if "stock" in self.layout and self.layout["stock"]:
            chart = Highstock(width=600, height=400)
        else:
            chart = Highchart(width=600, height=400)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        if self.layout["has_category"]:
            uniques = np.unique(self.data["z"])
            for i, c in enumerate(uniques):
                x = self._to_datetime(self.data["x"][self.data["z"] == c])
                y = self.data["Y"][:, 0][self.data["z"] == c]
                data = np.column_stack((x, y)).tolist()
                chart.add_data_set(data, self.layout["kind"], c)
        else:
            x = self._to_datetime(self.data["x"])
            y = self.data["Y"][:, 0]
            data = np.column_stack((x, y)).tolist()
            chart.add_data_set(data, self.layout["kind"], self.layout["columns"][0])
        return chart


class MultiLinePlot(LinePlot):

    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["line"]:
        return "line"

    @property
    def _compute_method(self) -> Literal["line"]:
        return "line"

    # Draw.

    def draw(self, **style_kwargs,) -> Union[Highchart, Highstock]:
        """
        Draws a multi-time series plot using the Matplotlib API.
        """
        if "stock" in self.layout and self.layout["stock"]:
            chart = Highstock(width=600, height=400)
        else:
            chart = Highchart(width=600, height=400)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        n, m = self.data["Y"].shape
        x = self._to_datetime(self.data["x"])
        for idx in range(m):
            y = self.data["Y"][:, idx]
            data = np.column_stack((x, y)).tolist()
            chart.add_data_set(data, self.layout["kind"], self.layout["columns"][idx])
        return chart
