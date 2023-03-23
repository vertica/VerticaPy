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
from typing import Literal

from vertica_highcharts import Highchart

from verticapy.plotting._highcharts.base import HighchartsBase


class HorizontalBarChart(HighchartsBase):

    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["barh"]:
        return "barh"

    @property
    def _compute_method(self) -> Literal["1D"]:
        return "1D"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "title": {"text": ""},
            "chart": {"type": "column", "inverted": True},
            "xAxis": {"type": "category"},
            "legend": {"enabled": False},
            "colors": [self.get_colors(idx=0)],
            "xAxis": {
                "title": {"text": self.layout["column"]},
                "categories": self.layout["labels"],
            },
            "yAxis": {"title": {"text": self.layout["method_of"]}},
            "tooltip": {"headerFormat": "", "pointFormat": "{point.y}"},
        }
        return None

    # Draw.

    def draw(self, **style_kwargs,) -> Highchart:
        """
        Draws a histogram using the HC API.
        """
        chart = Highchart(width=600, height=400)
        chart.set_dict_options(self.init_style)
        chart.add_data_set(self.data["y"], "bar", self.layout["column"])
        return chart


class HorizontalBarChart2D(HighchartsBase):

    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["barh"]:
        return "barh"

    @property
    def _compute_method(self) -> Literal["2D"]:
        return "2D"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "title": {"text": ""},
            "chart": {"type": "column", "inverted": True},
            "xAxis": {"type": "category"},
            "legend": {"enabled": False},
            "colors": self.get_colors(),
            "xAxis": {
                "title": {"text": self.layout["columns"][0]},
                "categories": self.layout["x_labels"],
            },
            "yAxis": {"title": {"text": self.layout["method_of"]}},
            "legend": {"enabled": True, "title": {"text": self.layout["columns"][1]}},
        }
        self.init_style_stacked = {"plotOptions": {"series": {"stacking": "normal"}}}
        self.init_style_fstacked = {"plotOptions": {"series": {"stacking": "percent"}}}
        return None

    # Draw.

    def draw(self, stacked: bool = False, **style_kwargs,) -> Highchart:
        """
        Draws a 2D BarChart using the HC API.
        """
        chart = Highchart(width=600, height=400)
        chart.set_dict_options(self.init_style)
        for idx, label in enumerate(self.layout["y_labels"]):
            chart.add_data_set(list(self.data["X"][:, idx]), "bar", name=label)
        if self.layout["bar_type"] == "stacked":
            chart.set_dict_options(self.init_style_stacked)
        elif self.layout["bar_type"] == "fully_stacked":
            chart.set_dict_options(self.init_style_fstacked)
        return chart
