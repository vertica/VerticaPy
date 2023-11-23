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
from verticapy.plotting._highcharts.bar import DrillDownBarChart


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
            "legend": {"enabled": False},
            "colors": [self.get_colors(idx=0)],
            "xAxis": {
                "type": "category",
                "title": {"text": self.layout["column"]},
                "categories": self.layout["labels"],
            },
            "yAxis": {"title": {"text": self.layout["method_of"]}},
            "tooltip": {"headerFormat": "", "pointFormat": "{point.y}"},
        }
        self.init_style_bar = {"pointPadding": self.data["bargap"] / 2}

    # Draw.

    def draw(
        self,
        chart: Optional[HChart] = None,
        **style_kwargs,
    ) -> HChart:
        """
        Draws a histogram using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        chart.add_data_set(
            self.data["y"], "bar", self.layout["column"], **self.init_style_bar
        )
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
            "legend": {"enabled": False},
            "xAxis": {
                "type": "category",
                "title": {"text": self.layout["columns"][0]},
                "categories": self.layout["x_labels"],
            },
            "yAxis": {"title": {"text": self.layout["method_of"]}},
            "legend": {"enabled": True, "title": {"text": self.layout["columns"][1]}},
            "colors": self.get_colors(),
        }
        self.init_style_stacked = {"plotOptions": {"series": {"stacking": "normal"}}}
        self.init_style_fstacked = {"plotOptions": {"series": {"stacking": "percent"}}}
        if self.layout["kind"] == "density":
            columns = self.layout["columns"]
            x_labels = self.layout["x_labels"]
            n, m = self.data["X"].shape
            if m != 2 and n == 2:
                columns.reverse()
                x_labels = self.layout["y_labels"]
            self.init_style = {
                "chart": {"type": "bar"},
                "title": {"text": ""},
                "subtitle": {"text": ""},
                "xAxis": [
                    {
                        "categories": x_labels,
                        "reversed": False,
                        "labels": {"step": 1},
                    },
                    {
                        "opposite": True,
                        "reversed": False,
                        "categories": x_labels,
                        "linkedTo": 0,
                        "labels": {"step": 1},
                    },
                ],
                "yAxis": {
                    "title": {"text": None},
                    "labels": {
                        "formatter": "function () {return (Math.abs(this.value));}"
                    },
                },
                "plotOptions": {"series": {"stacking": "normal"}},
                "tooltip": {
                    "formatter": "function () {return '<b>"
                    + columns[0]
                    + " : </b>' + this.series.name + '<br>' + '<b>"
                    + columns[1]
                    + "</b> : ' + '' + this.point.category + '<br/>' + '<b>"
                    + self.layout["method_of"]
                    + "</b> : ' + Math.abs(this.point.y);}"
                },
                "colors": self.get_colors(),
            }

    # Draw.

    def draw(self, chart: Optional[HChart] = None, **style_kwargs) -> HChart:
        """
        Draws a 2D BarChart using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        if self.layout["kind"] == "density":
            lookup = {0: 1, 1: -1}
            X = self.data["X"]
            y_labels = self.layout["y_labels"]
            n, m = X.shape
            if m != 2 and n == 2:
                X = np.transpose(X)
                y_labels = self.layout["x_labels"]
            for idx, label in enumerate(y_labels):
                chart.add_data_set(
                    list(lookup[idx] * X[:, idx]), "bar", name=str(label)
                )
        else:
            for idx, label in enumerate(self.layout["y_labels"]):
                chart.add_data_set(list(self.data["X"][:, idx]), "bar", name=str(label))
            if self.layout["kind"] == "stacked":
                chart.set_dict_options(self.init_style_stacked)
            elif self.layout["kind"] == "fully_stacked":
                chart.set_dict_options(self.init_style_fstacked)
        return chart


class DrillDownHorizontalBarChart(DrillDownBarChart):
    # Properties.

    @property
    def _kind(self) -> Literal["barh"]:
        return "barh"

    # Styling Methods.

    def _init_style(self) -> None:
        super()._init_style()
        self.init_style["chart"] = {"type": "column", "inverted": True}
