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


class BarChart(HighchartsBase):
    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["bar"]:
        return "bar"

    @property
    def _compute_method(self) -> Literal["1D"]:
        return "1D"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "title": {"text": ""},
            "chart": {"type": "column"},
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
        Draws a BarChart using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        chart.add_data_set(
            [float(val) for val in self.data["y"]],
            "bar",
            self.layout["column"],
            **self.init_style_bar,
        )
        return chart


class BarChart2D(HighchartsBase):
    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["bar"]:
        return "bar"

    @property
    def _compute_method(self) -> Literal["2D"]:
        return "2D"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "title": {"text": ""},
            "chart": {"type": "column"},
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

    # Draw.

    def draw(
        self,
        chart: Optional[HChart] = None,
        **style_kwargs,
    ) -> HChart:
        """
        Draws a 2D BarChart using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        for idx, label in enumerate(self.layout["y_labels"]):
            chart.add_data_set(list(self.data["X"][:, idx]), "bar", name=str(label))
        if self.layout["kind"] == "stacked":
            chart.set_dict_options(self.init_style_stacked)
        return chart


class DrillDownBarChart(HighchartsBase):
    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["bar"]:
        return "bar"

    @property
    def _compute_method(self) -> Literal["rollup"]:
        return "rollup"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "chart": {"type": "column"},
            "title": {"text": ""},
            "subtitle": {"text": ""},
            "xAxis": {"type": "category"},
            "yAxis": {"title": {"text": self.layout["method_of"]}},
            "legend": {"enabled": False},
            "plotOptions": {
                "series": {"borderWidth": 0, "dataLabels": {"enabled": True}}
            },
            "tooltip": {
                "headerFormat": "",
                "pointFormat": '<span style="color:{point.color}">{point.name}</span>: <b>{point.y}</b><br/>',
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
        Draws a 2D BarChart using the HC API.
        """
        kind = "bar" if self._kind == "barh" else "column"
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        chart.add_JSsource("https://code.highcharts.com/6/modules/drilldown.js")
        init_group = np.column_stack(self.data["groups"][1])
        data = []
        for row in init_group:
            data += [
                {"name": str(row[0]), "y": float(row[1]), "drilldown": str(row[0])}
            ]
        chart.add_data_set(data, kind, colorByPoint=True)
        drilldown_group = np.column_stack(self.data["groups"][0])
        uniques = np.unique(drilldown_group[:, 0])
        for c in uniques:
            data = drilldown_group[drilldown_group[:, 0] == c].tolist()
            data = [(str(x[1]), float(x[2])) for x in data]
            chart.add_drilldown_data_set(data, kind, str(c), name=str(c))
        return chart
