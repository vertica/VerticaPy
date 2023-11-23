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

from verticapy._typing import HChart
from verticapy.plotting._highcharts.base import HighchartsBase


class ImportanceBarChart(HighchartsBase):
    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["importance"]:
        return "importance"

    # Styling Methods.

    def _init_style(self) -> None:
        legend = len(self.data["signs"][self.data["signs"] == -1]) != 0
        self.init_style = {
            "title": {"text": ""},
            "chart": {"type": "column", "inverted": True},
            "xAxis": {"type": "category"},
            "legend": {"enabled": legend},
            "xAxis": {
                "title": {
                    "text": self.layout["y_label"]
                    if "y_label" in self.layout
                    else "Features"
                },
                "categories": self.layout["columns"],
            },
            "yAxis": {
                "title": {
                    "text": self.layout["x_label"]
                    if "x_label" in self.layout
                    else "Importance (%)"
                }
            },
            "tooltip": {"headerFormat": "", "pointFormat": "{point.y}%"},
            "plotOptions": {"series": {"stacking": "normal"}},
            "colors": self.get_colors(),
        }

    # Draw.

    def draw(
        self,
        chart: Optional[HChart] = None,
        **style_kwargs,
    ) -> HChart:
        """
        Draws a coefficient importance bar chart using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        importances_pos = copy.deepcopy(self.data["importance"])
        importances_pos[self.data["signs"] == -1] = 0.0
        importances_pos = importances_pos.tolist()
        importances_neg = copy.deepcopy(self.data["importance"])
        importances_neg[self.data["signs"] == 1] = 0.0
        importances_neg = importances_neg.tolist()
        chart.add_data_set(importances_pos, "bar", name="+1")
        chart.add_data_set(importances_neg, "bar", name="-1")
        return chart
