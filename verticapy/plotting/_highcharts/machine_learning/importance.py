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
        importances, coef_names, signs = self._compute_importance()
        signs = np.array(signs)
        legend = len(signs[signs == -1]) != 0
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
                "categories": coef_names,
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
        return None

    # Draw.

    def draw(self, chart: Optional[HChart] = None, **style_kwargs,) -> HChart:
        """
        Draws a coeff importance bar chart using the HC API.
        """
        importances, coef_names, signs = self._compute_importance()
        chart = self._get_chart(chart)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        importances_pos = np.array(importances)
        signs = np.array(signs)
        importances_pos[signs == -1] = 0.0
        importances_pos = importances_pos.tolist()
        importances_neg = np.array(importances)
        importances_neg[signs == 1] = 0.0
        importances_neg = importances_neg.tolist()
        chart.add_data_set(importances_pos, "bar", name="+1")
        chart.add_data_set(importances_neg, "bar", name="-1")
        return chart
