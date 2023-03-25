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


class ROCCurve(HighchartsBase):

    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["roc"]:
        return "roc"

    # Styling Methods.

    def _init_style(self) -> None:
        auc = self.data["auc"]
        self.init_style = {
            "title": {"text": ""},
            "xAxis": {
                "reversed": False,
                "title": {"enabled": True, "text": self.layout["x_label"]},
                "startOnTick": True,
                "endOnTick": True,
                "showLastLabel": True,
                "min": 0.0,
                "max": 1.0,
            },
            "yAxis": {
                "title": {"text": self.layout["y_label"]},
                "min": 0.0,
                "max": 1.0,
            },
            "legend": {"enabled": False},
            "subtitle": {"text": f"AUC = {round(auc, 4) * 100}%", "align": "left",},
            "tooltip": {"headerFormat": "", "pointFormat": "[{point.x}, {point.y}]",},
            "colors": self.get_colors(),
        }
        return None

    # Draw.

    def draw(self, chart: Optional[HChart] = None, **style_kwargs,) -> HChart:
        """
        Draws a Machine Learning Roc Curve using the Matplotlib API.
        """
        chart = self._get_chart(chart)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        data = np.column_stack((self.data["x"], self.data["y"])).tolist()
        chart.add_data_set(data, "area", self.layout["x_label"], step=True)
        return chart


class PRCCurve(ROCCurve):

    # Properties.

    @property
    def _kind(self) -> Literal["prc"]:
        return "prc"
