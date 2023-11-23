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


class Histogram(HighchartsBase):
    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["hist"]:
        return "hist"

    @property
    def _compute_method(self) -> Literal["hist"]:
        return "hist"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "title": {"text": ""},
            "chart": {"type": "column"},
            "legend": {"enabled": True},
            "colors": self.get_colors(),
            "yAxis": {"title": {"text": self.layout["method_of"]}},
            "legend": {
                "enabled": True,
            },
        }
        self.init_style_stacked = {"plotOptions": {"series": {"stacking": "overlap"}}}

    # Draw.

    def draw(
        self,
        chart: Optional[HChart] = None,
        **style_kwargs,
    ) -> HChart:
        """
        Draws a Histogram using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        for label in self.layout["columns"]:
            data_to_plot = np.column_stack(
                (
                    self.data[label]["x"],
                    [float(decimal) for decimal in self.data[label]["y"]],
                )
            ).tolist()
            chart.add_data_set(
                data_to_plot,
                "column",
                name=str(label),
                pointPadding=0,
                groupPadding=0,
                opacity=0.7,
            )
        chart.set_dict_options(self.init_style_stacked)
        return chart
