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
from verticapy.plotting._highcharts.machine_learning.regression import RegressionPlot


class RegressionTreePlot(RegressionPlot):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["regression_tree"]:
        return "regression_tree"

    @property
    def _compute_method(self) -> Literal["sample"]:
        return "sample"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (3, 3)

    # Draw.

    def draw(self, chart: Optional[HChart] = None, **style_kwargs) -> HChart:
        """
        Draws a regression tree plot using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        X = self.data["X"][self.data["X"][:, 0].argsort()]
        x0 = X[:, 0]
        x1 = X[:, 0]
        y0 = X[:, 2]
        y1 = X[:, 1]
        x0, y0 = zip(*sorted(zip(x0, y0)))
        x1, y1 = zip(*sorted(zip(x1, y1)))
        data = np.column_stack((x0, y0)).tolist()
        chart.add_data_set(
            data, "line", name="Prediction", **self.init_style_line, step="right"
        )
        data = np.column_stack((x1, y1)).tolist()
        chart.add_data_set(
            data, "scatter", name="Observations", **self.init_style_scatter
        )
        return chart
