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
from verticapy.plotting._highcharts.machine_learning.svm import SVMClassifierPlot


class LogisticRegressionPlot(SVMClassifierPlot):
    # Properties.

    @property
    def _kind(self) -> Literal["logit"]:
        return "logit"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (2, 2)

    # Styling Methods.

    def _init_style(self) -> None:
        super()._init_style()
        self.init_style["yAxis"]["title"]["text"] = (
            "p(" + self.init_style["yAxis"]["title"]["text"] + "=1)"
        )
        self.init_style_scatter["zIndex"] = 1
        self.init_style_line["zIndex"] = 0

    # Draw.

    def draw(self, chart: Optional[HChart] = None, **style_kwargs) -> HChart:
        """
        Draws a logistic regression plot using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        if len(self.layout["columns"]) == 2:

            def logit(x: float) -> float:
                return 1 / (1 + np.exp(-x))

            x, w = self.data["X"][:, 0], self.data["X"][:, -1]
            x0, x1 = x[w == 0], x[w == 1]
            min_logit_x, max_logit_x = np.nanmin(x), np.nanmax(x)
            step_x = (max_logit_x - min_logit_x) / 100.0
            x_logit = (
                np.arange(min_logit_x - step_x, max_logit_x + step_x, step_x)
                if (step_x > 0)
                else np.array([max_logit_x])
            )
            y_logit = logit(self.data["coef"][0] + self.data["coef"][1] * x_logit)
            data_logit = np.column_stack((x_logit, y_logit)).tolist()
            data_0 = np.column_stack(
                (x0, logit(self.data["coef"][0] + self.data["coef"][1] * x0))
            ).tolist()
            data_1 = np.column_stack(
                (x1, logit(self.data["coef"][0] + self.data["coef"][1] * x1))
            ).tolist()
        else:
            raise ValueError("The number of predictors is too big to draw the plot.")
        chart.add_data_set(data_logit, "line", name="Logit", **self.init_style_line)
        chart.add_data_set(data_1, "scatter", name="+1", **self.init_style_scatter)
        chart.add_data_set(data_0, "scatter", name="-1", **self.init_style_scatter)
        return chart
