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
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure

from verticapy.plotting._plotly.base import PlotlyBase


class LogisticRegressionPlot(PlotlyBase):

    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["logit"]:
        return "logit"

    @property
    def _compute_method(self) -> Literal["sample"]:
        return "sample"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (2, 3)

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style_0 = {
            "marker": "o",
            "s": 50,
            "color": self.get_colors(idx=0),
            "edgecolors": "black",
            "alpha": 0.8,
        }
        self.init_style_1 = {
            "marker": "o",
            "s": 50,
            "color": self.get_colors(idx=1),
            "edgecolors": "black",
        }
        self.init_style_Z = {
            "rstride": 1,
            "cstride": 1,
            "alpha": 0.5,
            "color": "gray",
        }
        return None

    # Draw.

    def draw(self, fig: Optional[Figure] = None, **style_kwargs,) -> Figure:
        """
        Draws a Logistic Regression plot using the Matplotlib API.
        """
        fig = self._get_fig(fig)
        logit = lambda x: 1 / (1 + np.exp(-x))
        x, z = self.data["X"][:, 0], self.data["X"][:, -1]
        x0, x1 = x[z == 0], x[z == 1]
        min_logit_x, max_logit_x = min(self.data["X"][:, 0]), max(self.data["X"][:, 0])
        y0 = self.data["coef"][0]
        slope = self.data["coef"][1]
        step_x = (max_logit_x - min_logit_x) / 40.0
        x_logit = (
            np.arange(min_logit_x - 5 * step_x, max_logit_x + 5 * step_x, step_x)
            if (step_x > 0)
            else np.array([max_logit_x])
        )
        y_logit=logit(self.data["coef"][0] + self.data["coef"][1] * x_logit)

        # y = logit(y0 + slope * x)
        if len(self.layout["columns"]) == 2:
            fig.add_trace(
                go.Scatter(x=x0, y=logit(y0 + slope * x0), 
                #marker_color="black",
                name="-1",mode="markers",
                        #marker_colorscale=["black","orange"]
                        )
            )
            fig.add_trace(
                go.Scatter(x=x1, y=logit(y0 + slope * x0), 
                #marker_color="orange",
                name="+1",mode="markers",
                        #marker_colorscale=["black","orange"]
                        )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_logit,
                    y=y_logit,
                    mode="lines",
                    line_shape="linear",
                    name="Logit",
                )
            )
        elif len(self.layout["columns"]) == 3:
            pass
        else:
            raise ValueError("The number of predictors is too big.")
        return fig
