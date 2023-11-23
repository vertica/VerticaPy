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

import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure

from verticapy.plotting._plotly.base import PlotlyBase


class RegressionPlot(PlotlyBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["regression"]:
        return "regression"

    @property
    def _compute_method(self) -> Literal["sample"]:
        return "sample"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (2, 3)

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "mode": "markers",
            "marker_line_width": 2,
            "marker_size": 10,
            "marker_line_color": "black",
        }
        self.init_layout_style = {
            "yaxis_title": self.layout["columns"][1],
            "xaxis_title": self.layout["columns"][0],
            "width": 700,
            "height": 600,
        }

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a regression plot using the Plotly API.
        """
        fig = self._get_fig(fig)
        x = self.data["X"][:, 0]
        y = self.data["X"][:, 1]
        min_reg_x, max_reg_x = min(x), max(x)
        y0 = self.data["coef"][0]
        slope = self.data["coef"][1]
        if len(self.layout["columns"]) == 2:
            fig = fig.add_trace(
                go.Scatter(x=x, y=y, **self.init_style, name="Scatter Points")
            )
            fig.add_trace(
                go.Scatter(
                    x=[min_reg_x, max_reg_x],
                    y=[y0 + slope * min_reg_x, y0 + slope * max_reg_x],
                    mode="lines",
                    line_shape="linear",
                    name="Regression Line",
                )
            )
            fig.update_layout(**self._update_dict(self.init_layout_style, style_kwargs))
        else:
            raise ValueError("The number of predictors is too big to draw the plot.")
        return fig
