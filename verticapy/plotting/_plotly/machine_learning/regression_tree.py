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


class RegressionTreePlot(PlotlyBase):
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

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_layout_style = {
            "yaxis_title": self.layout["columns"][1],
            "xaxis_title": self.layout["columns"][0],
        }
        self.init_style_hover_2d = {
            "hovertemplate": f"{self.layout['columns'][0]}: "
            "%{x} <br>"
            f"{self.layout['columns'][1]}: "
            "%{y} <br>"
        }

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a regression tree plot using the Plotly API.
        """
        X = self.data["X"][self.data["X"][:, 0].argsort()]
        x0 = X[:, 0]
        x1 = X[:, 0]
        y0 = X[:, 2]
        y1 = X[:, 1]
        fig = self._get_fig(fig)
        if len(self.layout["columns"]) == 3:
            fig = fig.add_trace(
                go.Scatter(
                    x=x0,
                    y=y0,
                    name="Observations",
                    mode="markers",
                    **self.init_style_hover_2d,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x1,
                    y=y1,
                    mode="lines",
                    line_shape="hv",
                    name="Prediction",
                    **self.init_style_hover_2d,
                )
            )
            fig.update_layout(**self._update_dict(self.init_layout_style, style_kwargs))
        else:
            raise ValueError("The number of predictors is too big to draw the plot.")
        return fig
