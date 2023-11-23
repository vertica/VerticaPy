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
        self.init_layout_style = {
            "yaxis_title": f"P({self.layout['columns'][-1]} = 1)",
            "xaxis_title": self.layout["columns"][0],
            "width": 700,
            "height": 600,
        }
        self.init_style_hover_2d = {
            "hovertemplate": f"{self.layout['columns'][0]}: "
            "%{x} <br>"
            f"P({self.layout['columns'][-1]} = 1): "
            " %{y} <br>"
        }
        self.init_style_hover_3d = {
            "hovertemplate": f"{self.layout['columns'][0]}: "
            "%{x} <br>"
            f"{self.layout['columns'][1]}: "
            " %{y} <br>"
            f"P({self.layout['columns'][-1]} = 1): "
            " %{y} <br>"
        }

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a logistic regression plot using the Plotly API.
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
        y_logit = logit(self.data["coef"][0] + self.data["coef"][1] * x_logit)
        if len(self.layout["columns"]) == 2:
            fig.add_trace(
                go.Scatter(
                    x=x0,
                    y=logit(y0 + slope * x0),
                    name="-1",
                    mode="markers",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x1,
                    y=logit(y0 + slope * x1),
                    name="+1",
                    mode="markers",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_logit,
                    y=y_logit,
                    mode="lines",
                    line_shape="linear",
                    name="Logit",
                    opacity=0.5,
                )
            )
            fig.update_traces(**self.init_style_hover_2d)
        elif len(self.layout["columns"]) == 3:
            y = self.data["X"][:, 1]
            y0, y1 = y[z == 0], y[z == 1]
            min_logit_y, max_logit_y = (
                min(self.data["X"][:, 1]),
                max(self.data["X"][:, 1]),
            )
            step_y = (max_logit_y - min_logit_y) / 40.0
            X_logit = (
                np.arange(min_logit_x - 5 * step_x, max_logit_x + 5 * step_x, step_x)
                if (step_x > 0)
                else np.array([max_logit_x])
            )
            Y_logit = (
                np.arange(min_logit_y - 5 * step_y, max_logit_y + 5 * step_y, step_y)
                if (step_y > 0)
                else np.array([max_logit_y])
            )
            X_logit, Y_logit = np.meshgrid(X_logit, Y_logit)
            Z_logit = 1 / (
                1
                + np.exp(
                    -(
                        self.data["coef"][0]
                        + self.data["coef"][1] * X_logit
                        + self.data["coef"][2] * Y_logit
                    )
                )
            )
            X_logit, Y_logit, Z_logit

            for i, x, y in [
                (0, x0, y0),
                (1, x1, y1),
            ]:
                fig.add_trace(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=logit(
                            self.data["coef"][0]
                            + self.data["coef"][1] * x
                            + self.data["coef"][2] * y
                        ),
                        name=str(i),
                        mode="markers",
                    )
                )
            fig = fig.add_trace(
                go.Surface(
                    z=Z_logit,
                )
            )
            self.init_layout_style["scene"] = dict(
                aspectmode="cube",
                xaxis_title=self.layout["columns"][0],
                yaxis_title=self.layout["columns"][1],
                zaxis_title=f"P({self.layout['columns'][-1]} = 1)",
            )
        else:
            raise ValueError("The number of predictors is too big.")
        fig.update_layout(**self._update_dict(self.init_layout_style, style_kwargs))
        return fig
