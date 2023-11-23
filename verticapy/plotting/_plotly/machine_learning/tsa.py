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

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure

from verticapy.plotting._plotly.base import PlotlyBase


class TSPlot(PlotlyBase):
    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["tsa"]:
        return "tsa"

    @property
    def _compute_method(self) -> Literal["tsa"]:
        return "tsa"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "xaxis_title": self.layout["order_by"],
            "yaxis_title": self.layout["columns"],
            "width": 800,
            "height": 450,
        }

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a time series plot using the plotly API.
        """
        fig_base = self._get_fig(fig)
        marker_colors = self.get_colors()
        if "colors" in style_kwargs:
            if isinstance(style_kwargs["colors"], list):
                marker_colors = style_kwargs["colors"] + marker_colors
            else:
                marker_colors = [style_kwargs["colors"]] + marker_colors
            del style_kwargs["colors"]
        # True Values
        data_args = dict(
            data=(np.column_stack((self.data["x"], self.data["y"]))),
            columns=["time", self.layout["columns"]],
        )
        df = pd.DataFrame(**data_args)
        fig_base.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[self.layout["columns"]],
                line_shape="spline",
                line_color=marker_colors[0],
                mode="lines",
                name=self.layout["columns"],
            )
        )
        # One step ahead forecast
        if not (self.layout["is_forecast"]):
            data_args = dict(
                data=(
                    np.column_stack((self.data["x_pred_one"], self.data["y_pred_one"]))
                ),
                columns=["time", self.layout["columns"]],
            )
            df = pd.DataFrame(**data_args)
            fig_base.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=df[self.layout["columns"]],
                    line_shape="spline",
                    line_color=marker_colors[1],
                    mode="lines",
                    name="one-sted-ahead-forecast",
                )
            )
        # Forecast
        data_args = dict(
            data=(np.column_stack((self.data["x_pred"], self.data["y_pred"]))),
            columns=["time", self.layout["columns"]],
        )
        df = pd.DataFrame(**data_args)
        fig_base.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[self.layout["columns"]],
                line_shape="spline",
                line_color=marker_colors[2],
                mode="lines",
                name="forecast",
            )
        )
        # STD Error
        if self.layout["has_se"]:
            fig_base.add_trace(
                go.Scatter(
                    x=np.hstack((self.data["se_x"], self.data["se_x"][::-1])),
                    y=np.hstack((self.data["se_low"], self.data["se_high"][::-1])),
                    fill="toself",
                    name="95% confidence interval",
                    mode="lines",
                    marker=dict(color=marker_colors[3]),
                    opacity=0.5,
                )
            )
        # Final
        for i in range(len(fig.data) if fig else 0):
            fig_base.add_trace(fig.data[i])
        fig_base.update_layout(**self._update_dict(self.init_style, style_kwargs))
        fig_base.update_layout(fig.layout if fig else [])
        return fig_base
