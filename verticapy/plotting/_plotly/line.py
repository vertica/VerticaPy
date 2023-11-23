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
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure

from verticapy.plotting._plotly.base import PlotlyBase


class LinePlot(PlotlyBase):
    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["line"]:
        return "line"

    @property
    def _compute_method(self) -> Literal["line"]:
        return "line"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "xaxis_title": self.layout["order_by"],
            "yaxis_title": self.layout["columns"][0] if "z" in self.data else "values",
            "width": 800,
            "height": 450,
        }

    def _get_kind(self) -> [str]:
        self.init_stack = None
        if self.layout["kind"] == "area":
            return "tozeroy"
        elif self.layout["kind"] in ("area_stacked", "area_percent"):
            self.init_stack = "group"
            return "tonexty"

    def _line_shape(self, step) -> [str]:
        if self.layout["kind"] == "step" or step == True:
            return "hv"
        elif self.layout["kind"] == "spline":
            return "spline"

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        step: bool = False,
        markers: bool = True,
        line_shape: Optional[str] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a time series plot using the plotly API.
        """
        fig_base = self._get_fig(fig)
        marker_colors = self.get_colors()
        if "colors" in style_kwargs:
            marker_colors = (
                style_kwargs["colors"] + marker_colors
                if isinstance(style_kwargs["colors"], list)
                else [style_kwargs["colors"]] + marker_colors
            )
            del style_kwargs["colors"]
        if "z" in self.data:
            data_args = dict(
                data=(
                    np.column_stack((self.data["x"], self.data["Y"], self.data["z"]))
                ),
                columns=["time", self.layout["columns"][0], "color"],
            )
        else:
            data_args = dict(
                data=(np.column_stack((self.data["x"], self.data["Y"]))),
                columns=["time", self.layout["columns"][0]],
            )
        df = pd.DataFrame(**data_args)
        if "z" in self.data:
            for idx, elem in enumerate(df["color"].unique()):
                DF = df[df["color"] == elem]
                fig_base.add_trace(
                    go.Scatter(
                        x=DF["time"],
                        y=DF[self.layout["columns"][0]],
                        name=elem,
                        line_shape=self._line_shape(step),
                        line_color=marker_colors[idx],
                        mode="lines+markers" if markers else "lines",
                        fill=self._get_kind(),
                        stackgroup=self.init_stack,
                    )
                )
        else:
            fig_base.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=df[self.layout["columns"][0]],
                    line_shape=self._line_shape(step),
                    line_color=marker_colors[0],
                    mode="lines+markers" if markers else "lines",
                    fill=self._get_kind(),
                )
            )
        for i in range(len(fig.data) if fig else 0):
            fig_base.add_trace(fig.data[i])
        fig_base.update_layout(**self._update_dict(self.init_style, style_kwargs))
        fig_base.update_layout(fig.layout if fig else [])
        return fig_base


class MultiLinePlot(LinePlot):
    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["line"]:
        return "line"

    @property
    def _compute_method(self) -> Literal["line"]:
        return "line"

    def draw(
        self,
        fig: Optional[Figure] = None,
        step: bool = False,
        line_shape: Optional[str] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a time series plot using the plotly API.
        """
        fig_base = self._get_fig(fig)
        if self.layout["kind"] == "area_percent":
            row_sums = self.data["Y"].sum(axis=1)
            self.data["Y"] = (((self.data["Y"]).T / row_sums) * 100).T
        marker_colors = self.get_colors()
        if "colors" in style_kwargs:
            marker_colors = (
                style_kwargs["colors"] + marker_colors
                if isinstance(style_kwargs["colors"], list)
                else [style_kwargs["colors"]] + marker_colors
            )
            del style_kwargs["colors"]
        for idx, elem in enumerate(self.layout["columns"]):
            fig_base.add_trace(
                go.Scatter(
                    x=self.data["x"],
                    y=self.data["Y"][:, idx],
                    name=elem,
                    line_shape=self._line_shape(step),
                    line_color=marker_colors[idx],
                    fill=self._get_kind(),
                    stackgroup=self.init_stack,
                )
            )
        fig_base.update_layout(**self._update_dict(self.init_style, style_kwargs))
        return fig_base
