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

import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure

from verticapy.plotting._plotly.base import PlotlyBase


class BarChart(PlotlyBase):
    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["bar"]:
        return "bar"

    @property
    def _compute_method(self) -> Literal["1D"]:
        return "1D"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_trace_style = {"marker_color": self.get_colors(idx=0)}
        self.init_layout_style = {
            "yaxis_title": self.layout["method"],
            "xaxis_title": self.layout["column"],
            "width": 100 + 100 * len(self.layout["labels"]),
            "height": 500,
        }

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a bar chart using the Plotly API.
        """
        fig_base = self._get_fig(fig)
        fig = px.bar(x=self.layout["labels"], y=self.data["y"])
        if self.data["is_categorical"]:
            fig.update_xaxes(type="category")
        params = self._update_dict(self.init_layout_style, style_kwargs)
        fig.update_layout(**params)
        fig.update_traces(**self.init_trace_style)
        fig_base.add_trace(fig.data[0])
        fig_base.update_layout(fig.layout)
        return fig_base


class BarChart2D(PlotlyBase):
    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["bar"]:
        return "bar"

    @property
    def _compute_method(self) -> Literal["2D"]:
        return "2D"

    # Styling Methods.
    def _init_style(self) -> None:
        self.init_layout_style = {
            "yaxis_title": self.layout["method"],
            "legend_title_text": self.layout["columns"][1],
            "xaxis_title": self.layout["columns"][0],
            "width": (150 + 40 * len(self.layout["x_labels"]))
            * 0.8
            * len(self.layout["y_labels"]),
            "height": 500,
        }

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a 2D BarChart using the Matplotlib API.
        """
        fig_base = self._get_fig(fig)
        for i in range(len(self.layout["y_labels"])):
            fig = go.Bar(
                name=self.layout["y_labels"][i],
                x=self.layout["x_labels"],
                y=self.data["X"][:, i],
            )
            fig_base.add_trace(fig)
        params = self._update_dict(self.init_layout_style, style_kwargs)
        fig_base.update_layout(**params)
        if self.layout["kind"] == "stacked":
            fig_base.update_layout(barmode="stack")
        return fig_base
