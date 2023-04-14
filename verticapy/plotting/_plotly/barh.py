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

import plotly.express as px
from plotly.graph_objs._figure import Figure

from verticapy.plotting._plotly.base import PlotlyBase


class HorizontalBarChart(PlotlyBase):

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
            "xaxis_title": self.layout["method"],
            "yaxis_title": self.layout["column"],
            # "width": 500 ,
            "height": 100 * len(self.layout["labels"]),
        }
        return None

    # Draw.

    def draw(self, fig: Optional[Figure] = None, **style_kwargs,) -> Figure:
        """
        Draws a horizontal bar chart using the Plotly API.
        """
        fig_base = self._get_fig(fig)
        fig = px.bar(y=self.layout["labels"], x=self.data["y"], orientation="h")
        if self.data["is_categorical"]:
            fig.update_yaxes(type="category")
        params = self._update_dict(self.init_layout_style, style_kwargs)
        fig.update_layout(**params)
        fig.update_traces(**self.init_trace_style)
        fig_base.add_trace(fig.data[0])
        fig_base.update_layout(fig.layout)
        return fig_base


class HorizontalBarChart2D(PlotlyBase):

    ...
