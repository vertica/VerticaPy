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
from plotly.graph_objs._figure import Figure

from verticapy.plotting._plotly.base import PlotlyBase


class ElbowCurve(PlotlyBase):
    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["elbow"]:
        return "elbow"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "mode": "markers+lines",
            "marker_line_width": 2,
            "marker_color": "white",
            "marker_size": 10,
            "marker_line_color": "black",
        }
        self.init_layout_style = {
            "yaxis_title": self.layout["y_label"],
            "xaxis_title": self.layout["x_label"],
            "width": 650,
            "height": 650,
        }

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a machine learning bubble plot using the Plotly API.
        """
        fig_base = self._get_fig(fig)
        fig = px.line(x=self.data["x"], y=self.data["y"], markers=True)
        fig_base.add_trace(fig.data[0])
        fig_base.update_traces(**self.init_style)
        fig_base.update_layout(
            **self._update_dict(self.init_layout_style, style_kwargs)
        )
        return fig_base
