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


class SpiderChart(PlotlyBase):
    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["spider"]:
        return "spider"

    @property
    def _compute_method(self) -> Literal["2D"]:
        return "2D"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "width": 500,
            "height": 500,
            "legend": dict(
                title=self.layout["columns"][1]
                if len(self.layout["columns"]) > 1
                else None
            ),
            "title": self.layout["columns"][0],
            "annotations": [
                dict(
                    x=0.5,
                    y=-0.1,
                    xref="paper",
                    yref="paper",
                    text=f"(Method: {self.layout['method_of'].title()})",
                    showarrow=False,
                )
            ],
        }

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a spider plot using the Plotly API.
        """
        fig = self._get_fig(fig)
        for i in range(self.data["X"].shape[1]):
            fig.add_trace(
                go.Scatterpolar(
                    r=self.data["X"][:, i].flatten(),
                    theta=self.layout["x_labels"],
                    fill="toself",
                    name=self.layout["y_labels"][i],
                )
            )
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    nticks=5,
                ),
                angularaxis=dict(type="category"),
            ),
            showlegend=True,
        )
        fig.update_layout(**self._update_dict(self.init_style, style_kwargs))
        return fig
