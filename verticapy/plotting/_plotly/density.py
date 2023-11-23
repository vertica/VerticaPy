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


class DensityPlot(PlotlyBase):
    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["density"]:
        return "density"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "width": 700,
            "height": 500,
            "autosize": False,
            "xaxis_title": self._clean_quotes(self.layout["x_label"]),
            "yaxis_title": self.layout["y_label"],
            "xaxis": dict(
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
                zeroline=False,
            ),
            "yaxis": dict(
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
                zeroline=False,
            ),
        }

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a density plot using the Plotly API.
        """
        fig = self._get_fig(fig)
        fig.add_trace(go.Scatter(x=self.data["x"], y=self.data["y"], fill="tozeroy"))
        fig.update_layout(**self._update_dict(self.init_style, style_kwargs))
        return fig


class MultiDensityPlot(DensityPlot):
    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a multi-density plot using the Plotly API.
        """
        fig = self._get_fig(fig)
        for i in range(self.data["X"].shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=self.data["X"][:, i],
                    y=self.data["Y"][:, i],
                    fill="tozeroy",
                    name=str(self.layout["labels"][i]),
                )
            )
        fig.update_layout(**self._update_dict(self.init_style, style_kwargs))
        return fig


class DensityPlot2D(PlotlyBase):
    ...
