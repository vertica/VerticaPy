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

from plotly.graph_objs._figure import Figure

from verticapy.plotting._plotly.base import PlotlyBase


class ACFPlot(PlotlyBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["acf"]:
        return "acf"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "height": 500,
            "width": 900,
            "xaxis_title": "Lag",
            "yaxis_title": "Value",
            "showlegend": False,
            "xaxis": dict(
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
                zeroline=True,
                dtick=1,
                range=[-0.5, self.data["x"][-1] + 0.5],
            ),
            "yaxis": dict(
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
                zeroline=True,
                tickmode="linear",
                tick0=-0.2,
                dtick=0.2,
            ),
        }
        self.init_confidence_style = {
            "mode": "lines",
            "marker_color": self.get_colors()[0],
        }
        self.init_scatter_style = {
            "marker_color": "orange",
            "marker_size": 12,
            "hovertemplate": "ACF <br><b>lag</b>: %{x}<br><b>value</b>: %{y:0.3f}<br><extra></extra>",
        }

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws an ACF time series plot using the Plotly API.
        """
        X = self.data["x"]
        Y = self.data["y"]
        Z = self.data["z"]
        fig = self._get_fig(fig)
        if "colors" in style_kwargs:
            self.init_confidence_style["marker_color"] = (
                style_kwargs["colors"]
                if isinstance(style_kwargs["colors"], str)
                else style_kwargs["colors"][0]
            )
            style_kwargs.pop("colors")
        fig.add_scatter(
            x=X,
            y=Z,
            **self.init_confidence_style,
            hoverinfo="none",
        )
        fig.add_scatter(
            x=X,
            y=-Z,
            **self.init_confidence_style,
            fill="tonexty",
            hovertemplate="Confidence <br><b>lag</b>: %{x}<br><b>value</b>: %{customdata:.3f}<br><extra></extra>",
            customdata=np.abs(-Z),
        )
        if self.layout["kind"] == "line":
            scatter_mode = "lines+markers"
        else:
            scatter_mode = "markers"
            for i in range(len(Y)):
                fig.add_shape(
                    type="line",
                    x0=X[i],
                    x1=X[i],
                    y0=0,
                    y1=Y[i],
                    line=dict(color="black", width=2),
                )
        fig.add_scatter(
            x=X,
            y=Y,
            **self.init_scatter_style,
            mode=scatter_mode,
        )
        fig.update_layout(**self._update_dict(self.init_style, style_kwargs))
        return fig


class ACFPACFPlot(ACFPlot):
    ...
