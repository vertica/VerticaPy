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


class RangeCurve(PlotlyBase):
    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["range"]:
        return "range"

    @property
    def _compute_method(self) -> Literal["range"]:
        return "range"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "yaxis_title": self.layout["columns"][0],
            "xaxis_title": self.layout["order_by"],
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
        plot_scatter: bool = True,
        plot_median: bool = True,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a range curve using the Plotly API.
        """
        fig = self._get_fig(fig)
        marker_colors = self.get_colors()
        if "colors" in style_kwargs:
            marker_colors = (
                style_kwargs["colors"] + marker_colors
                if isinstance(style_kwargs["colors"], list)
                else [style_kwargs["colors"]] + marker_colors
            )
            del style_kwargs["colors"]
        for idx, col in enumerate(self.layout["columns"]):
            y_data = self.data["Y"][:, idx * 3 : idx * 3 + 3]
            fig.add_trace(
                go.Scatter(
                    x=np.hstack((self.data["x"], self.data["x"][::-1])),
                    y=np.hstack((y_data[:, 0], y_data[:, 2][::-1])),
                    fill="toself",
                    showlegend=False,
                    name=f"{col}:: Bounds:[{self.data['q'][0]},{self.data['q'][1]}]"
                    if "q" in self.data
                    else col,
                    mode="lines+markers" if not plot_scatter else "markers",
                    marker=dict(color=marker_colors[idx]),
                    opacity=0.5,
                )
            )
            if plot_median:
                fig.add_trace(
                    go.Scatter(
                        x=self.data["x"],
                        y=y_data[:, 1],
                        name=f"{col}: Median",
                        marker=dict(color=marker_colors[idx]),
                    )
                )

        fig.update_layout(**self._update_dict(self.init_style, style_kwargs))
        return fig
