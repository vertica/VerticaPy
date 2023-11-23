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
import numpy as np

from verticapy.plotting._plotly.base import PlotlyBase


class LOFPlot(PlotlyBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["lof"]:
        return "lof"

    @property
    def _compute_method(self) -> Literal["sample"]:
        return "sample"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (2, 4)

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_layout_style = {
            "yaxis_title": self.layout["columns"][1],
            "xaxis_title": self.layout["columns"][0],
            "width": 700,
            "height": 600,
        }
        self.init_of_scatter_style = {
            "customdata": self.data["X"][:, -1],
            "mode": "markers",
            "opacity": 0.3,
            "marker": dict(
                size=self.data["X"][:, -1],
                sizemode="diameter",
                sizemin=1,
                color="white",
                line=dict(
                    width=2,
                    color="black",
                ),
            ),
            "name": "IOF",
            "hovertemplate": f"{self.layout['columns'][0]}: "
            "%{x} <br>"
            f"{self.layout['columns'][1]}: "
            " %{y} <br>"
            "IOF: %{customdata:.2f} <extra></extra>",
        }
        col_hover = self.layout["columns"][2] if len(self.layout["columns"]) > 2 else ""
        self.init_of_scatter3d_style = {
            "customdata": self.data["X"][:, -1],
            "mode": "markers",
            "opacity": 0.3,
            "marker": dict(
                size=self.data["X"][:, -1],
                sizemode="diameter",
                sizemin=1,
                color="white",
                line=dict(
                    width=2,
                    color="black",
                ),
            ),
            "name": "IOF",
            "hovertemplate": f"{self.layout['columns'][0]}: "
            "%{x} <br>"
            f"{self.layout['columns'][1]}: "
            " %{y} <br>"
            f"{col_hover}: "
            " %{z} <br>"
            "IOF: %{customdata:.2f} <extra></extra>",
        }

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        marker_sizeref: Optional[float] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a local outlier plot using the Plotly API.
        """
        if "colors" in style_kwargs:
            colors = style_kwargs["colors"]
            style_kwargs.pop("colors")
        else:
            colors = None
        X = self.data["X"][:, 0]
        Y = self.data["X"][:, 1]
        Z = self.data["X"][:, 2] if self.data["X"].shape[1] == 4 else []
        fig = self._get_fig(fig)
        if 2 <= len(self.layout["columns"]) <= 3:
            fig.add_trace(
                go.Scatter(
                    x=X,
                    y=Y,
                    mode="markers",
                    name="Scatter Points",
                    hoverinfo="none",
                    marker_color=colors[0] if isinstance(colors, list) else colors,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=X,
                    y=Y,
                    **self.init_of_scatter_style,
                    marker_line_color=colors[1]
                    if isinstance(colors, list) and len(colors) > 1
                    else self.get_colors()[1],
                )
            )
        elif len(self.layout["columns"]) == 4:
            fig.add_trace(
                go.Scatter3d(
                    x=X,
                    y=Y,
                    z=Z,
                    mode="markers",
                    marker_size=3,
                    name="Scatter Points",
                    hoverinfo="none",
                    marker_color=colors[0] if isinstance(colors, list) else colors,
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=X,
                    y=Y,
                    z=Z,
                    **self.init_of_scatter3d_style,
                    marker_line_color=colors[1]
                    if isinstance(colors, list) and len(colors) > 1
                    else self.get_colors()[1],
                )
            )
            self.init_layout_style["scene"] = dict(
                aspectmode="cube",
                xaxis_title=self.layout["columns"][0],
                yaxis_title=self.layout["columns"][1],
                zaxis_title=self.layout["columns"][2],
            )
            self.init_layout_style["height"] = 700
            self.init_layout_style.pop("width")
        else:
            raise Exception(
                "LocalOutlierFactor Plot is available for a maximum of 3 columns."
            )
        if not marker_sizeref:
            marker_sizeref = 2.0 * max(self.data["X"][:, -1]) / (8.0**2)
        fig.update_traces(marker_sizeref=marker_sizeref, selector=dict(type="scatter"))
        fig.update_traces(
            marker_sizeref=marker_sizeref, selector=dict(type="scatter3d")
        )
        fig.update_layout(**self._update_dict(self.init_layout_style, style_kwargs))
        return fig
