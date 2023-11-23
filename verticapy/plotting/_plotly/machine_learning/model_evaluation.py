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


class ROCCurve(PlotlyBase):
    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["roc"]:
        return "roc"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_layout_style = {
            "title": "ROC Curve",
            "yaxis_title": self.layout["y_label"],
            "xaxis_title": self.layout["x_label"],
            "width": 700,
            "height": 600,
            "annotations": [
                dict(
                    x=0.5,
                    y=-0.25,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    text=f"AUC: {self.data['auc']}",
                    font=dict(size=14),
                )
            ],
            "showlegend": False,
        }
        return None

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        line_shape: Literal["linear", "spline", "vhv", "hv", "vh", "hvh"] = "hv",
        **style_kwargs,
    ) -> Figure:
        """
        Draws a machine learning ROC curve using the Plotly API.
        """
        fig = self._get_fig(fig)
        fig.add_trace(
            go.Scatter(
                x=self.data["x"],
                y=self.data["y"],
                mode="lines",
                line_shape=line_shape,
                fill="tozeroy",
            )
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(width=2, dash="dot"))
        )
        fig.update_layout(**self._update_dict(self.init_layout_style, style_kwargs))
        return fig


class CutoffCurve(PlotlyBase):
    # Properties.

    @property
    def _kind(self) -> Literal["cutoff"]:
        return "cutoff"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_layout_style = {
            "title": "Cutoff Curve",
            "yaxis_title": "Values",
            "xaxis_title": "Decision Boundary",
            "width": 800,
            "height": 450,
            "showlegend": True,
        }
        return None

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        line_shape: Literal["linear", "spline", "vhv", "hv", "vh", "hvh"] = "hv",
        **style_kwargs,
    ) -> Figure:
        """
        Draws a machine cutoff curve using the Plotly API.
        """
        fig = self._get_fig(fig)
        fig.add_trace(
            go.Scatter(
                x=self.data["x"],
                y=self.data["y"],
                mode="lines+markers",
                line_shape=line_shape,
                name="Specificity",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.data["x"],
                y=self.data["z"],
                mode="lines+markers",
                line_shape=line_shape,
                name="Sensitivity",
            )
        )
        fig.update_layout(**self._update_dict(self.init_layout_style, style_kwargs))
        return fig


class PRCCurve(ROCCurve):
    # Properties.

    @property
    def _kind(self) -> Literal["prc"]:
        return "prc"

    def _init_style(self) -> None:
        self.init_layout_style = {
            "title": self.layout["title"],
            "yaxis_title": self.layout["y_label"],
            "xaxis_title": self.layout["x_label"],
            "width": 800,
            "height": 600,
            "showlegend": False,
            "annotations": [
                dict(
                    x=0.5,
                    y=0.05,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    text=f"AUC: {self.data['auc']:.4f}",
                    font=dict(size=14),
                )
            ],
        }
        return None

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        line_shape: Literal["linear", "spline", "vhv", "hv", "vh", "hvh"] = "hv",
        **style_kwargs,
    ) -> Figure:
        """
        Draws a machine learning PRC curve using the Plotly API.
        """
        fig = self._get_fig(fig)
        fig.add_trace(
            go.Scatter(
                x=self.data["x"],
                y=self.data["y"],
                mode="lines",
                line_shape=line_shape,
                name="Specificity",
                fill="tozeroy",
            )
        )
        fig.update_layout(**self._update_dict(self.init_layout_style, style_kwargs))
        return fig


class LiftChart(ROCCurve):
    # Properties.

    @property
    def _kind(self) -> Literal["lift"]:
        return "lift"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_layout_style = {
            "title": self.layout["title"],
            "yaxis_title": "Values",
            "xaxis_title": self.layout["x_label"],
            "width": 800,
            "height": 400,
            "showlegend": True,
        }
        return None

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a machine cutoff curve using the Plotly API.
        """
        fig = self._get_fig(fig)
        fig.add_trace(
            go.Scatter(
                x=self.data["x"],
                y=self.data["y"],
                mode="lines+markers",
                line_shape="linear",
                fill="tozeroy",
                name="Cumulative Capture Rate",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.data["x"],
                y=self.data["z"],
                mode="lines+markers",
                line_shape="linear",
                fill="tonexty",
                name="Cumulative Lift",
            )
        )
        fig.update_layout(**self._update_dict(self.init_layout_style, style_kwargs))
        return fig
