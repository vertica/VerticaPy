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
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs._figure import Figure

from verticapy.plotting._plotly.base import PlotlyBase


class SVMClassifierPlot(PlotlyBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["svm"]:
        return "svm"

    @property
    def _compute_method(self) -> Literal["sample"]:
        return "sample"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (2, 4)

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_layout_style_1d = {
            "xaxis": dict(visible=False),
            "width": 600,
            "height": 400,
        }
        self.init_layout_style = {
            "yaxis_title": self.layout["columns"][1],
            "xaxis_title": self.layout["columns"][0],
            "width": 900,
            "height": 500,
        }
        self.init_layout_style_3d = {
            "scene": dict(
                xaxis_title=self.layout["columns"][0],
                yaxis_title=self.layout["columns"][1],
                zaxis_title=self.layout["columns"][2]
                if len(self.layout["columns"]) >= 3
                else None,
            ),
            "scene_aspectmode": "cube",
            "height": 700,
            "autosize": False,
        }
        self.hline_style = {
            "y": -self.data["coef"][0] / self.data["coef"][1],
            "line_width": 2,
            "line_dash": "dash",
            "line_color": "green",
        }
        self.hover_style_3d = {
            "mode": "markers",
            "hovertemplate": f"{self.layout['columns'][0]}: "
            "%{x} <br> "
            f"{self.layout['columns'][1]}:"
            " %{y} <br>"
            f"{self.layout['columns'][2]}:"
            " %{z} <extra></extra>"
            if len(self.layout["columns"]) >= 3
            else None,
        }
        self.hover_style_2d = {
            "mode": "markers",
            "hovertemplate": f"{self.layout['columns'][0]}: "
            "%{x} <br> "
            f"{self.layout['columns'][1]}:"
            " %{y} <extra></extra>",
        }

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a SVM Classifier plot using the Plotly API.
        """
        marker_colors = self.get_colors()
        if "colors" in style_kwargs:
            marker_colors = (
                style_kwargs["colors"] + marker_colors
                if isinstance(style_kwargs["colors"], list)
                else [style_kwargs["colors"]] + marker_colors
            )
            del style_kwargs["colors"]
        fig = self._get_fig(fig)
        x, w = self.data["X"][:, 0], self.data["X"][:, -1]
        x0, x1 = x[w == 0], x[w == 1]
        if len(self.layout["columns"]) == 2:
            x_axis = ["X"] * (len(x0) + len(x1))
            df = pd.DataFrame(
                {
                    self.layout["columns"][0]: self.data["X"][:, 0],
                    "category": self.data["X"][:, 1].astype(int),
                    "x_axis": x_axis,
                }
            )
            fig = px.strip(
                df,
                x="x_axis",
                y=self.layout["columns"][0],
                color="category",
                stripmode="overlay",
                hover_data={
                    self.layout["columns"][0]: True,
                    "category": False,
                    "x_axis": False,
                },
            )
            fig.add_hline(**self.hline_style)
            fig.update_layout(
                **self._update_dict(self.init_layout_style_1d, style_kwargs)
            )
        else:
            y = self.data["X"][:, 1]
            y0, y1 = y[w == 0], y[w == 1]
            min_svm_x, max_svm_x = np.nanmin(x), np.nanmax(x)
            if len(self.layout["columns"]) == 3:
                min_svm_x, max_svm_x = np.nanmin(x), np.nanmax(x)
                x_svm = [min_svm_x, max_svm_x]
                y_svm = [
                    -(self.data["coef"][0] + self.data["coef"][1] * x)
                    / self.data["coef"][2]
                    for x in x_svm
                ]
                fig.add_trace(
                    go.Scatter(
                        name="0",
                        x=x0,
                        y=y0,
                        marker=dict(color=marker_colors[0]),
                        **self.hover_style_2d,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        name="1",
                        x=x1,
                        y=y1,
                        marker=dict(color=marker_colors[1]),
                        **self.hover_style_2d,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        name="SVM",
                        x=x_svm,
                        y=y_svm,
                        mode="lines",
                    )
                )
                fig.update_layout(
                    **self._update_dict(self.init_layout_style, style_kwargs)
                )
            elif len(self.layout["columns"]) == 4:
                z = self.data["X"][:, 2]
                z0, z1 = z[w == 0], z[w == 1]
                min_svm_y, max_svm_y = np.nanmin(y), np.nanmax(y)
                num = 100
                x_range = np.linspace(min_svm_x, max_svm_x, num)
                y_range = np.linspace(min_svm_y, max_svm_y, num)
                X_svm, Y_svm = np.meshgrid(x_range, y_range)
                Z_svm = (
                    -(
                        self.data["coef"][0]
                        + self.data["coef"][1] * X_svm
                        + self.data["coef"][2] * Y_svm
                    )
                    / self.data["coef"][3]
                )
                color_values = np.full((Z_svm.shape[0], Z_svm.shape[1]), "blue")
                fig.add_trace(
                    go.Surface(
                        z=Z_svm,
                        x=X_svm,
                        y=Y_svm,
                        showscale=False,
                        surfacecolor=color_values,
                        opacity=0.4,
                    )
                )
                fig.add_trace(
                    go.Scatter3d(
                        name="0",
                        x=x0,
                        y=y0,
                        z=z0,
                        marker=dict(color=marker_colors[0]),
                        **self.hover_style_3d,
                    )
                )
                fig.add_trace(
                    go.Scatter3d(
                        name="1",
                        x=x1,
                        y=y1,
                        z=z1,
                        marker=dict(color=marker_colors[1]),
                        **self.hover_style_3d,
                    )
                )
                fig.update_layout(**self.init_layout_style_3d)
            else:
                raise ValueError("The number of predictors is too big.")
        return fig
