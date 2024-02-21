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

from plotly.graph_objs._figure import Figure
from plotly.figure_factory import create_quiver
import plotly.express as px
import pandas as pd

from verticapy.plotting._plotly.base import PlotlyBase


class PCACirclePlot(PlotlyBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["pca_circle"]:
        return "pca_circle"

    # Styling Methods.

    def _init_style(self) -> None:
        if self.data["explained_variance"][0]:
            dim1 = f"({round(self.data['explained_variance'][0] * 100, 1)}%)"
        else:
            dim1 = ""
        if self.data["explained_variance"][1]:
            dim2 = f"({round(self.data['explained_variance'][1] * 100, 1)}%)"
        else:
            dim2 = ""
        self.init_layout_style = {
            "yaxis_title": f"Dim{self.data['dim'][1]} {dim1}",
            "xaxis_title": f"Dim{self.data['dim'][0]} {dim2}",
            "width": 700,
            "height": 600,
            "legend": dict(orientation="v"),
        }

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a PCA circle plot using the Plotly API.
        """
        fig = self._get_fig(fig)
        fig.add_shape(
            type="circle",
            xref="x",
            yref="y",
            x0=-1,
            y0=-1,
            x1=1,
            y1=1,
            line_color="LightSeaGreen",
        )
        for i in range(len(self.data["x"])):
            fig.add_trace(
                create_quiver(
                    x=[0],
                    y=[0],
                    u=[self.data["x"][i]],
                    v=[self.data["y"][i]],
                    scale=1,
                    arrow_scale=0.1,
                    line=dict(width=1),
                    name=self.layout["columns"][i],
                ).data[0]
            )
        fig.update_layout(**self._update_dict(self.init_layout_style, style_kwargs))
        return fig


class PCAScreePlot(PlotlyBase):
    # Properties

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["pca_scree"]:
        return "pca_scree"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_layout_style = {
            "yaxis_title": "Percentage Explained Variance (%)",
            "xaxis_title": "Dimensions",
            "width": 400,
            "height": 500,
        }
        return None

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a PCA Scree plot using the Plotly API.
        """
        fig_base = self._get_fig(fig)
        fig = px.bar(
            x=self.data["x"],
            y=self.data["y"],
        )
        fig2 = px.line(
            x=self.data["x"],
            y=self.data["y"],
            markers=True,
            color_discrete_sequence=[self.get_colors(idx=1)],
        )
        fig.update_xaxes(type="category")
        params = self._update_dict(self.init_layout_style, style_kwargs)
        fig_base.add_trace(fig.data[0])
        fig_base.update_layout(fig.layout)
        fig_base.add_trace(fig2.data[0])
        fig_base.update_layout(fig2.layout)
        fig_base.update_layout(**params)
        return fig_base


class PCAVarPlot(PlotlyBase):
    # Properties

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["pca_var"]:
        return "pca_var"

    def _init_style(self) -> None:
        self.init_layout_style = {
            "yaxis_title": f"Dim{self.data['dim'][1]} ({round(self.data['explained_variance'][0]*100,2)}%)",
            "xaxis_title": f"Dim{self.data['dim'][0]} ({round(self.data['explained_variance'][1]*100,2)}%)",
        }
        return None

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a PCA Var plot using the Plotly API.
        """
        fig_base = self._get_fig(fig)
        # Get rid of quotes
        cols = [item.replace('"', "") for item in self.layout["columns"]]
        # Create DataFrame
        df = pd.DataFrame({"x": self.data["x"], "y": self.data["y"], "cols": cols})
        # Create the plot
        fig = px.scatter(df, y="y", x="x", color="cols", symbol="cols")
        # Add bold line for x=0
        fig.add_shape(
            type="line", x0=0, y0=-1, x1=0, y1=1, line=dict(color="black", width=2)
        )
        # Add bold line for y=0
        fig.add_shape(
            type="line", x0=-1, y0=0, x1=1, y1=0, line=dict(color="black", width=2)
        )
        for i in range(len(fig.data)):
            fig_base.add_trace(fig.data[i])
        fig_base.update_layout(fig.layout)
        params = self._update_dict(self.init_layout_style, style_kwargs)
        fig_base.update_layout(width=700, height=500)
        fig_base.update_layout(**params)
        return fig_base
