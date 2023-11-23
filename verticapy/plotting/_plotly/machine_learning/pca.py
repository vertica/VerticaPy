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
    ...


class PCAVarPlot(PlotlyBase):
    ...
