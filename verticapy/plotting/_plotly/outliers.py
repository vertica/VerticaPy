"""
(c)  Copyright  [2018-2023]  OpenText  or one of its
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

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure

from verticapy.plotting._plotly.base import PlotlyBase


class OutliersPlot(PlotlyBase):

    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["outliers"]:
        return "outliers"

    @property
    def _compute_method(self) -> Literal["outliers"]:
        return "outliers"

    @property
    def _dimension_bounds(self) -> tuple[int, int]:
        return (1, 2)

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = { }
        return None

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws an outliers contour plot using the Plotly API.
        """
        fig = self._get_fig(fig)
        if len(self.layout["columns"]) == 1:
            X1 = self.data['inliers'][:,0].flatten().tolist()
            X2 = self.data['outliers'][:,0].flatten().tolist()
            cat_X1 = ["inliers"] * len(X1)
            cat_X2 = ["outliers"] * len(X2)
            x_axis=["age"]*(len(X1)+len(X2))
            df = pd.DataFrame({self.layout['columns'][0]: X1 + X2, "category": cat_X1 + cat_X2,"x_axis":x_axis})
            fig_scatter = px.strip(df, x="x_axis", y=self.layout['columns'][0], color="category",stripmode="overlay")
            fig_scatter.update_layout(xaxis={'visible': False})
        elif len(self.layout["columns"]) == 2:
            X1 = self.data['inliers'][:,0].flatten().tolist()
            X2 = self.data['outliers'][:,0].flatten().tolist()
            Y1 = self.data['inliers'][:,1].flatten().tolist()
            Y2 = self.data['outliers'][:,1].flatten().tolist()
            delta_x=self.data['map']['X'][0][1]-self.data['map']['X'][0][0]
            delta_y=self.data['map']['Y'][1][0]-self.data['map']['Y'][0][1]
            cat_X1 = ["inliers"] * len(X1)
            cat_X2 = ["outliers"] * len(X2)
            x_axis=["age"]*(len(X1)+len(X2))
            df = pd.DataFrame({self.layout['columns'][0]: X1, "category": cat_X1,self.layout['columns'][1]:Y1})
            df2 = pd.DataFrame({self.layout['columns'][0]: X2, "category": cat_X2,self.layout['columns'][1]:Y2})
            concatenated_df = pd.concat([df, df2], ignore_index=True)
            fig_scatter = px.scatter(concatenated_df, x="age", y="fare", color="category")
            fig.add_trace(
                go.Contour(
                    z=self.data["map"]["Z"],
                    dx=delta_x,
                    x0=self.data['map']['X'][0][0],
                    dy=delta_y,
                    y0=self.data['map']['Y'][0][0],
                    )
                        )
        for i in range(len(fig_scatter.data)):
            fig.add_trace(fig_scatter.data[i])

        fig.update_layout(width=500,height=500)

        return fig
