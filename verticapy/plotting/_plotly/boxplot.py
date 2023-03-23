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
from typing import Literal

import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure

from verticapy.plotting._plotly.base import PlotlyBase


class BoxPlot(PlotlyBase):

    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["box"]:
        return "bar"

    @property
    def _compute_method(self) -> Literal["describe"]:
        return "describe"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_layout_style = {}
        return None

    # Draw.

    def draw(self, **style_kwargs) -> Figure:
        """
        Draws a boxplot using the Plotly API.
        """
        fig = go.Figure()
        fig.add_trace(go.Box(
            x=self.data['fliers'], 
            boxpoints='outliers',
            hovertemplate ='%{x}',
                        ),**style_kwargs)
        fig.update_traces(q1=self.data['X'][1], 
                        median=self.data['X'][2],
                        q3=self.data['X'][3], 
                        lowerfence=self.data['X'][0],
                        upperfence=self.data['X'][4],
                        )
        fig.update_layout(
            yaxis = dict(
                showticklabels=False,
                title=self.layout['labels'][0][1:-1]
            )
        )
        fig.update_layout(hovermode='y')
        return fig


class BarChart2D(PlotlyBase):

    ...
