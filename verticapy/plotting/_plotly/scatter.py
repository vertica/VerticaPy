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
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure

import verticapy._config.config as conf

from verticapy.plotting._plotly.base import PlotlyBase


class ScatterMatrix(PlotlyBase):
    ...


class ScatterPlot(PlotlyBase):

    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["scatter"]:
        return "scatter"

    @property
    def _compute_method(self) -> Literal["sample"]:
        return "sample"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "width":700, "height":500, "autosize": True,
            "xaxis_title": self.layout["columns"][0][1:-1],
            "yaxis_title": self.layout["columns"][1][1:-1],
            "xaxis": dict(showline=True, linewidth=1, linecolor='black', mirror=True,zeroline= False),
            "yaxis": dict(showline=True, linewidth=1, linecolor='black', mirror=True,zeroline= False)
            }
        return None

    # Draw.

    def draw(
        self,
        catcol: str = "",
        **style_kwargs,
    ) -> None:
        """
        Draws a scatter plot using the Plotly API.
        """ 
        column_names=self._format_col_names(self.layout['columns'])
        if self.layout['c']:
            df = pd.DataFrame(data = np.column_stack((self.data['X'], self.data['c'])), 
                  columns = column_names+[self.layout['c'][1:-1]])
            fig=px.scatter(df,x=column_names[0],y=column_names[1],color=self.layout['c'][1:-1])
        else:
            df = pd.DataFrame(data = self.data['X'], columns = column_names)
            fig=px.scatter(df,x=column_names[0],y=column_names[1])
        fig.update_layout(**self.init_style)
        return self.data
