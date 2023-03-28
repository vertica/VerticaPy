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
import plotly.express as px
from plotly.graph_objs._figure import Figure

from verticapy.plotting._plotly.base import PlotlyBase

class HeatMap(PlotlyBase):

    # Properties.

    @property
    def _category(self) -> Literal["map"]:
        return "map"

    @property
    def _kind(self) -> Literal["heatmap"]:
        return "heatmap"

    @property
    def _compute_method(self) -> Literal["2D"]:
        return "2D"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {"width":800,"height":600}
        return None

    # Draw.

    def draw(
        self,
        colorbar: str = "",
        extent: Optional[list] = None,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a heatmap using the Matplotlib API.
        """
        print(self.data)
        print(self.layout)
        fig = px.imshow(np.transpose((self.data['X'])),
                        labels=dict(x=self.layout['columns'][0], y=self.layout['columns'][1]),
                        x=self.layout['x_labels'],
                        y=list(self.layout['y_labels'][::-1]),
                        aspect="auto"
                    )
        fig.update_xaxes(type='category')
        fig.update_yaxes(type='category')
        fig.update_layout(**self._update_dict(self.init_style,style_kwargs))
        return fig
