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

import plotly.express as px
from plotly.graph_objs._figure import Figure

from verticapy.plotting._plotly.base import PlotlyBase


class ElbowCurve(PlotlyBase):

    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["elbow"]:
        return "elbow"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "color": self.get_colors(idx=0),
            "marker": "o",
            "markerfacecolor": "white",
            "markersize": 7,
            "markeredgecolor": "black",
        }
        return None

    # Draw.

    def draw(self, ax: Optional[Axes] = None, **style_kwargs,) -> Axes:
        """
        Draws a Machine Learning Bubble Plot using the Plotly API.
        """
        return self.data,self.layout
