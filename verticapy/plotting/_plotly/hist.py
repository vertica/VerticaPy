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
import plotly.graph_objects as go

from verticapy.plotting._plotly.base import PlotlyBase


class Histogram(PlotlyBase):
    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["hist"]:
        return "hist"

    @property
    def _compute_method(self) -> Literal["hist"]:
        return "hist"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {"width": 800, "height": 450}

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws an histogram using the Plotly API.
        """
        fig = self._get_fig(fig)
        key = "categories" if self.layout["has_category"] else "columns"
        for i in range(len(self.layout[key])):
            fig.add_trace(
                go.Bar(
                    name=self.layout["columns"][i],
                    x=self.data[self.layout[key][i]]["x"],
                    y=self.data[self.layout[key][i]]["y"],
                    width=self.data["width"],
                    offset=0,
                    opacity=0.8 if len(self.layout[key]) > 1 else 1,
                )
            )
        fig.update_layout(yaxis_title=self.layout["method_of"])
        if len(self.layout["columns"]) == 1:
            fig.update_layout(xaxis_title=self.layout["columns"][0])
        else:
            title = self.layout["by"]
        fig.update_layout(**self._update_dict(self.init_style, style_kwargs))
        return fig
