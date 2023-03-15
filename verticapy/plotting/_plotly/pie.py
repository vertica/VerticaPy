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

import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure

from verticapy.plotting._plotly.base import PlotlyBase


class PieChart(PlotlyBase):

    # Properties

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["pie"]:
        return "pie"

    @property
    def _compute_method(self) -> Literal["1D"]:
        return "1D"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_trace_style = {
            "hovertemplate": "%{label} <extra></extra>",
            "marker_colors": PlotlyBase.get_colors(),
        }
        self.init_layout_style = {
            "title_text": self.layout["column"][1:-1],
            "title_x": 0.5,
            "title_xanchor": "center",
        }

        return None

    # Draw

    def draw(
        self,
        pie_type: Literal["auto", "donut"] = "auto",
        exploded: bool = False,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a pie chart using the Plotly API.
        """
        labels = self.layout["labels"]
        labels = ["Null" if elm is None else elm for elm in labels]
        if pie_type == "donut":
            hole_fraction = 0.2
        else:
            hole_fraction = 0
        if exploded:
            exploded_parameters = [0] * len(self.data["y"])
            exploded_parameters[self.data["y"].index(max(self.data["y"]))] = 0.2
        else:
            exploded_parameters = []
        param = {
            "hole": hole_fraction,
            "pull": exploded_parameters,
        }
        param = self._update_dict(self.init_trace_style, param)
        labels, values = zip(*sorted(zip(labels, self.data["y"]), key=lambda t: t[0]))
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    **self._update_dict(param, style_kwargs),
                    sort=False,
                )
            ]
        )
        fig.update_layout(**self.init_layout_style)
        fig.show()
        return fig


class NestedPieChart(PlotlyBase):
    ...
