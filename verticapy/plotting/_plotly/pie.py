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
        }
        self.init_layout_style = {
            "title_text": self.layout["column"],
            "title_x": 0.5,
            "title_xanchor": "center",
        }

    # Draw

    def draw(
        self,
        fig: Optional[Figure] = None,
        exploded: bool = False,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a pie chart using the Plotly API.
        """
        fig_base = self._get_fig(fig)
        user_colors = style_kwargs.get("color", style_kwargs.get("colors"))
        if isinstance(user_colors, str):
            user_colors = [user_colors]
        color_list = (
            user_colors + self.get_colors() if user_colors else self.get_colors()
        )
        if "colors" in style_kwargs:
            del style_kwargs["colors"]
        labels = self.layout["labels"]
        if self.layout["kind"] == "donut":
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
                    marker=dict(colors=color_list),
                    **param,
                    sort=False,
                )
            ]
        )
        fig_base.add_trace(fig.data[0])
        fig_base.update_layout(**self.init_layout_style, **style_kwargs)
        return fig_base


class NestedPieChart(PlotlyBase):
    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["pie"]:
        return "pie"

    @property
    def _compute_method(self) -> Literal["rollup"]:
        return "rollup"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_trace_style = {
            "hovertemplate": "<b>Fraction: %{percentEntry:.2f} </b> <extra></extra>",
        }

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a sunburst/nested pie chart using the plotly API.
        """
        user_colors = style_kwargs.get("color", style_kwargs.get("colors"))
        if isinstance(user_colors, str):
            user_colors = [user_colors]
        color_list = (
            user_colors + self.get_colors() if user_colors else self.get_colors()
        )
        if "colors" in style_kwargs:
            del style_kwargs["colors"]
        ids, labels, parents, values = self._convert_labels_and_get_counts(
            self.data["groups"][0]
        )
        trace = go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            outsidetextfont={"size": 20},
            marker={"line": {"width": 2}, "colors": color_list},
            **self.init_trace_style,
        )
        layout = go.Layout(margin=go.layout.Margin(t=0, l=0, r=0, b=0))
        figure = {"data": [trace], "layout": layout}
        fig = go.Figure(figure)
        fig.update_layout(**style_kwargs)
        return fig
