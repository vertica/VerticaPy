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

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.graph_objs._figure import Figure

from verticapy.plotting._plotly.base import PlotlyBase


class ChampionChallengerPlot(PlotlyBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["champion"]:
        return "champion"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_layout_style = {
            "legend": dict(orientation="h", yanchor="bottom", xanchor="center", x=0.5),
            "width": 800,
            "height": 500,
        }
        self.label_layout = {
            "xref": "x",
            "yref": "y",
            "showarrow": False,
            "font": dict(
                # family="Courier New, monospace",
                size=16,
                color="black",
            ),
            "align": "center",
            "bordercolor": "#c7c7c7",
            "borderwidth": 2,
            "borderpad": 4,
            "opacity": 0.8,
        }
        self.hover_style = {
            "hovertemplate": "<b>%{customdata}</b><br>"
            f"{self.layout['x_label']}: "
            "%{x}<br> "
            f"{self.layout['x_label']}:"
            " %{y}<br>"
            f"STD:"
            " %{marker.size:.3f} <extra></extra>"
        }

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a Machine Learning Bubble Plot using the Plotly API.
        """
        fig = self._get_fig(fig)
        n = len(self.data["x"])
        df = pd.DataFrame(self.data)
        df.loc[df["s"] == 0, "s"] = 1e-8
        fig_scatter = px.scatter(df, x="x", y="y", size="s", color="c", custom_data="c")
        fig_scatter.update_traces(
            marker=dict(sizemin=5),
            **self.hover_style,
        )
        for i in range(len(fig_scatter.data)):
            fig.add_trace(fig_scatter.data[i])

        x = df["x"]
        y = df["y"]
        if self.layout["reverse"][0]:
            x_lim = (
                max(x) + 0.1 * (1 + max(x) - min(x)),
                min(x) - 0.1 - 0.1 * (1 + max(x) - min(x)),
            )
        else:
            x_lim(min(x), max(x))
        if self.layout["reverse"][1]:
            y_lim = (
                max(y) + 0.1 * (1 + max(y) - min(y)),
                min(y) - 0.1 * (1 + max(y) - min(y)),
            )
        else:
            y_lim(min(y), max(y))
        fig.update_layout(
            xaxis=dict(
                tickmode="linear",
                range=[x_lim[0], x_lim[1]],
                linecolor="black",
                anchor="free",
                position=0.5,
                zeroline=False,
            ),
            yaxis=dict(
                tickmode="linear",
                range=[y_lim[0], y_lim[1]],
                linecolor="black",
                anchor="free",
                position=0.5,
                dtick=0.1,
                zeroline=False,
            ),
        )
        fig.add_annotation(
            x=(x_lim[0] + x_lim[1]) / 2,
            y=y_lim[1],
            text=self.layout["y_label"],
            showarrow=False,
            xshift=10,
            yshift=-30,
            textangle=90,
        )
        fig.add_annotation(
            x=x_lim[0],
            y=(y_lim[0] + y_lim[1]) / 2,
            text=self.layout["x_label"],
            showarrow=False,
            xshift=30,
            yshift=10,
            xanchor="left",
        )
        fig.add_annotation(
            x=x_lim[0],
            y=y_lim[1],
            text="Efficient",
            textangle=-45,
            bgcolor="#32a8a2",
            **self.label_layout,
        )
        fig.add_annotation(
            x=x_lim[1],
            y=y_lim[1],
            text="Performant & Efficient",
            textangle=45,
            bgcolor="#58ed73",
            **self.label_layout,
        )
        fig.add_annotation(
            x=x_lim[1],
            y=y_lim[0],
            text="Performant",
            textangle=-45 - 180,
            bgcolor="#de52d9",
            **self.label_layout,
        )
        fig.add_annotation(
            x=x_lim[0],
            y=y_lim[0],
            text="Modest",
            textangle=45 - 180,
            bgcolor="#e6c77e",
            **self.label_layout,
        )
        fig.update_layout(**self._update_dict(self.init_layout_style, style_kwargs))
        return fig
