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

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure

from verticapy.plotting._plotly.base import PlotlyBase


class StepwisePlot(PlotlyBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["stepwise"]:
        return "stepwise"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_layout_style = {
            "yaxis_title": self.layout["y_label"],
            "xaxis_title": self.layout["x_label"],
            "title": f"Direction : {self.layout['direction']}",
            "xaxis": dict(tickmode="linear", tick0=1, dtick=1),
        }
        self.init_start_style = {
            "mode": "markers",
            "marker": dict(size=60, color="blue"),
            "opacity": 0.4,
        }
        self.init_other_style = {
            "mode": "markers",
            "opacity": 0.4,
        }

    @staticmethod
    def _create_hovertemplate(text):
        hovertemplate = (
            "<b>"
            + text
            + " <b>"
            + "<br><b>bic</b>: %{y:.2f}"
            + "<br><b>No. of features</b>: %{x}<br><extra></extra>"
        )
        return hovertemplate

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a stepwise plot using the Plotly API.
        """
        fig = self._get_fig(fig)
        df = pd.DataFrame(self.data)
        min_value = min(df["s"])
        max_value = max(df["s"])
        normalized_list = [
            ((x - min_value) / (max_value - min_value)) * 50 for x in df["s"]
        ]
        fig.add_trace(
            go.Scatter(
                name="Start",
                x=[df["x"][0]],
                y=[df["y"][0]],
                **self.init_start_style,
                hovertemplate=self._create_hovertemplate("Start"),
            )
        )
        for i in range(len(self.data["c"]) - 1):
            fig.add_trace(
                go.Scatter(
                    name=f"{df['sign'][i+1]} {df['c'][i+1]}",
                    x=[list(df["x"])[i + 1]],
                    y=[list(df["y"])[i + 1]],
                    **self.init_other_style,
                    marker=dict(
                        size=normalized_list[i + 1]
                        if normalized_list[i + 1] > 15
                        else 15,
                        color="green" if df["sign"][i + 1] == "+" else "red",
                        sizemode="area",
                    ),
                    hovertemplate="<b>bic</b>: %{y:.2f}"
                    + "<br><b>No. of features</b>: %{x}<br>"
                    + f"{df['sign'][i+1]} {df['c'][i+1]} <extra></extra>",
                )
            )
        i = -1
        if self.layout["direction"] == "forward":
            condition = self.data["sign"] != "-"
            while self.data["sign"][i] == "-":
                i -= 1
        else:
            condition = self.data["sign"] != "+"
            while self.data["sign"][i] == "+":
                i -= 1
        fig.add_trace(
            go.Scatter(
                name="End",
                x=[list(df["x"])[i]],
                y=[list(df["y"])[i]],
                **self.init_start_style,
                hovertemplate=self._create_hovertemplate("End"),
            )
        )
        data = np.column_stack(
            (self.data["x"][condition], self.data["y"][condition])
        ).tolist()
        fig.add_trace(
            go.Scatter(
                x=[row[0] for row in data],
                y=[row[1] for row in data],
                mode="lines",
                line=dict(shape="spline", dash="dash"),
                showlegend=False,
            )
        )
        fig.update_layout(**self._update_dict(self.init_layout_style, style_kwargs))
        return fig
