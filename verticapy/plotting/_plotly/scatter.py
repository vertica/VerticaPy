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
            "width": 700,
            "height": 500,
            "autosize": False,
            "xaxis_title": self.layout["columns"][0],
            "yaxis_title": self.layout["columns"][1],
            "xaxis": dict(
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
                zeroline=False,
            ),
            "yaxis": dict(
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
                zeroline=False,
            ),
        }

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a scatter plot using the Plotly API.
        """
        fig_base = self._get_fig(fig)
        color_option = {}
        data = self.data["X"]
        if self.data.get("s") is not None:
            data = np.column_stack((data, self.data["s"]))
        if self.data.get("c") is not None:
            data = np.column_stack((data, self.data["c"]))
        column_names = self.layout["columns"]
        columns = column_names
        if self.layout.get("size") is not None:
            columns = column_names + [self.layout["size"]]
        if self.layout.get("c") is not None:
            columns = columns + [self.layout["c"]]
        df = pd.DataFrame(
            data=data,
            columns=columns,
        )
        if self.data.get("s") is not None:
            df[column_names[0]] = df[column_names[0]].astype(float)
            df[column_names[1]] = df[column_names[1]].astype(float)
            df[self.layout["size"]] = df[self.layout["size"]].astype(float)
            min_value = df[self.layout["size"]].min()
            max_value = df[self.layout["size"]].max()
            df[self.layout["size"]] = (df[self.layout["size"]] - min_value) / (
                max_value - min_value
            )
        if self.layout["c"]:
            color_option["color"] = self.layout["c"]
        user_colors = style_kwargs.get("color", style_kwargs.get("colors"))
        if isinstance(user_colors, str):
            user_colors = [user_colors]
        color_list = (
            user_colors + self.get_colors() if user_colors else self.get_colors()
        )
        if "colors" in style_kwargs:
            del style_kwargs["colors"]
        if self.data["X"].shape[1] < 3:
            fig = px.scatter(
                df,
                x=column_names[0],
                y=column_names[1],
                color_discrete_sequence=color_list,
                size=self.layout["size"] if self.layout["size"] else None,
                **color_option,
            )
            fig.update_layout(**self._update_dict(self.init_style, style_kwargs))
        elif self.data["X"].shape[1] == 3:
            fig = px.scatter_3d(
                df,
                x=column_names[0],
                y=column_names[1],
                z=column_names[2],
                color_discrete_sequence=color_list,
                size=self.layout["size"] if self.layout["size"] else None,
                **color_option,
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title=columns[0],
                    yaxis_title=columns[1],
                    zaxis_title=columns[2],
                ),
                scene_aspectmode="cube",
                height=700,
                autosize=False,
            )
        for i in range(len(fig.data)):
            fig_base.add_trace(fig.data[i])
        fig_base.update_layout(fig.layout)
        return fig_base
