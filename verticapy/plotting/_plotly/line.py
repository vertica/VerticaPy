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

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure

from verticapy.plotting._plotly.base import PlotlyBase


class LinePlot(PlotlyBase):
    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["line"]:
        return "line"

    @property
    def _compute_method(self) -> Literal["line"]:
        return "line"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {"width": 800, "height": 450}

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        step: bool = False,
        markers: bool = False,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a time series plot using the plotly API.
        """
        fig_base = self._get_fig(fig)
        add_params = {}
        add_params["markers"] = markers
        if "z" in self.data:
            add_params["color"] = "color"
            data_args = dict(
                data=(
                    np.column_stack((self.data["x"], self.data["Y"], self.data["z"]))
                ),
                columns=["time", self.layout["columns"][0], "color"],
            )
        else:
            data_args = dict(
                data=(np.column_stack((self.data["x"], self.data["Y"]))),
                columns=["time", self.layout["columns"][0]],
            )
        df = pd.DataFrame(**data_args)
        if step and "z" in self.data:
            for elem in df["color"].unique():
                DF = df[df["color"] == elem]
                fig_base.add_trace(
                    go.Scatter(
                        x=DF["time"],
                        y=DF[self.layout["columns"][0]],
                        name=elem,
                        line_shape="hv",
                    )
                )
        else:
            fig = px.line(df, x="time", y=self.layout["columns"][0], **add_params)
        for i in range(len(fig.data) if fig else 0):
            fig_base.add_trace(fig.data[i])
        fig_base.update_layout(**self._update_dict(self.init_style, style_kwargs))
        fig_base.update_layout(fig.layout if fig else [])
        return fig_base


class MultiLinePlot(PlotlyBase):
    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["line"]:
        return "line"

    @property
    def _compute_method(self) -> Literal["line"]:
        return "line"

    # Styling Methods.
    def _init_style(self) -> None:
        self.init_style = {"width": 800, "height": 450}

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a time series plot using the plotly API.
        """
        return LinePlot.draw(self, fig, **style_kwargs)
