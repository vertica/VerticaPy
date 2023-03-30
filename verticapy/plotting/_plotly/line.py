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
        return None

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a time series plot using the plotly API.
        """
        fig_base = self._get_fig(fig)
        if "z" in self.data:
            self.init_style["markers"] = False
            self.init_style["color"] = "color"
            data_args = dict(
                data=(
                    np.column_stack((self.data["x"], self.data["Y"], self.data["z"]))
                ),
                columns=["time", self.layout["columns"][0], "color"],
            )
        else:
            self.init_style["markers"] = True
            data_args = dict(
                data=(np.column_stack((self.data["x"], self.data["Y"]))),
                columns=["time", self.layout["columns"][0]],
            )
        df = pd.DataFrame(**data_args)
        fig = px.line(
            df,
            x="time",
            y=self.layout["columns"][0],
            **self._update_dict(self.init_style, style_kwargs),
        )
        fig_base.add_trace(fig.data[0])
        fig_base.update_layout(fig.layout)
        return fig_base


class MultiLinePlot(PlotlyBase):
    ...
