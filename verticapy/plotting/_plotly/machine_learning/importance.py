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
import copy

from typing import Literal, Optional

import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure

from verticapy.plotting._plotly.base import PlotlyBase


class ImportanceBarChart(PlotlyBase):
    # Properties.

    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["importance"]:
        return "importance"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_layout_style = {
            "yaxis_title": self.layout["y_label"]
            if "xylabel" in self.layout
            else "Features",
            "xaxis_title": self.layout["x_label"]
            if "x_label" in self.layout
            else "Importance (%)",
            "margin": dict(l=200, r=200, t=100, b=100),
            "barmode": "stack",
            "yaxis": {"categoryorder": "total descending"},
        }

    # Draw.

    def draw(
        self,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a coefficient importance bar chart using the Plotly API.
        """
        fig = self._get_fig(fig)
        importances_pos = copy.deepcopy(self.data["importance"])
        importances_pos[self.data["signs"] == -1] = 0.0
        importances_pos = importances_pos.tolist()
        importances_neg = copy.deepcopy(self.data["importance"])
        importances_neg[self.data["signs"] == 1] = 0.0
        importances_neg = importances_neg.tolist()
        fig.add_trace(
            go.Bar(
                x=importances_pos,
                y=self.layout["columns"],
                orientation="h",
                name="Postive",
            )
        )
        showlegend = False
        if len(self.data["signs"][self.data["signs"] == -1]) != 0:
            fig.add_trace(
                go.Bar(
                    x=importances_neg,
                    y=self.layout["columns"],
                    orientation="h",
                    name="Negative",
                )
            )
            showlegend = True
        fig.update_layout(
            showlegend=showlegend,
            **self._update_dict(self.init_layout_style, style_kwargs),
        )
        return fig
