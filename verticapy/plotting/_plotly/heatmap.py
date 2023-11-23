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

import plotly.express as px
from plotly.graph_objs._figure import Figure

from verticapy.plotting._plotly.base import PlotlyBase


class HeatMap(PlotlyBase):
    # Properties.

    @property
    def _category(self) -> Literal["map"]:
        return "map"

    @property
    def _kind(self) -> Literal["heatmap"]:
        return "heatmap"

    @property
    def _compute_method(self) -> Literal["2D"]:
        return "2D"

    # Styling Methods.
    def _init_style(self) -> None:
        self.init_style = {
            "width": 500 + 100 * max(len(self.layout["x_labels"]) - 5, 0),
            "height": 400 + 50 * max(len(self.layout["y_labels"]) - 5, 0),
            "xaxis": dict(
                showline=True,
                linewidth=2,
                linecolor="black",
                mirror=True,
                zeroline=False,
            ),
            "yaxis": dict(
                showline=True,
                linewidth=2,
                linecolor="black",
                mirror=True,
                zeroline=False,
            ),
        }

    def _get_cmap_style(self, style_kwargs: dict) -> dict:
        if (
            "color_continuous_scale" not in style_kwargs
            and "method" in self.layout
            and (
                self.layout["method"]
                in (
                    "pearson",
                    "spearman",
                    "spearmand",
                    "kendall",
                    "biserial",
                )
            )
        ):
            return {
                "color_continuous_midpoint": 0,
                "color_continuous_scale": [
                    [0, self.get_colors()[1]],
                    [0.5, "white"],
                    [1, self.get_colors()[0]],
                ],
            }
        elif "color_continuous_scale" not in style_kwargs:
            return {"color_continuous_scale": [[0, "white"], [1, self.get_colors()[0]]]}
        else:
            return {
                key: value
                for key, value in style_kwargs.items()
                if key == "color_continuous_scale"
            }

    # Draw.

    def draw(
        self,
        colorbar: Optional[str] = None,
        extent: Optional[list] = None,
        fig: Optional[Figure] = None,
        **style_kwargs,
    ) -> Figure:
        """
        Draws a heatmap using the Plotly API.
        """
        params = {}
        trace_params = {}
        data = np.transpose((self.data["X"]))
        if data.shape[1] == 1:
            self.init_style["width"] = 250
        decimal_points = self._get_max_decimal_point(data)
        if decimal_points > 3:
            data = np.around(data.astype(np.float32), decimals=3)
        if len(self.layout["x_labels"][0].split(";")) > 1:
            x = self._convert_labels_for_heatmap(self.layout["x_labels"])
        else:
            x = self.layout["x_labels"]
        if len(self.layout["y_labels"][0].split(";")) > 1:
            y = self._convert_labels_for_heatmap(self.layout["y_labels"])[::-1]
        else:
            y = self.layout["y_labels"]
        if self.layout["with_numbers"]:
            params["text_auto"] = True
            text = data
            text.astype(str)
            trace_params = {
                "text": text,
                "texttemplate": "%{text}",
                "textfont": dict(size=12),
            }
            if decimal_points > 2:
                trace_params["texttemplate"] = "%{text:.2f}"
            if decimal_points > 8:
                trace_params["texttemplate"] = "%{text:.2e}"
        fig = px.imshow(
            data,
            labels=dict(x=self.layout["columns"][0], y=self.layout["columns"][1]),
            x=x,
            y=y,
            aspect="auto",
            **params,
            **self._get_cmap_style(style_kwargs=style_kwargs),
        )
        if "color_continuous_scale" in style_kwargs:
            del style_kwargs["color_continuous_scale"]
        fig.update_xaxes(type="category")
        fig.update_yaxes(type="category")
        fig.layout.yaxis.automargin = True
        fig.layout.xaxis.automargin = True
        fig.update_traces(
            **trace_params,
        )
        fig.update_layout(**self._update_dict(self.init_style, style_kwargs))
        return fig
