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
from typing import Literal, Optional, Union

from verticapy._typing import ArrayLike

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs._figure import Figure

from verticapy.plotting._plotly.base import PlotlyBase


class BoxPlot(PlotlyBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["box"]:
        return "bar"

    @property
    def _compute_method(self) -> Literal["describe"]:
        return "describe"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_layout_style = {}

    def _create_bar_info(
        self,
        y_value: str,
        base: Union[int, float],
        bar: Union[int, float],
        labl_value: str,
        labl: str,
        orientation: Literal["h", "v"],
    ) -> Figure:
        data = [[y_value, bar - base, base]]
        df = pd.DataFrame(data, columns=["Y", "bar", "base"])
        if orientation == "h":
            data_dic = {"y": "Y", "x": "bar"}
        else:
            data_dic = {"x": "Y", "y": "bar"}
        fig = px.bar(
            df, base="base", orientation=orientation, barmode="stack", **data_dic
        ).update_traces(opacity=0.00, hovertemplate=f"{labl}:{labl_value}")
        return fig

    def _create_dataframe_for_outliers(
        self, fliers: list, traces: ArrayLike
    ) -> pd.DataFrame:
        a = []
        b = []
        c = []
        for i, ele in enumerate(fliers):
            for j in range(len(ele)):
                if ele[j]:
                    a.append(ele[j])
                    b.append(traces[i])
                    c.append(self.get_colors()[i])
        df = pd.DataFrame({"value": a, "trace": b, "color": c})
        return df

    # Draw.

    def draw(self, fig: Optional[Figure] = None, **style_kwargs) -> Figure:
        """
        Draws a boxplot using the Plotly API.
        """
        fig = self._get_fig(fig)
        if self.data["X"].shape[1] < 2:
            min_val = self.data["X"][0][0]
            q1 = self.data["X"][1][0]
            median = self.data["X"][2][0]
            q3 = self.data["X"][3][0]
            max_val = self.data["X"][4][0]

            if (self.data["fliers"][0].size) > 0:
                points_dic = dict(x=self.data["fliers"], boxpoints="outliers")
            else:
                points_dic = dict(x=self.data["X"][2], boxpoints=False)
            fig.add_trace(
                go.Box(
                    name=self.layout["labels"][0],
                    hovertemplate="%{x}",
                    **points_dic,
                ),
            )
            fig.update_traces(
                q1=[q1],
                median=[median],
                q3=[q3],
                lowerfence=[min_val],
                upperfence=[max_val],
                orientation="h",
            )
            fig.update_layout(
                yaxis=dict(
                    showticklabels=False,
                ),
                xaxis=dict(title=self.layout["labels"][0]),
            )
            bins = [
                min_val,
                (min_val + q1) / 2,
                (q1 + median) / 2,
                (median + q3) / 2,
                (q3 + max_val) / 2,
                max_val,
            ]
            labels = [
                "Lower",
                f"{self.data['q'][0]*100}% ",
                "Median",
                f"{self.data['q'][1]*100}% ",
                "Upper",
            ]
            values = [min_val, q1, median, q3, max_val]
            for i in range(len(values)):
                fig_add = self._create_bar_info(
                    0, bins[i], bins[i + 1], values[i], labels[i], orientation="h"
                )
                fig.add_traces(fig_add.data)
            fig.update_layout(barmode="relative", **style_kwargs)
        else:
            for I in range(self.data["X"].shape[1]):
                I = [I]
                fig.add_trace(
                    go.Box(
                        x=([self.layout["labels"][I[0]]]),
                        name=self.layout["labels"][I[0]],
                        hovertemplate="%{x}",
                    )
                )
                fig.update_traces(
                    q1=self.data["X"][1, I],
                    median=self.data["X"][2, I],
                    q3=self.data["X"][3, I],
                    lowerfence=self.data["X"][0, I],
                    upperfence=self.data["X"][4, I],
                    orientation="v",
                    selector=({"name": self.layout["labels"][I[0]]}),
                )
            df = self._create_dataframe_for_outliers(
                self.data["fliers"], self.layout["labels"]
            )
            fig_1 = px.strip(
                df,
                x="trace",
                y="value",
                color="color",
                stripmode="overlay",
                hover_data={**{"color": False, "trace": False}},
            )
            fig_1.update_layout(showlegend=False)
            for i in range(len(fig_1.data)):
                fig.add_trace(fig_1.data[i])
            fig.update_layout(showlegend=False)
            for I in range(self.data["X"].shape[1]):
                y_value = self.layout["labels"][I]
                min_val = self.data["X"][0][I]
                q1 = self.data["X"][1][I]
                median = self.data["X"][2][I]
                q3 = self.data["X"][3][I]
                max_val = self.data["X"][4][I]
                bins = [
                    min_val,
                    (min_val + q1) / 2,
                    (q1 + median) / 2,
                    (median + q3) / 2,
                    (q3 + max_val) / 2,
                    max_val,
                ]
                labels = [
                    "Lower",
                    f"{self.data['q'][0]*100}% ",
                    "Median",
                    f"{self.data['q'][1]*100}% ",
                    "Upper",
                ]
                values = [min_val, q1, median, q3, max_val]
                for i in range(len(values)):
                    fig_add = self._create_bar_info(
                        y_value,
                        bins[i],
                        bins[i + 1],
                        values[i],
                        labels[i],
                        orientation="v",
                    )
                    fig.add_traces(fig_add.data)
                fig.update_layout(
                    barmode="relative",
                    yaxis=dict(title=self.layout["y_label"]),
                    xaxis=dict(title=self.layout["x_label"]),
                    **style_kwargs,
                )
        return fig


class BarChart2D(PlotlyBase):
    ...
