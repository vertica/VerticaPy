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

from verticapy._typing import HChart
from verticapy.plotting._highcharts.base import HighchartsBase


class PCACirclePlot(HighchartsBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["pca_circle"]:
        return "pca_circle"

    # Styling Methods.

    def _init_style(self) -> None:
        if self.data["explained_variance"][0]:
            x_label = f"({round(self.data['explained_variance'][0] * 100, 1)}%)"
            x_label = f"Dim{self.data['dim'][0]} {x_label}"
        else:
            x_label = ""
        if self.data["explained_variance"][1]:
            y_label = f"({round(self.data['explained_variance'][1] * 100, 1)}%)"
            y_label = f"Dim{self.data['dim'][1]} {y_label}"
        else:
            y_label = ""
        self.init_style = {
            "title": {"text": ""},
            "xAxis": {
                "reversed": False,
                "title": {"enabled": True, "text": x_label},
                "startOnTick": True,
                "endOnTick": True,
                "showLastLabel": True,
                "min": -1,
                "max": 1,
            },
            "yAxis": {
                "title": {"text": y_label},
                "min": -1,
                "max": 1,
            },
            "legend": {"enabled": True},
            "tooltip": {
                "headerFormat": '<span style="color:{series.color}">\u25CF</span> {series.name} <br/>',
                "pointFormat": f"Dim{self.data['dim'][0]}"
                + ": {point.x} <br/> "
                + f"Dim{self.data['dim'][1]}"
                + ": {point.y}",
            },
            "colors": self.get_colors(),
        }
        self.init_style_circle = {"color": "#EFEFEF", "zIndex": 0}
        self.layout["columns"] = self._clean_quotes(self.layout["columns"])

    # Draw.

    def draw(self, chart: Optional[HChart] = None, **style_kwargs) -> HChart:
        """
        Draws a PCA circle plot using the HC API.
        """
        chart, style_kwargs = self._get_chart(
            chart, width=400, height=400, style_kwargs=style_kwargs
        )
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        data = [
            [np.cos(2 * np.pi * x / 1000), np.sin(2 * np.pi * x / 1000)]
            for x in range(-1000, 1000, 1)
        ]
        chart.add_data_set(data, "spline", name="Circle", **self.init_style_circle)
        n = len(self.data["x"])
        for i in range(n):
            data = [[0, 0], [self.data["x"][i], self.data["y"][i]]]
            chart.add_data_set(data, "line", name=self.layout["columns"][i])
        return chart


class PCAScreePlot(HighchartsBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["pca_scree"]:
        return "pca_scree"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "title": {"text": ""},
            "chart": {"type": "column"},
            "legend": {"enabled": False},
            "colors": self.get_colors(),
            "xAxis": {
                "type": "category",
                "title": {"text": self.layout["x_label"]},
                "categories": self.data["x"].tolist(),
            },
            "yAxis": {"title": {"text": self.layout["y_label"]}},
            "tooltip": {"headerFormat": "", "pointFormat": "{point.y}%"},
        }
        self.init_style_bar = {
            "zIndex": 0,
        }
        self.init_style_line = {
            "zIndex": 1,
            "color": "black",
            "marker": {
                "fillColor": "white",
                "lineWidth": 1,
                "lineColor": "black",
            },
        }

    # Draw.

    def draw(self, chart: Optional[HChart] = None, **style_kwargs) -> HChart:
        """
        Draws a PCA Scree plot using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        chart.add_data_set(
            np.round(self.data["y"], 3).tolist(),
            "bar",
            name="percentage_explained_variance",
            **self.init_style_bar,
        )
        chart.add_data_set(
            np.round(self.data["y"], 3).tolist(),
            "line",
            name="percentage_explained_variance_line",
            **self.init_style_line,
        )
        return chart


class PCAVarPlot(HighchartsBase):
    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["pca_var"]:
        return "pca_var"

    # Styling Methods.

    def _init_style(self) -> None:
        min_x, max_x = np.nanmin(self.data["x"]), np.nanmax(self.data["x"])
        min_y, max_y = np.nanmin(self.data["y"]), np.nanmax(self.data["y"])
        if self.data["explained_variance"][0]:
            x_label = f"Dim{self.data['dim'][0]} ({round(self.data['explained_variance'][0] * 100, 1)}%)"
        else:
            x_label = ""
        if self.data["explained_variance"][1]:
            y_label = f"Dim{self.data['dim'][1]} ({round(self.data['explained_variance'][1] * 100, 1)}%)"
        else:
            y_label = ""
        self.init_style = {
            "title": {"text": ""},
            "xAxis": {
                "reversed": False,
                "title": {"enabled": True, "text": x_label},
                "startOnTick": True,
                "endOnTick": True,
                "showLastLabel": True,
                "min": min_x,
                "max": max_x,
            },
            "yAxis": {
                "title": {"text": y_label},
                "min": min_y,
                "max": max_y,
            },
            "legend": {"enabled": True},
            "plotOptions": {
                "scatter": {
                    "marker": {
                        "radius": 5,
                        "states": {
                            "hover": {"enabled": True, "lineColor": "rgb(100,100,100)"}
                        },
                    },
                    "states": {"hover": {"marker": {"enabled": False}}},
                }
            },
            "tooltip": {
                "headerFormat": '<span style="color:{series.color}">\u25CF</span> {series.name} <br/>',
                "pointFormat": "<b>"
                + f"Dim{self.data['dim'][0]}"
                + "</b>: {point.x}<br><b>"
                + f"Dim{self.data['dim'][1]}"
                + "</b>: {point.y}<br>",
            },
            "colors": self.get_colors(),
        }
        if self.layout["has_category"]:
            self.init_style["tooltip"]["pointFormat"] += (
                "<b>" + self.layout["method"] + "</b>: {point.z}<br>"
            )
        self.init_style_axis = {
            "color": "black",
            "dashStyle": "longdash",
            "enableMouseTracking": False,
        }

    # Draw.

    def draw(self, chart: Optional[HChart] = None, **style_kwargs) -> HChart:
        """
        Draws a PCA variance plot using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        kind = "scatter"
        for i, col in enumerate(self._clean_quotes(self.layout["columns"])):
            data = [self.data["x"][i], self.data["y"][i]]
            if self.layout["has_category"]:
                kind = "bubble"
                data += [self.data["c"][i]]
            chart.add_data_set([data], kind, name=col)
        min_x, max_x = np.nanmin(self.data["x"]), np.nanmax(self.data["x"])
        min_y, max_y = np.nanmin(self.data["y"]), np.nanmax(self.data["y"])
        data_x = [[min_x + (max_x - min_x) * i / 1000, 0] for i in range(1001)]
        data_y = [[0, min_y + (max_y - min_y) * i / 1000] for i in range(1001)]
        chart.add_data_set(data_x, "line", name="axis", **self.init_style_axis)
        chart.add_data_set(
            data_y, "line", name="axis", linkedTo=":previous", **self.init_style_axis
        )
        return chart
