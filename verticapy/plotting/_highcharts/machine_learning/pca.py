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
            "yAxis": {"title": {"text": y_label}, "min": -1, "max": 1,},
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
        return None

    # Draw.

    def draw(self, chart: Optional[HChart] = None, **style_kwargs) -> HChart:
        """
        Draws a PCA circle plot using the HC API.
        """
        chart = self._get_chart(chart, width=400, height=400)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        data = [[x / 1000, np.sqrt(1 - (x / 1000) ** 2)] for x in range(-1000, 1000, 1)]
        data += [
            [x / 1000, -np.sqrt(1 - (x / 1000) ** 2)] for x in range(1000, -1000, -1)
        ]
        data += [[-1, 0]]
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
            "marker": {"fillColor": "white", "lineWidth": 1, "lineColor": "black",},
        }
        return None

    # Draw.

    def draw(self, chart: Optional[HChart] = None, **style_kwargs) -> HChart:
        """
        Draws a PCA Scree Plot using the HC API.
        """
        chart = self._get_chart(chart)
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


class PCAVarPlot(PCACirclePlot):

    # Properties.

    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["pca_var"]:
        return "pca_var"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "marker": "^",
            "s": 100,
            "edgecolors": "black",
            "color": self.get_colors(idx=0),
        }
        self.init_style_plot = {
            "linestyle": "--",
            "color": "black",
        }
        return None

    # Draw.

    def draw(self, chart: Optional[HChart] = None, **style_kwargs) -> HChart:
        """
        Draws a PCA Variance Plot using the Matplotlib API.
        """
        ax, fig = self._get_ax_fig(ax, size=(6, 6), set_axis_below=True, grid=True)
        n = len(self.data["x"])
        min_x, max_x = min(self.data["x"]), max(self.data["x"])
        min_y, max_y = min(self.data["y"]), max(self.data["y"])
        delta_x = (max_x - min_x) * 0.04
        delta_y = (max_y - min_y) * 0.04
        for i in range(n):
            ax.text(
                self.data["x"][i],
                self.data["y"][i] + delta_y,
                self.layout["columns"][i],
                horizontalalignment="center",
            )
        img = ax.scatter(
            self.data["x"],
            self.data["y"],
            **self._update_dict(self.init_style, style_kwargs, 0),
        )
        ax.plot(
            [min_x - 5 * delta_x, max_x + 5 * delta_x],
            [0.0, 0.0],
            **self.init_style_plot,
        )
        ax.plot(
            [0.0, 0.0],
            [min_y - 5 * delta_y, max_y + 5 * delta_y],
            **self.init_style_plot,
        )
        ax.set_xlim(min_x - 5 * delta_x, max_x + 5 * delta_x)
        ax.set_ylim(min_y - 5 * delta_y, max_y + 5 * delta_y)
        if self.data["explained_variance"][0]:
            dim1 = f"({round(self.data['explained_variance'][0] * 100, 1)}%)"
        else:
            dim1 = ""
        ax.set_xlabel(f"Dim{self.data['dim'][0]} {dim1}")
        if self.data["explained_variance"][1]:
            dim1 = f"({round(self.data['explained_variance'][1] * 100, 1)}%)"
        else:
            dim1 = ""
        ax.set_ylabel(f"Dim{self.data['dim'][1]} {dim1}")
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        if "c" in style_kwargs:
            fig.colorbar(img).set_label(self.layout["method"])
        return ax
