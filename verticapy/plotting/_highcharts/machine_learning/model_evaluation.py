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


class ROCCurve(HighchartsBase):
    # Properties.

    @property
    def _category(self) -> Literal["graph"]:
        return "graph"

    @property
    def _kind(self) -> Literal["roc"]:
        return "roc"

    # Styling Methods.

    def _init_style(self) -> None:
        auc = round(self.data["auc"], 3)
        self.init_style = {
            "title": {"text": self.layout["title"]},
            "xAxis": {
                "reversed": False,
                "title": {"enabled": True, "text": self.layout["x_label"]},
                "startOnTick": True,
                "endOnTick": True,
                "showLastLabel": True,
                "min": 0.0,
                "max": 1.0,
            },
            "yAxis": {
                "title": {"text": self.layout["y_label"]},
                "min": 0.0,
                "max": 1.0,
            },
            "legend": {"enabled": False},
            "subtitle": {
                "text": f"AUC = {round(auc, 4) * 100}%",
                "align": "left",
            },
            "tooltip": {
                "headerFormat": "",
                "pointFormat": "[{point.x}, {point.y}]",
            },
            "colors": self.get_colors(),
        }

    # Draw.

    def draw(
        self,
        chart: Optional[HChart] = None,
        **style_kwargs,
    ) -> HChart:
        """
        Draws a machine learning ROC curve using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        data = np.column_stack((self.data["x"], self.data["y"])).tolist()
        chart.add_data_set(data, "area", self.layout["y_label"], step=True)
        return chart


class PRCCurve(ROCCurve):
    # Properties.

    @property
    def _kind(self) -> Literal["prc"]:
        return "prc"


class CutoffCurve(HighchartsBase):
    # Properties.

    @property
    def _kind(self) -> Literal["cutoff"]:
        return "cutoff"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "title": {"text": self.layout["title"]},
            "xAxis": {
                "reversed": False,
                "title": {"enabled": True, "text": self.layout["x_label"]},
                "startOnTick": True,
                "endOnTick": True,
                "showLastLabel": True,
                "min": 0.0,
                "max": 1.0,
            },
            "yAxis": {
                "min": 0.0,
                "max": 1.0,
            },
            "legend": {"enabled": True},
            "tooltip": {"crosshairs": True, "shared": True},
            "colors": self.get_colors(),
        }

    # Draw.

    def draw(
        self,
        chart: Optional[HChart] = None,
        **style_kwargs,
    ) -> HChart:
        """
        Draws a machine cutoff curve using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        data = np.column_stack((self.data["x"], self.data["y"])).tolist()
        chart.add_data_set(data, "spline", self.layout["y_label"])
        data = np.column_stack((self.data["x"], self.data["z"])).tolist()
        chart.add_data_set(data, "spline", self.layout["z_label"])
        return chart


class LiftChart(HighchartsBase):
    # Properties.

    @property
    def _kind(self) -> Literal["lift"]:
        return "lift"

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "title": {"text": self.layout["title"]},
            "xAxis": {
                "reversed": False,
                "title": {"enabled": True, "text": self.layout["x_label"]},
                "startOnTick": True,
                "endOnTick": True,
                "showLastLabel": True,
                "min": 0.0,
                "max": 1.0,
            },
            "legend": {"enabled": True},
            "tooltip": {"crosshairs": True, "shared": True},
            "colors": self.get_colors(),
        }
        self.init_style_y = {"zIndex": 1, "fillOpacity": 0.9}
        self.init_style_z = {"zIndex": 0, "fillOpacity": 0.9}

    # Draw.

    def draw(
        self,
        chart: Optional[HChart] = None,
        **style_kwargs,
    ) -> HChart:
        """
        Draws a machine cutoff curve using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        data = np.column_stack((self.data["x"], self.data["y"])).tolist()
        chart.add_data_set(data, "area", self.layout["y_label"], **self.init_style_y)
        data = np.column_stack((self.data["x"], self.data["z"])).tolist()
        chart.add_data_set(data, "area", self.layout["z_label"], **self.init_style_z)
        return chart
