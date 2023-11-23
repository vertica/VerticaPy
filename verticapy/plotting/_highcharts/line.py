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
from datetime import date, datetime
from typing import Literal, Optional

import numpy as np

from verticapy._typing import HChart
from verticapy.plotting._highcharts.base import HighchartsBase


class LinePlot(HighchartsBase):
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

    # Formatting Methods.

    @staticmethod
    def _to_datetime(x: list) -> list:
        if len(x) > 0 and not isinstance(x[0], datetime) and isinstance(x[0], date):
            return [datetime.combine(d, datetime.min.time()) for d in x]
        else:
            return copy.deepcopy(x)

    def _get_kind(self) -> tuple[str, dict]:
        if self.layout["kind"] in {"area_stacked", "area_percent"}:
            kind = "area"
            stacking = "normal" if self.layout["kind"] == "area_stacked" else "percent"
            kwargs = {
                "plotOptions": {
                    "area": {
                        "stacking": stacking,
                        "lineColor": "#666666",
                        "lineWidth": 1,
                        "marker": {"lineWidth": 1, "lineColor": "#666666"},
                    }
                }
            }
        elif self.layout["kind"] == "step":
            kind = "line"
            kwargs = {"step": "right"}
        else:
            kind = self.layout["kind"]
            kwargs = {}
        return kind, kwargs

    # Styling Methods.

    def _init_style(self) -> None:
        self.init_style = {
            "title": {"text": ""},
            "xAxis": {
                "reversed": False,
                "title": {"enabled": True, "text": self.layout["order_by"]},
                "startOnTick": True,
                "endOnTick": True,
                "showLastLabel": True,
            },
            "yAxis": {
                "title": {
                    "text": self.layout["columns"][0]
                    if len(self.layout["columns"]) == 1
                    else ""
                }
            },
            "legend": {"enabled": False},
            "plotOptions": {
                "scatter": {
                    "marker": {
                        "radius": 5,
                        "states": {
                            "hover": {
                                "enabled": True,
                                "lineColor": "rgb(100,100,100)",
                            }
                        },
                    },
                    "states": {"hover": {"marker": {"enabled": False}}},
                    "tooltip": {
                        "headerFormat": '<span style="color:{series.color}">\u25CF</span> {series.name} <br/>',
                        "pointFormat": "<b>"
                        + self.layout["order_by"]
                        + "</b>: {point.x} <br/> <b>"
                        + self.layout["columns"][0]
                        if len(self.layout["columns"]) == 1
                        else "value" + "</b>: {point.y}",
                    },
                }
            },
            "colors": self.get_colors(),
        }
        if "order_by_cat" in self.layout:
            if self.layout["order_by_cat"] == "date":
                self.init_style["xAxis"] = {
                    **self.init_style["xAxis"],
                    "type": "datetime",
                    "dateTimeLabelFormats": {},
                }
        if "has_category" in self.layout and self.layout["has_category"]:
            self.init_style["legend"] = {
                "enabled": True,
                "title": {"text": self.layout["columns"][1]},
            }
        elif len(self.layout["columns"]) > 1:
            self.init_style["legend"] = {"enabled": True}

    # Draw.

    def draw(
        self,
        chart: Optional[HChart] = None,
        **style_kwargs,
    ) -> HChart:
        """
        Draws a time series plot using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        kind, kind_kwargs = self._get_kind()
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        if self.layout["kind"] == "step":
            step_kwargs = kind_kwargs
        else:
            step_kwargs = {}
            chart.set_dict_options(kind_kwargs)
        if self.layout["has_category"]:
            uniques = np.unique(self.data["z"])
            X = np.array([int(int_string) for int_string in self.data["x"]])
            for c in uniques:
                x = self._to_datetime(X[self.data["z"] == c])
                y = self.data["Y"][:, 0][self.data["z"] == c]
                data = np.column_stack((x, y)).tolist()
                chart.add_data_set(data, kind, str(c), **step_kwargs)
        else:
            x = self._to_datetime(self.data["x"])
            y = self.data["Y"][:, 0]
            data = np.column_stack((x, y)).tolist()
            chart.add_data_set(data, kind, self.layout["columns"][0], **step_kwargs)
        return chart


class MultiLinePlot(LinePlot):
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

    # Draw.

    def draw(
        self,
        chart: Optional[HChart] = None,
        **style_kwargs,
    ) -> HChart:
        """
        Draws a multi-time series plot using the HC API.
        """
        chart, style_kwargs = self._get_chart(chart, style_kwargs=style_kwargs)
        kind, kind_kwargs = self._get_kind()
        chart.set_dict_options(self.init_style)
        chart.set_dict_options(style_kwargs)
        if self.layout["kind"] == "step":
            step_kwargs = kind_kwargs
        else:
            step_kwargs = {}
            chart.set_dict_options(kind_kwargs)
        m = self.data["Y"].shape[1]
        x = self._to_datetime(self.data["x"])
        for idx in range(m):
            y = self.data["Y"][:, idx]
            data = np.column_stack((x, y)).tolist()
            chart.add_data_set(data, kind, self.layout["columns"][idx], **step_kwargs)
        return chart
