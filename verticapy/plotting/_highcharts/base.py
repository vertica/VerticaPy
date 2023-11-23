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
from typing import Optional

from vertica_highcharts import Highchart, Highstock

from verticapy._utils._sql._format import format_type
from verticapy._typing import HChart, NoneType

from verticapy.plotting.base import PlottingBase


class HighchartsBase(PlottingBase):
    def _get_chart(
        self,
        chart: Optional[HChart] = None,
        width: int = 600,
        height: int = 400,
        stock: bool = False,
        style_kwargs: Optional[dict] = None,
    ) -> HChart:
        style_kwargs = format_type(style_kwargs, dtype=dict)
        kwargs = copy.deepcopy(style_kwargs)
        if not isinstance(chart, NoneType):
            return chart, kwargs
        if "figsize" in kwargs and isinstance(kwargs, tuple):
            width, height = kwargs["figsize"]
            del kwargs["size"]
        if "width" in kwargs:
            width = kwargs["width"]
            del kwargs["width"]
        if "height" in kwargs:
            height = kwargs["height"]
            del kwargs["height"]
        if stock or ("stock" in self.layout and self.layout["stock"]):
            return Highstock(width=width, height=height), kwargs
        else:
            return Highchart(width=width, height=height), kwargs
