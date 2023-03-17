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
from verticapy.plotting._plotly.bar import BarChart
from verticapy.plotting._plotly.pie import PieChart, NestedPieChart
from verticapy.plotting._plotly.barh import HorizontalBarChart
from verticapy.plotting._plotly.base import PlotlyBase

import plotly.io as pio
import plotly.graph_objects as go

pio.templates["VerticaPy"] = go.layout.Template(layout_colorway=PlotlyBase.get_colors())
pio.templates.default = "VerticaPy"
