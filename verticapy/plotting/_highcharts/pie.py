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
from vertica_highcharts import Highchart

from verticapy._config.colors import get_colors
from verticapy._utils._sql._sys import _executeSQL
from verticapy.connection import current_cursor


def pie(
    query: str,
    options: dict = {},
    width: int = 600,
    height: int = 400,
    chart_type: str = "regular",
):
    data = _executeSQL(
        query,
        title="Selecting the categories and their respective aggregations to draw the chart.",
        method="fetchall",
    )
    names = [desc[0] for desc in current_cursor().description]
    n = len(names)
    chart = Highchart(width=width, height=height)
    default_options = {
        "title": {"text": ""},
        "chart": {"inverted": True},
        "xAxis": {
            "reversed": False,
            "title": {"text": names[0], "enabled": True},
            "maxPadding": 0.05,
            "showLastLabel": True,
        },
        "yAxis": {"title": {"text": names[1], "enabled": True}},
        "plotOptions": {
            "pie": {
                "allowPointSelect": True,
                "cursor": "pointer",
                "showInLegend": True,
                "size": "110%",
            }
        },
        "tooltip": {"pointFormat": str(names[1]) + ": <b>{point.y}</b>"},
    }
    if "3d" not in chart_type:
        default_options["colors"] = get_colors()
    chart.set_dict_options(default_options)
    if "3d" in chart_type:
        chart.set_dict_options(
            {"chart": {"type": "pie", "options3d": {"enabled": True, "alpha": 45}}}
        )
        chart.add_JSsource("https://code.highcharts.com/6/highcharts-3d.js")
        chart_type = chart_type.replace("3d", "")
    data_pie = []
    for elem in data:
        try:
            val = float(elem[1])
        except:
            val = elem[1]
        try:
            key = float(elem[0])
        except:
            key = elem[0]
            if key == None:
                key = "None"
        data_pie += [{"name": key, "y": val}]
    data_pie[-1]["sliced"], data_pie[-1]["selected"] = True, True
    chart.add_data_set(data_pie, "pie")
    if chart_type == "half":
        chart.set_dict_options(
            {
                "plotOptions": {"pie": {"startAngle": -90, "endAngle": 90}},
                "legend": {"enabled": False},
            }
        )
    elif chart_type == "donut":
        chart.set_dict_options(
            {
                "chart": {"type": "pie"},
                "plotOptions": {"pie": {"innerSize": 100, "depth": 45}},
            }
        )
    chart.set_dict_options(options)
    return chart
