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
# High Chart
from vertica_highcharts import Highchart

# VerticaPy Modules
from verticapy.connect import current_cursor
from verticapy._utils._sql import _executeSQL
from verticapy.plotting._colors import gen_colors


def spider(query: str, options: dict = {}, width: int = 600, height: int = 400):
    from verticapy.plotting._highcharts.highchart import data_to_columns

    data = _executeSQL(
        query,
        title=(
            "Selecting the categories and their respective "
            "aggregations to draw the chart."
        ),
        method="fetchall",
    )
    names = [desc[0] for desc in current_cursor().description]
    n = len(names)
    chart = Highchart(width=width, height=height)
    default_options = {
        "chart": {"polar": True, "type": "line", "renderTo": "test"},
        "title": {"text": "", "x": -80},
        "pane": {"size": "80%"},
        "xAxis": {"tickmarkPlacement": "on", "lineWidth": 0},
        "yAxis": {"gridLineInterpolation": "polygon", "lineWidth": 0, "min": 0},
        "tooltip": {
            "shared": True,
            "pointFormat": '<span style="color:{series.color}">{series.name}: <b>{point.y:,.0f}</b><br/>',
        },
        "legend": {
            "align": "right",
            "verticalAlign": "top",
            "y": 70,
            "layout": "vertical",
        },
    }
    default_options["colors"] = gen_colors()
    chart.set_dict_options(default_options)
    columns = data_to_columns(data, n)
    chart.set_dict_options({"xAxis": {"categories": columns[0]}})
    for i in range(1, n):
        chart.add_data_set(columns[i], name=names[i], pointPlacement="on")
    chart.set_dict_options(options)
    return chart
