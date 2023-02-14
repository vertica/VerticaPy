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


def heatmap(
    query: str = "",
    data: list = [],
    options: dict = {},
    width: int = 600,
    height: int = 400,
):
    from verticapy.plotting._highcharts.highchart import data_to_columns

    chart = Highchart(width=width, height=height)
    default_options = {
        "chart": {
            "type": "heatmap",
            "marginTop": 40,
            "marginBottom": 80,
            "plotBorderWidth": 1,
        },
        "title": {"text": ""},
        "legend": {},
        "colorAxis": {"minColor": "#FFFFFF", "maxColor": gen_colors()[0]},
        "xAxis": {"title": {"text": ""}},
        "yAxis": {"title": {"text": ""}},
        "tooltip": {
            "formatter": (
                "function () {return '<b>[' + this.series.xAxis."
                "categories[this.point.x] + ', ' + this.series.yAxis"
                ".categories[this.point.y] + ']</b>: ' + this.point"
                ".value + '</b>';}"
            )
        },
    }
    default_options["colors"] = gen_colors()
    chart.set_dict_options(default_options)
    if query:
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
        columns = data_to_columns(data, n)
        all_categories = list(set(columns[0]))
        all_subcategories = list(set(columns[1]))
        dict_categories = {}
        for elem in all_categories:
            dict_categories[elem] = {}
        for i in range(len(columns[0])):
            dict_categories[columns[0][i]][columns[1][i]] = columns[2][i]
        data = []
        for idx, elem in enumerate(dict_categories):
            for idx2, cat in enumerate(all_subcategories):
                try:
                    data += [[idx, idx2, dict_categories[elem][cat]]]
                except:
                    data += [[idx, idx2, None]]
        for i in range(len(all_categories)):
            if all_categories[i] == None:
                all_categories[i] = "None"
        for i in range(len(all_subcategories)):
            if all_subcategories[i] == None:
                all_subcategories[i] = "None"
        chart.set_dict_options(
            {
                "xAxis": {"categories": all_categories, "title": {"text": names[0]},},
                "yAxis": {
                    "categories": all_subcategories,
                    "title": {"text": names[1]},
                },
            }
        )
    chart.set_options(
        "legend",
        {
            "align": "right",
            "layout": "vertical",
            "margin": 0,
            "verticalAlign": "top",
            "y": 25,
            "symbolHeight": height * 0.8 - 25,
        },
    )
    chart.add_data_set(
        data,
        series_type="heatmap",
        borderWidth=1,
        dataLabels={"enabled": True, "color": "#000000"},
    )
    chart.set_dict_options(options)
    return chart
