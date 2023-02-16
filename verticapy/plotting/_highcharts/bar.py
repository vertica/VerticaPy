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


def bar(
    query: str,
    options: dict = {},
    width: int = 600,
    height: int = 400,
    chart_type: str = "regular",
):
    from verticapy.plotting._highcharts.base import data_to_columns, sort_classes

    is_stacked = "stacked" in chart_type
    if chart_type == "stacked_hist":
        chart_type = "hist"
    if chart_type == "stacked_bar":
        chart_type = "bar"
    data = _executeSQL(
        query,
        title=(
            "Selecting the categories and their"
            " respective aggregations to draw the chart."
        ),
        method="fetchall",
    )
    names = [desc[0] for desc in current_cursor().description]
    n = len(names)
    chart = Highchart(width=width, height=height)
    if chart_type == "hist":
        default_options = {
            "title": {"text": ""},
            "chart": {"type": "column"},
            "xAxis": {"type": "category"},
            "legend": {"enabled": False},
        }
    else:
        default_options = {
            "title": {"text": ""},
            "chart": {"inverted": True},
            "xAxis": {
                "reversed": False,
                "title": {"enabled": True, "text": names[-2]},
                "maxPadding": 0.05,
                "showLastLabel": True,
            },
            "yAxis": {"title": {"text": names[-1]}},
            "legend": {"enabled": False},
        }
    default_options["colors"] = gen_colors()
    chart.set_dict_options(default_options)
    columns = data_to_columns(data, n)
    if n == 2:
        for i in range(len(columns[0])):
            if columns[0][i] == None:
                columns[0][i] = "None"
        chart.set_dict_options(
            {
                "xAxis": {"categories": columns[0]},
                "yAxis": {"title": {"text": names[1]}},
            }
        )
        chart.set_dict_options(
            {"tooltip": {"headerFormat": "", "pointFormat": "{point.y}"}}
        )
        chart.add_data_set(columns[1], "bar", names[-1], colorByPoint=True)
    elif n == 3:
        all_categories = sort_classes(list(set(columns[0])))
        all_subcategories = sort_classes(list(set(columns[1])))
        dict_categories = {}
        for elem in all_categories:
            dict_categories[elem] = {}
        for i in range(len(columns[0])):
            dict_categories[columns[0][i]][columns[1][i]] = columns[2][i]
        for idx, elem in enumerate(dict_categories):
            data = []
            for cat in all_subcategories:
                try:
                    data += [dict_categories[elem][cat]]
                except:
                    data += [None]
            chart.add_data_set(data, "bar", name=str(elem))
        chart.set_dict_options(
            {
                "xAxis": {"categories": all_subcategories},
                "yAxis": {"title": {"text": names[2]}},
                "legend": {"enabled": True, "title": {"text": names[0]}},
                "plotOptions": {"bar": {"dataLabels": {"enabled": True}}},
            }
        )
        if is_stacked:
            chart.set_dict_options({"plotOptions": {"series": {"stacking": "normal"}}})
    chart.set_dict_options(options)
    return chart
