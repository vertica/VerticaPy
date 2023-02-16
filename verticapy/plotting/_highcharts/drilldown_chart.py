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
from verticapy._utils._sql._execute import _executeSQL
from verticapy.plotting._colors import gen_colors


def drilldown_chart(
    query: list,
    options: dict = {},
    width: int = 600,
    height: int = 400,
    chart_type: str = "column",
):
    data = _executeSQL(
        query[0],
        title=(
            "Selecting the categories and their "
            "respective aggregations to draw the first part of the drilldown."
        ),
        method="fetchall",
    )
    names = [desc[0] for desc in current_cursor().description]
    chart = Highchart(width=width, height=height)
    default_options = {
        "chart": {"type": "column"},
        "title": {"text": ""},
        "subtitle": {"text": ""},
        "xAxis": {"type": "category"},
        "yAxis": {"title": {"text": names[1]}},
        "legend": {"enabled": False},
        "plotOptions": {"series": {"borderWidth": 0, "dataLabels": {"enabled": True}}},
        "tooltip": {
            "headerFormat": "",
            "pointFormat": '<span style="color:{point.color}">{point.name}</span>: <b>{point.y}</b><br/>',
        },
    }
    default_options["colors"] = gen_colors()
    chart.set_dict_options(default_options)
    if chart_type == "bar":
        chart.set_dict_options({"chart": {"inverted": True}})
    data_final = []
    for elem in data:
        try:
            val = float(elem[1])
        except:
            val = elem[1]
        key = str(elem[0])
        data_final += [{"name": key, "y": val, "drilldown": key}]
    chart.add_data_set(data_final, chart_type, colorByPoint=True)
    data = _executeSQL(
        query[1],
        title=(
            "Selecting the categories and their respective aggregations "
            "to draw the second part of the drilldown."
        ),
        method="fetchall",
    )
    names = [desc[0] for desc in current_cursor().description]
    n = len(names)
    all_categories = list(set([elem[0] for elem in data]))
    categories = {}
    for elem in all_categories:
        categories[elem] = []
    for elem in data:
        categories[elem[0]] += [[str(elem[1]), elem[2]]]
    for elem in categories:
        chart.add_drilldown_data_set(
            categories[elem], chart_type, str(elem), name=str(elem)
        )
    chart.set_dict_options(options)
    chart.add_JSsource("https://code.highcharts.com/6/modules/drilldown.js")
    return chart
