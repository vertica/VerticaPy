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

from verticapy._utils._sql._sys import _executeSQL
from verticapy.connection import current_cursor

from verticapy.plotting._colors import gen_colors


def negative_bar(query: str, options: dict = {}, width: int = 600, height: int = 400):
    from verticapy.plotting._highcharts.base import data_to_columns, sort_classes

    data = _executeSQL(
        query,
        title="Selecting the categories and their respective aggregations to draw the chart.",
        method="fetchall",
    )
    names = [desc[0] for desc in current_cursor().description]
    n = len(names)
    chart = Highchart(width=width, height=height)
    columns = data_to_columns(data, n)
    all_categories = list(set(columns[0]))
    all_subcategories = sort_classes(list(set(columns[1])))
    default_options = {
        "chart": {"type": "bar"},
        "title": {"text": ""},
        "subtitle": {"text": ""},
        "xAxis": [
            {
                "categories": all_subcategories,
                "reversed": False,
                "labels": {"step": 1},
            },
            {
                "opposite": True,
                "reversed": False,
                "categories": all_subcategories,
                "linkedTo": 0,
                "labels": {"step": 1},
            },
        ],
        "yAxis": {
            "title": {"text": None},
            "labels": {"formatter": "function () {return (Math.abs(this.value));}"},
        },
        "plotOptions": {"series": {"stacking": "normal"}},
        "tooltip": {
            "formatter": "function () {return '<b>"
            + names[0]
            + " : </b>' + this.series.name + '<br>' + '<b>"
            + names[1]
            + "</b> : ' + '' + this.point.category + '<br/>' + '<b>"
            + names[2]
            + "</b> : ' + Math.abs(this.point.y);}"
        },
    }
    default_options["colors"] = gen_colors()
    chart.set_dict_options(default_options)
    dict_categories = {}
    for elem in all_categories:
        dict_categories[elem] = {}
    for i in range(len(columns[0])):
        dict_categories[columns[0][i]][columns[1][i]] = columns[2][i]
    for idx, elem in enumerate(dict_categories):
        data = []
        for cat in all_subcategories:
            try:
                if idx == 0:
                    data += [dict_categories[elem][cat]]
                else:
                    data += [-dict_categories[elem][cat]]
            except:
                data += [None]
        chart.add_data_set(data, "bar", name=str(elem))
    chart.set_dict_options(options)
    return chart
