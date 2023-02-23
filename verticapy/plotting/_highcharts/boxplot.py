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

from verticapy._config.colors import get_color


def boxplot(
    data: list = [],
    options: dict = {},
    width: int = 600,
    height: int = 400,
    vdf=None,
    columns: list = [],
    by: str = "",
):
    chart = Highchart(width=width, height=height)
    default_options = {
        "chart": {"type": "boxplot"},
        "title": {"text": ""},
        "legend": {"enabled": False},
        "xAxis": {"title": {"text": ""}},
        "yAxis": {"title": {"text": ""}},
    }
    default_options["colors"] = get_color()
    chart.set_dict_options(default_options)
    aggregations = ["min", "approx_25%", "approx_50%", "approx_75%", "max"]
    if (vdf) and not (by):
        x = vdf.agg(func=aggregations, columns=columns).transpose().values
        data = [x[elem] for elem in x]
        del data[0]
        chart.set_dict_options({"xAxis": {"categories": columns}})
        chart.set_dict_options({"yAxis": {"title": {"text": "Observations"}}})
        title = "Observations"
    elif vdf:
        categories = vdf[by].distinct()
        data = []
        for elem in categories:
            data += [
                vdf.search(f"{by} = '{elem}'", usecols=[columns[0], by])[columns[0]]
                .agg(func=aggregations)
                .values[columns[0]]
            ]
        chart.set_dict_options({"xAxis": {"categories": categories}})
        chart.set_dict_options({"yAxis": {"title": {"text": str(columns[0])}}})
        title = by
    chart.add_data_set(
        data,
        "boxplot",
        title,
        tooltip={"headerFormat": "<em>{point.key}</em><br/>"},
        colorByPoint=True,
    )
    chart.set_dict_options(options)
    return chart
