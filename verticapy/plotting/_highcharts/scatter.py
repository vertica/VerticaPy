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
from collections.abc import Iterable

from vertica_highcharts import Highchart

from verticapy._utils._sql._sys import _executeSQL
from verticapy.connection import current_cursor

from verticapy.plotting._colors import gen_colors


def scatter(
    query: str,
    options: dict = {},
    width: int = 600,
    height: int = 400,
    chart_type: str = "regular",
):
    data = _executeSQL(
        query,
        title="Selecting the different values to draw the chart.",
        method="fetchall",
    )
    names = [desc[0] for desc in current_cursor().description]
    n = len(names)
    chart = Highchart(width=width, height=height)
    default_options = {
        "title": {"text": ""},
        "xAxis": {
            "reversed": False,
            "title": {"enabled": True, "text": names[0]},
            "startOnTick": True,
            "endOnTick": True,
            "showLastLabel": True,
        },
        "yAxis": {"title": {"text": names[1]}},
        "legend": {"enabled": False},
        "plotOptions": {
            "scatter": {
                "marker": {
                    "radius": 5,
                    "states": {
                        "hover": {"enabled": True, "lineColor": "rgb(100,100,100)"}
                    },
                },
                "states": {"hover": {"marker": {"enabled": False}}},
            }
        },
    }
    if chart_type != "3d":
        default_options["colors"] = gen_colors()
    chart.set_dict_options(default_options)
    for i in range(len(data)):
        for j in range(n):
            try:
                data[i][j] = float(data[i][j])
            except:
                pass
    if n == 2:
        chart.add_data_set(data, "scatter", name="Scatter")
        chart.set_dict_options(
            {
                "tooltip": {
                    "headerFormat": "",
                    "pointFormat": str(names[0])
                    + ": <b>{point.x}</b><br>"
                    + str(names[1])
                    + ": <b>{point.y}</b>",
                }
            }
        )
    elif (n == 3) and (chart_type not in ("bubble", "3d")):
        all_categories = list(set([elem[-1] for elem in data]))
        dict_categories = {}
        for elem in all_categories:
            dict_categories[elem] = []
        for i in range(len(data)):
            dict_categories[data[i][-1]] += [[data[i][0], data[i][1]]]
        for idx, elem in enumerate(dict_categories):
            chart.add_data_set(dict_categories[elem], "scatter", name=str(elem))
        chart.set_dict_options(
            {"legend": {"enabled": True, "title": {"text": names[-1]}}}
        )
        chart.set_dict_options(
            {
                "tooltip": {
                    "pointFormat": str(names[0])
                    + ": <b>{point.x}</b><br>"
                    + str(names[1])
                    + ": <b>{point.y}</b>"
                }
            }
        )
    elif (n == 3) and (chart_type == "bubble"):
        chart.add_data_set(data, "bubble", name="Bubble")
        chart.set_dict_options(
            {
                "tooltip": {
                    "headerFormat": "",
                    "pointFormat": str(names[0])
                    + ": <b>{point.x}</b><br>"
                    + str(names[1])
                    + ": <b>{point.y}</b><br>"
                    + str(names[2])
                    + ": <b>{point.z}</b>",
                },
            }
        )
    elif (n == 3) and (chart_type == "3d"):
        chart.add_data_set(data, "scatter", name="Scatter")
        chart.set_dict_options(
            {
                "tooltip": {
                    "headerFormat": "",
                    "pointFormat": str(names[0])
                    + ": <b>{point.x}</b><br>"
                    + str(names[1])
                    + ": <b>{point.y}</b><br>"
                    + str(names[2])
                    + ": <b>{point.z}</b>",
                }
            }
        )
        chart.set_dict_options(
            {
                "chart": {
                    "renderTo": "container",
                    "margin": 100,
                    "type": "scatter",
                    "options3d": {
                        "enabled": True,
                        "alpha": 10,
                        "beta": 30,
                        "depth": 400,
                        "viewDistance": 8,
                        "frame": {
                            "bottom": {"size": 1, "color": "rgba(0,0,0,0.02)"},
                            "back": {"size": 1, "color": "rgba(0,0,0,0.04)"},
                            "side": {"size": 1, "color": "rgba(0,0,0,0.06)"},
                        },
                    },
                },
                "zAxis": {"title": {"text": names[2]}},
            }
        )
        chart.add_3d_rotation()
        chart.add_JSsource("https://code.highcharts.com/6/highcharts-3d.js")
    elif n == 4:
        all_categories = list(set([elem[-1] for elem in data]))
        dict_categories = {}
        for elem in all_categories:
            dict_categories[elem] = []
        for i in range(len(data)):
            dict_categories[data[i][-1]] += [[data[i][0], data[i][1], data[i][2]]]
        if chart_type == "3d":
            chart_type = "scatter"
        for idx, elem in enumerate(dict_categories):
            chart.add_data_set(dict_categories[elem], chart_type, name=str(elem))
        chart.set_dict_options(
            {"legend": {"enabled": True, "title": {"text": names[-1]}}}
        )
        if chart_type == "scatter":
            chart.set_dict_options(
                {
                    "chart": {
                        "renderTo": "container",
                        "margin": 100,
                        "type": "scatter",
                        "options3d": {
                            "enabled": True,
                            "alpha": 10,
                            "beta": 30,
                            "depth": 400,
                            "viewDistance": 8,
                            "frame": {
                                "bottom": {"size": 1, "color": "rgba(0,0,0,0.02)"},
                                "back": {"size": 1, "color": "rgba(0,0,0,0.04)"},
                                "side": {"size": 1, "color": "rgba(0,0,0,0.06)"},
                            },
                        },
                    },
                    "zAxis": {"title": {"text": names[2]}},
                }
            )
            chart.add_3d_rotation()
            chart.add_JSsource("https://code.highcharts.com/6/highcharts-3d.js")
        chart.set_dict_options(
            {
                "tooltip": {
                    "pointFormat": str(names[0])
                    + ": <b>{point.x}</b><br>"
                    + str(names[1])
                    + ": <b>{point.y}</b><br>"
                    + str(names[2])
                    + ": <b>{point.z}</b>"
                }
            }
        )
    chart.set_dict_options(options)
    return chart
