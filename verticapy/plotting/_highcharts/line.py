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
from vertica_highcharts import Highchart, Highstock

from verticapy._utils._sql._sys import _executeSQL
from verticapy.connection import current_cursor

from verticapy.plotting._colors import gen_colors


def line(
    query: str,
    options: dict = {},
    width: int = 600,
    height: int = 400,
    chart_type: str = "line",
    stock: bool = False,
):
    is_ts = True if (chart_type == "area_ts") else False
    is_range = True if (chart_type == "area_range") else False
    is_date = False
    is_multi = True if ("multi" in chart_type) else False
    if chart_type in ("area_ts", "area_range", "multi_area"):
        chart_type = "area"
    if chart_type == "multi_line":
        chart_type = "line"
    if chart_type == "multi_spline":
        chart_type = "spline"
    data = _executeSQL(
        query,
        title="Selecting the different values to draw the chart.",
        method="fetchall",
    )
    names = [desc[0] for desc in current_cursor().description]
    n = len(names)
    if stock:
        chart = Highstock(width=width, height=height)
        default_options = {
            "rangeSelector": {"selected": 0},
            "title": {"text": ""},
            "tooltip": {
                "style": {"width": "200px"},
                "valueDecimals": 4,
                "shared": True,
            },
            "yAxis": {"title": {"text": ""}},
        }
    else:
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
            "yAxis": {"title": {"text": names[1] if len(names) == 2 else ""}},
            "legend": {"enabled": False},
            "plotOptions": {
                "scatter": {
                    "marker": {
                        "radius": 5,
                        "states": {
                            "hover": {"enabled": True, "lineColor": "rgb(100,100,100)",}
                        },
                    },
                    "states": {"hover": {"marker": {"enabled": False}}},
                    "tooltip": {
                        "headerFormat": "",
                        "pointFormat": "[{point.x}, {point.y}]",
                    },
                }
            },
        }
    default_options["colors"] = gen_colors()
    chart.set_dict_options(default_options)
    for i in range(len(data)):
        if "datetime" in str(type(data[i][0])):
            is_date = True
        for j in range(n):
            try:
                data[i][j] = float(data[i][j])
            except:
                pass
    if is_date:
        chart.set_options("xAxis", {"type": "datetime", "dateTimeLabelFormats": {}})
    if n == 2:
        chart.add_data_set(data, chart_type, names[1])
    elif (n >= 3) and (is_multi):
        for i in range(1, n):
            chart.add_data_set([elem[i] for elem in data], chart_type, name=names[i])
        chart.set_dict_options(
            {
                "legend": {"enabled": True},
                "plotOptions": {
                    "area": {
                        "stacking": "normal",
                        "lineColor": "#666666",
                        "lineWidth": 1,
                        "marker": {"lineWidth": 1, "lineColor": "#666666"},
                    }
                },
                "tooltip": {"shared": True},
                "xAxis": {
                    "categories": [elem[0] for elem in data],
                    "tickmarkPlacement": "on",
                    "title": {"enabled": False},
                },
            }
        )
        if is_date:
            chart.set_dict_options(
                {
                    "xAxis": {
                        "type": "datetime",
                        "labels": {
                            "formatter": (
                                "function() {return "
                                "Highcharts.dateFormat('%a %d %b', this.value);}"
                            )
                        },
                    }
                }
            )
    elif n == 3:
        all_categories = list(set([elem[-1] for elem in data]))
        dict_categories = {}
        for elem in all_categories:
            dict_categories[elem] = []
        for i in range(len(data)):
            dict_categories[data[i][-1]] += [[data[i][0], data[i][1]]]
        for idx, elem in enumerate(dict_categories):
            chart.add_data_set(dict_categories[elem], chart_type, name=str(elem))
        chart.set_dict_options(
            {"legend": {"enabled": True, "title": {"text": names[-1]}}}
        )
    elif (n == 4) and (is_range):
        data_value = [[elem[0], elem[1]] for elem in data]
        data_range = [[elem[0], elem[2], elem[3]] for elem in data]
        chart.add_data_set(
            data_value,
            "line",
            names[1],
            zIndex=1,
            marker={"fillColor": "#263133", "lineWidth": 2},
        )
        chart.add_data_set(
            data_range,
            "arearange",
            "Range",
            lineWidth=0,
            linkedTo=":previous",
            fillOpacity=0.3,
            zIndex=0,
        )
    if is_range:
        chart.set_dict_options({"tooltip": {"crosshairs": True, "shared": True}})
    if is_ts:
        chart.set_options(
            "plotOptions",
            {
                "area": {
                    "fillColor": {
                        "linearGradient": {"x1": 0, "y1": 0, "x2": 0, "y2": 1},
                        "stops": [[0, "#FFFFFF"], [1, gen_colors()[0]]],
                    },
                    "marker": {"radius": 2},
                    "lineWidth": 1,
                    "states": {"hover": {"lineWidth": 1}},
                    "threshold": None,
                }
            },
        )
    chart.set_dict_options(options)
    return chart
