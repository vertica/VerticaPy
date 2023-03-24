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
from vertica_highcharts import Highstock

from verticapy._config.colors import get_colors
from verticapy._utils._sql._sys import _executeSQL
from verticapy.connection import current_cursor


def candlestick(
    query: str, options: dict = {}, width: int = 600, height: int = 400
) -> Highstock:
    """
    Draws a candlestick using the High Chart API
    and the input SQL query.
    """
    data = _executeSQL(
        query,
        title=(
            "Selecting the categories and their "
            "respective aggregations to draw the chart."
        ),
        method="fetchall",
    )
    names = [desc[0] for desc in current_cursor().description]
    n = len(names)
    chart = Highstock(width=width, height=height)
    default_options = {
        "rangeSelector": {"selected": 1},
        "title": {"text": ""},
        "yAxis": [
            {
                "labels": {"align": "right", "x": -3},
                "title": {"text": ""},
                "height": "60%",
                "lineWidth": 2,
            },
            {
                "labels": {"align": "right", "x": -3},
                "title": {"text": ""},
                "top": "65%",
                "height": "35%",
                "offset": 0,
                "lineWidth": 2,
            },
        ],
    }
    default_options["colors"] = get_colors()
    chart.set_dict_options(default_options)
    for i in range(len(data)):
        for j in range(1, n):
            try:
                data[i][j] = float(data[i][j])
            except:
                pass
    data1 = [[elem[0], elem[1], elem[2], elem[3], elem[4]] for elem in data]
    data2 = [[elem[0], elem[5]] for elem in data]
    chart.add_data_set(data1, "candlestick", name="Candlesticks")
    chart.add_data_set(data2, "column", yAxis=1, name="Volume")
    chart.set_dict_options(options)
    return chart
