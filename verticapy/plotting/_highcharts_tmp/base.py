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
from typing import Literal, Optional, TYPE_CHECKING, Union

from vertica_highcharts import Highchart, Highstock

from verticapy._config.colors import get_colors
from verticapy._utils._sql._sys import _executeSQL
from verticapy.connection import current_cursor

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame

from verticapy.plotting._highcharts_tmp.bar import bar
from verticapy.plotting._highcharts_tmp.boxplot import boxplot
from verticapy.plotting._highcharts_tmp.candlestick import candlestick
from verticapy.plotting._highcharts_tmp.drilldown_chart import drilldown_chart
from verticapy.plotting._highcharts_tmp.heatmap import heatmap
from verticapy.plotting._highcharts_tmp.line import line
from verticapy.plotting._highcharts_tmp.negative_bar import negative_bar
from verticapy.plotting._highcharts_tmp.pie import pie
from verticapy.plotting._highcharts_tmp.scatter import scatter
from verticapy.plotting._highcharts_tmp.spider import spider


def hchart_from_vdf(
    vdf: "vDataFrame",
    x: Optional[str] = None,
    y: Optional[str] = None,
    z: Optional[str] = None,
    c: Optional[str] = None,
    aggregate: bool = True,
    kind: Literal[
        "area",
        "area_range",
        "area_ts",
        "bar",
        "biserial",
        "boxplot",
        "bubble",
        "candlestick",
        "cramer",
        "donut",
        "donut3d",
        "heatmap",
        "hist",
        "kendall",
        "line",
        "multi_area",
        "multi_line",
        "multi_spline",
        "negative_bar",
        "pearson",
        "pie",
        "pie3d",
        "pie_half",
        "scatter",
        "spearman",
        "spearmand",
        "spider",
        "spline",
        "stacked_bar",
        "stacked_hist",
    ] = "boxplot",
    width: int = 600,
    height: int = 400,
    options: dict = {},
    h: float = -1,
    max_cardinality: int = 10,
    limit=10000,
    drilldown: bool = False,
    stock: bool = False,
    alpha: float = 0.25,
) -> Union[Highchart, Highstock]:
    """
    Draws a custom chart using the High Chart API
    and the input SQL query.
    """
    if not (x):
        x = vdf.numcol()
    x, y, z, c = vdf._format_colnames(x, y, z, c, raise_error=False)
    groupby = " GROUP BY 1 " if (aggregate) else ""
    if drilldown:
        if not (z):
            z = "COUNT(*)"
        if kind == "hist":
            kind = "column"
        if isinstance(x, Iterable) and not (isinstance(x, str)):
            x = x[0]
        query = [
            f"""SELECT 
                    {x},
                    {z}
                FROM {vdf._genSQL()} 
                GROUP BY 1
                LIMIT {limit}""",
            f"""SELECT 
                    {x},
                    {y},
                    {z}
                FROM {vdf._genSQL()}
                GROUP BY 1, 2
                LIMIT {limit}""",
        ]
    elif (kind in ("pie_half", "pie", "donut", "pie3d", "donut3d")) or (
        kind in ("bar", "hist") and not (z)
    ):
        if not (y):
            y = "COUNT(*)"
        if isinstance(x, Iterable) and not (isinstance(x, str)):
            x = x[0]
        unique = vdf[x].nunique()
        is_num = vdf[x].isnum()
        order_by = " ORDER BY 2 DESC "
        if unique > max_cardinality:
            if not (aggregate):
                limit = min(limit, max_cardinality)
            elif is_num:
                order_by = f" ORDER BY MIN({x}) DESC "
                x = (
                    vdf[x].discretize(h=h, return_enum_trans=True)[0].replace("{}", x)
                    + f" AS {x}"
                )
            else:
                result = _executeSQL(
                    query=f"""
                        SELECT 
                            /*+LABEL('highchart.hchart_from_vdf')*/ 
                            {x},
                            {y} 
                        FROM {vdf._genSQL()}
                        GROUP BY 1
                        ORDER BY 2 DESC
                        LIMIT {max_cardinality}""",
                    title=(
                        "Selecting the categories and their respective aggregations "
                        "to draw the chart."
                    ),
                    method="fetchall",
                )
                result = [elem[0] for elem in result]
                result = ["NULL" if elem == None else f"'{elem}'" for elem in result]
                x = f"""
                    (CASE 
                        WHEN {x} IN ({", ".join(result)}) 
                            THEN {x} 
                        ELSE 'Others'
                     END) AS {x}"""
        query = f"""
            SELECT 
                {x},
                {y}
            FROM {vdf._genSQL()}
            {groupby}
            {order_by}
            LIMIT {limit}"""
    elif kind in (
        "bar",
        "hist",
        "stacked_bar",
        "stacked_hist",
        "heatmap",
        "negative_bar",
    ):
        where = ""
        if kind == "heatmap" and not (z):
            if not (y):
                y = "COUNT(*)"
            z = y
            y = "'' AS \"_\""
        else:
            if not (z):
                z = "COUNT(*)"
            if isinstance(x, Iterable) and not (isinstance(x, str)):
                x = x[0]
            # y
            unique = vdf[y].nunique()
            is_num = vdf[y].isnum()
            if unique > max_cardinality:
                if is_num:
                    where = f" WHERE {y} IS NOT NULL "
                    y = (
                        vdf[y]
                        .discretize(h=h, return_enum_trans=True)[0]
                        .replace("{}", y)
                        + f" AS {y}"
                    )
                else:
                    y = (
                        vdf[y]
                        .discretize(
                            k=max_cardinality, method="topk", return_enum_trans=True
                        )[0]
                        .replace("{}", y)
                        + f" AS {y}"
                    )
        # x
        unique = vdf[x].nunique()
        is_num = vdf[x].isnum()
        if unique > max_cardinality:
            if is_num:
                x = (
                    vdf[x].discretize(h=h, return_enum_trans=True)[0].replace("{}", x)
                    + f" AS {x}"
                )
            else:
                x = (
                    vdf[x]
                    .discretize(
                        k=max_cardinality, method="topk", return_enum_trans=True
                    )[0]
                    .replace("{}", x)
                    + f" AS {x}"
                )
        groupby = " GROUP BY 1, 2 " if (aggregate) else ""
        query = f"""
            SELECT
                {x},
                {y},
                {z} 
            FROM {vdf._genSQL()}
            {where}
            {groupby}
            LIMIT {limit}"""
    elif kind in ("area", "area_ts", "line", "spline"):
        if isinstance(x, Iterable) and not (isinstance(x, str)):
            x = x[0]
        if isinstance(y, Iterable) and not (isinstance(y, str)) and kind == "area_ts":
            y = y[0]
        cast = "::timestamp" if (vdf[x].isdate()) else ""
        if not (z):
            if not (isinstance(y, str)):
                y = ", ".join(y)
                kind = "multi_" + kind
            order_by = " ORDER BY 1 " if (vdf[x].isdate() or vdf[x].isnum()) else ""
            query = f"""
                SELECT 
                    {x}{cast},
                    {y}
                FROM {vdf._genSQL()}
                WHERE {x} IS NOT NULL
                {groupby}
                {order_by}
                LIMIT {limit}"""
        else:
            # z
            unique = vdf[z].nunique()
            is_num = vdf[z].isnum()
            z_copy = z
            if unique > max_cardinality:
                if is_num:
                    z = (
                        vdf[z]
                        .discretize(h=h, return_enum_trans=True)[0]
                        .replace("{}", z)
                        + f" AS {z}"
                    )
                else:
                    z = (
                        vdf[z]
                        .discretize(
                            k=max_cardinality, method="topk", return_enum_trans=True
                        )[0]
                        .replace("{}", z)
                        + f" AS {z}"
                    )
            query = f"""
                SELECT 
                    {x}{cast},
                    {y},
                    {z}
                FROM {vdf._genSQL()}
                WHERE {x} IS NOT NULL
                  AND {y} IS NOT NULL
                LIMIT {max(int(limit / unique), 1)} 
                    OVER (PARTITION BY {z_copy} 
                          ORDER BY {x} DESC)"""
    elif kind in ("scatter", "bubble"):
        if isinstance(x, Iterable) and not (isinstance(x, str)):
            x = x[0]
        cast = "::timestamp" if (vdf[x].isdate()) else ""
        if not (z) and not (c) and (kind == "scatter"):
            query = f"""
                SELECT 
                    {x}{cast},
                    {y} 
                FROM {vdf._genSQL()} 
                WHERE {x} IS NOT NULL 
                  AND {y} IS NOT NULL 
                LIMIT {limit}"""
        elif not (c) and (z):
            query = f"""
                SELECT 
                    {x}{cast},
                    {y},
                    {z}
                FROM {vdf._genSQL()}
                WHERE {x} IS NOT NULL 
                  AND {y} IS NOT NULL 
                  AND {z} IS NOT NULL 
                LIMIT {limit}"""
        else:
            # c
            unique = vdf[c].nunique()
            is_num = vdf[c].isnum()
            c_copy = c
            if unique > max_cardinality:
                if is_num:
                    c = (
                        vdf[c]
                        .discretize(h=h, return_enum_trans=True)[0]
                        .replace("{}", c)
                        + f" AS {c}"
                    )
                else:
                    c = (
                        vdf[c]
                        .discretize(
                            k=max_cardinality, method="topk", return_enum_trans=True
                        )[0]
                        .replace("{}", c)
                        + f" AS {c}"
                    )
            if z:
                z_str = f", {z}"
                z_is_not_null = f" AND {z} IS NOT NULL "
            else:
                z_str = ""
                z_is_not_null = ""
            query = f"""
                SELECT 
                    {x}{cast},
                    {y}{z_str},
                    {c} 
                FROM {vdf._genSQL()} 
                WHERE {x} IS NOT NULL 
                  AND {y} IS NOT NULL
                  {z_is_not_null}
                LIMIT {max(int(limit / unique), 1)} 
                    OVER (PARTITION BY {c_copy} 
                          ORDER BY {c_copy})"""
    elif kind == "area_range":
        if isinstance(x, Iterable) and not (isinstance(x, str)):
            x = x[0]
        order_by = " ORDER BY 1 " if (vdf[x].isdate() or vdf[x].isnum()) else ""
        cast = "::timestamp" if (vdf[x].isdate()) else ""
        query = f"""
            SELECT
                {x}{cast},
                {", ".join(y)}
            FROM {vdf._genSQL()}
            {groupby}
            {order_by}
            LIMIT {limit}"""
    elif kind == "spider":
        if not (y):
            y = "COUNT(*)"
        if isinstance(y, str):
            y = [y]
        if isinstance(x, Iterable) and not (isinstance(x, str)):
            x = x[0]
        # x
        unique = vdf[x].nunique()
        is_num = vdf[x].isnum()
        if unique > max_cardinality:
            if is_num:
                x = (
                    vdf[x].discretize(h=h, return_enum_trans=True)[0].replace("{}", x)
                    + f" AS {x}"
                )
            else:
                if len(y) == 1:
                    result = _executeSQL(
                        query=f"""
                            SELECT 
                                /*+LABEL('highchart.hchart_from_vdf')*/ 
                                {x}, 
                                {y[0]} 
                            FROM {vdf._genSQL()} 
                            GROUP BY 1 
                            ORDER BY 2 DESC 
                            LIMIT {max_cardinality}""",
                        title=(
                            "Selecting the categories and their "
                            "respective aggregations to draw the chart."
                        ),
                        method="fetchall",
                    )
                    result = [q[0] for q in result]
                    result = ["NULL" if q == None else f"'{q}'" for q in result]
                    x = f"""
                        (CASE 
                            WHEN {x} IN ({", ".join(result)}) 
                                THEN {x} 
                            ELSE 'Others' 
                         END) AS {x}"""
                else:
                    x = (
                        vdf[x]
                        .discretize(
                            k=max_cardinality, method="topk", return_enum_trans=True
                        )[0]
                        .replace("{}", x)
                        + f" AS {x}"
                    )
        query = f"""
            SELECT 
                {x}, 
                {", ".join(y)} 
            FROM {vdf._genSQL()}
            {groupby} 
            LIMIT {limit}"""
    elif kind == "candlestick":
        if isinstance(x, Iterable) and not (isinstance(x, str)):
            x = x[0]
        if aggregate:
            if isinstance(y, Iterable) and not (isinstance(y, str)) and len(y) == 1:
                y = y[0]
            if isinstance(y, str):
                query = f"""
                    SELECT 
                        {x}::timestamp, 
                        APPROXIMATE_PERCENTILE({y} 
                          USING PARAMETERS percentile = {1 - alpha}) AS open,
                        MAX({y}) AS high,
                        MIN({y}) AS low,
                        APPROXIMATE_PERCENTILE({y} 
                          USING PARAMETERS percentile = {alpha}) AS close,
                        SUM({y}) AS volume
                	FROM {vdf._genSQL()} 
                    GROUP BY 1 
                    ORDER BY 1"""
            else:
                query = f"""
                    SELECT 
                        {x}::timestamp, 
                        {", ".join(y)} 
                    FROM {vdf._genSQL()} 
                    GROUP BY 1 
                    ORDER BY 1"""
        else:
            query = f"""
                SELECT 
                    {x}::timestamp, 
                    {', '.join(y)} 
                FROM {vdf._genSQL()} 
                ORDER BY 1"""
    if drilldown:
        return drilldown_chart(
            query=query, options=options, width=width, height=height, chart_type=kind,
        )
    elif kind == "candlestick":
        return candlestick(query=query, options=options, width=width, height=height,)
    elif kind in (
        "area",
        "area_range",
        "area_ts",
        "line",
        "spline",
        "multi_area",
        "multi_line",
        "multi_spline",
    ):
        return line(
            query=query,
            options=options,
            width=width,
            height=height,
            chart_type=kind,
            stock=stock,
        )
    elif kind in ("bar", "hist", "stacked_bar", "stacked_hist"):
        return bar(
            query=query, options=options, width=width, height=height, chart_type=kind,
        )
    elif kind == "boxplot":
        if isinstance(x, str):
            x = [x]
        return boxplot(
            options=options, width=width, height=height, vdf=vdf, columns=x, by=y
        )
    elif kind in ("bubble", "scatter"):
        if x and y and z and kind == "scatter":
            kind = "3d"
        return scatter(
            query=query, options=options, width=width, height=height, chart_type=kind,
        )
    elif kind in ("pie", "pie_half", "donut", "pie3d", "donut3d"):
        if kind == "pie_half":
            kind = "half"
        return pie(
            query=query, options=options, width=width, height=height, chart_type=kind,
        )
    elif kind == "heatmap":
        chart = heatmap(query=query, width=width, height=height)
        chart.set_dict_options(
            {"colorAxis": {"maxColor": get_colors()[0], "minColor": "#FFFFFF"}}
        )
        chart.set_dict_options(options)
        return chart
    elif kind == "negative_bar":
        return negative_bar(query=query, options=options, width=width, height=height)
    elif kind == "spider":
        return spider(query=query, options=options, width=width, height=height)
    elif kind in ("pearson", "kendall", "cramer", "biserial", "spearman", "spearmand",):
        data = vdf.corr(method=kind, show=False, columns=x)
        narrow_data = data.narrow(use_number_as_category=True)
        for idx, elem in enumerate(narrow_data[0]):
            try:
                narrow_data[0][idx][2] = round(elem[2], 2)
            except:
                pass
        chart = heatmap(data=narrow_data[0], width=width, height=height)
        chart.set_dict_options(
            {
                "xAxis": {"categories": narrow_data[1]},
                "yAxis": {"categories": narrow_data[2]},
                "colorAxis": {
                    "minColor": get_colors()[1],
                    "maxColor": get_colors()[0],
                    "min": -1,
                    "max": 1,
                },
            }
        )
        if kind != "cramer":
            chart.set_dict_options(
                {
                    "colorAxis": {
                        "stops": [
                            [0, get_colors()[1]],
                            [0.45, "#FFFFFF"],
                            [0.55, "#FFFFFF"],
                            [1, get_colors()[0]],
                        ]
                    }
                }
            )
        else:
            chart.set_dict_options(
                {
                    "colorAxis": {
                        "stops": [
                            [0, "#FFFFFF"],
                            [0.2, "#FFFFFF"],
                            [1, get_colors()[0]],
                        ],
                        "min": 0,
                    }
                }
            )
        chart.set_dict_options(options)
        return chart
