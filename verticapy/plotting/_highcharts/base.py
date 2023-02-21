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

from verticapy._utils._sql._execute import _executeSQL
from verticapy.connect import current_cursor

from verticapy.plotting._colors import gen_colors
from verticapy.plotting._highcharts.bar import bar
from verticapy.plotting._highcharts.boxplot import boxplot
from verticapy.plotting._highcharts.candlestick import candlestick
from verticapy.plotting._highcharts.drilldown_chart import drilldown_chart
from verticapy.plotting._highcharts.heatmap import heatmap
from verticapy.plotting._highcharts.line import line
from verticapy.plotting._highcharts.negative_bar import negative_bar
from verticapy.plotting._highcharts.pie import pie
from verticapy.plotting._highcharts.scatter import scatter
from verticapy.plotting._highcharts.spider import spider


def sort_classes(categories):
    try:
        try:
            order = []
            for item in categories:
                order += [float(item.split(";")[0].split("[")[1])]
        except:
            order = []
            for item in all_subcategories:
                order += [float(item)]
        order = [x for _, x in sorted(zip(order, categories))]
    except:
        return categories
    return order


def data_to_columns(data: list, n: int):
    columns = [[]] * n
    for elem in data:
        for i in range(n):
            try:
                columns[i] = columns[i] + [float(elem[i])]
            except:
                columns[i] = columns[i] + [elem[i]]
    return columns


def hchart_from_vdf(
    vdf,
    x=None,
    y=None,
    z=None,
    c=None,
    aggregate: bool = True,
    kind="boxplot",
    width: int = 600,
    height: int = 400,
    options: dict = {},
    h: float = -1,
    max_cardinality: int = 10,
    limit=10000,
    drilldown: bool = False,
    stock: bool = False,
    alpha: float = 0.25,
):
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
            {"colorAxis": {"maxColor": gen_colors()[0], "minColor": "#FFFFFF"}}
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
                    "minColor": gen_colors()[1],
                    "maxColor": gen_colors()[0],
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
                            [0, gen_colors()[1]],
                            [0.45, "#FFFFFF"],
                            [0.55, "#FFFFFF"],
                            [1, gen_colors()[0]],
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
                            [1, gen_colors()[0]],
                        ],
                        "min": 0,
                    }
                }
            )
        chart.set_dict_options(options)
        return chart


def hchartSQL(
    query: str, kind="auto", width: int = 600, height: int = 400, options: dict = {},
):
    from verticapy.core.vdataframe.base import vDataFrame

    aggregate, stock = False, False
    data = _executeSQL(
        query=f"""
            SELECT 
                /*+LABEL('highchart.hchartSQL')*/ * 
            FROM ({query}) VERTICAPY_SUBTABLE LIMIT 0""",
        method="fetchall",
        print_time_sql=False,
    )
    names = [desc[0] for desc in current_cursor().description]
    vdf = vDataFrame(query)
    allnum = vdf.numcol()
    if kind == "auto":
        if len(names) == 1:
            kind = "pie"
        elif (len(names) == len(allnum)) and (len(names) < 5):
            kind = "scatter"
        elif len(names) == 2:
            if vdf[names[0]].isdate() and vdf[names[1]].isnum():
                kind = "line"
            else:
                kind = "bar"
        elif len(names) == 3:
            if vdf[names[0]].isdate() and vdf[names[1]].isnum():
                kind = "line"
            elif vdf[names[2]].isnum():
                kind = "hist"
            else:
                kind = "boxplot"
        else:
            kind = "boxplot"
    if kind in (
        "pearson",
        "kendall",
        "cramer",
        "biserial",
        "spearman",
        "spearmand",
        "boxplot",
    ):
        x, y, z, c = allnum, None, None, None
    elif kind == "scatter":
        if len(names) < 2:
            raise ValueError("Scatter Plots need at least 2 columns.")
        x, y, z, c = names[0], names[1], None, None
        if len(names) == 3 and len(allnum) == 3:
            z = names[2]
        elif len(names) == 3:
            c = names[2]
        elif len(names) > 3:
            z, c = names[2], names[3]
    elif kind == "bubble":
        if len(names) < 3:
            raise ValueError("Bubble Plots need at least 3 columns.")
        x, y, z, c = names[0], names[1], names[2], None
        if len(names) > 3:
            c = names[3]
    elif kind in (
        "area",
        "area_ts",
        "spline",
        "line",
        "area_range",
        "spider",
        "candlestick",
    ):
        if vdf[names[0]].isdate():
            stock = True
        if len(names) < 2:
            raise ValueError(f"{kind} Plots need at least 2 columns.")
        x, y, z, c = names[0], names[1:], None, None
        if kind == "candlestick":
            aggregate = True
    else:
        if len(names) == 1:
            aggregate = True
            x, y, z, c = names[0], "COUNT(*) AS cnt", None, None
        else:
            x, y, z, c = names[0], names[1], None, None
        if len(names) > 2:
            z = names[2]
    return vdf.hchart(
        x=x,
        y=y,
        z=z,
        c=c,
        aggregate=aggregate,
        kind=kind,
        width=width,
        height=height,
        options=options,
        max_cardinality=100,
        stock=stock,
    )
