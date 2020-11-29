# (c) Copyright [2018-2020] Micro Focus or one of its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# |_     |~) _  _| _  /~\    _ |.
# |_)\/  |_)(_|(_||   \_/|_|(_|||
#    /
#              ____________       ______
#             / __        `\     /     /
#            |  \/         /    /     /
#            |______      /    /     /
#                   |____/    /     /
#          _____________     /     /
#          \           /    /     /
#           \         /    /     /
#            \_______/    /     /
#             ______     /     /
#             \    /    /     /
#              \  /    /     /
#               \/    /     /
#                    /     /
#                   /     /
#                   \    /
#                    \  /
#                     \/
#                    _
# \  / _  __|_. _ _ |_)
#  \/ (/_|  | |(_(_|| \/
#                     /
# VerticaPy is a Python library with scikit-like functionality to use to conduct
# data science projects on data stored in Vertica, taking advantage Vertica’s
# speed and built-in analytics and machine learning features. It supports the
# entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize
# data transformation operations, and offers beautiful graphical options.
#
# VerticaPy aims to solve all of these problems. The idea is simple: instead
# of moving data around for processing, VerticaPy brings the logic to the data.
#
#
# Modules
#
# Standard Python Modules
from collections.abc import Iterable

# High Chart
try:
    from highcharts import Highchart, Highstock
except:
    raise ImportError(
        "The highcharts module seems to not be installed in your environment.\nTo be able to use this method, you'll have to install it.\n[Tips] Run: 'pip3 install python-highcharts' in your terminal to install the module."
    )

# VerticaPy Modules
from verticapy.utilities import *
from verticapy.toolbox import *

#
##
#
#  ___  ___  ________  ___  ___  ________  ________  _________
# |\  \|\  \|\   ____\|\  \|\  \|\   __  \|\   __  \|\___   ___\
# \ \  \\\  \ \  \___|\ \  \\\  \ \  \|\  \ \  \|\  \|___ \  \_|
#  \ \   __  \ \  \    \ \   __  \ \   __  \ \   _  _\   \ \  \
#   \ \  \ \  \ \  \____\ \  \ \  \ \  \ \  \ \  \\  \|   \ \  \
#    \ \__\ \__\ \_______\ \__\ \__\ \__\ \__\ \__\\ _\    \ \__\
#     \|__|\|__|\|_______|\|__|\|__|\|__|\|__|\|__|\|__|    \|__|
#
##
#
# Functions used by vDataFrames to draw graphics using High Chart API.
#
# ---#
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
    x = (
        vdf_columns_names([x], vdf)[0]
        if (isinstance(x, str))
        else vdf_columns_names(x, vdf)
    )
    cursor = vdf._VERTICAPY_VARIABLES_["cursor"]
    if drilldown:
        if not (z):
            z = "COUNT(*)"
        if kind == "hist":
            kind = "column"
        check_types([("y", y, [str, list],)])
        if isinstance(x, Iterable) and not (isinstance(x, str)):
            x = x[0]
        columns_check([x], vdf)
        if isinstance(y, str):
            columns_check([y], vdf)
            y = vdf_columns_names([y], vdf)[0]
        else:
            columns_check(y, vdf)
            y = vdf_columns_names(y, vdf)[0]
        query = [
            "SELECT {}, {} FROM {} GROUP BY 1 LIMIT {}".format(
                x, z, vdf.__genSQL__(), limit
            ),
            "SELECT {}, {}, {} FROM {} GROUP BY 1, 2 LIMIT {}".format(
                x, y, z, vdf.__genSQL__(), limit
            ),
        ]
    elif (kind in ("pie_half", "pie", "donut", "pie3d", "donut3d")) or (
        kind in ("bar", "hist") and not (z)
    ):
        if not (y):
            y = "COUNT(*)"
        if isinstance(x, Iterable) and not (isinstance(x, str)):
            x = x[0]
        columns_check([x], vdf)
        unique = vdf[x].nunique()
        is_num = vdf[x].isnum()
        order_by = " ORDER BY 2 DESC "
        if unique > max_cardinality:
            if is_num:
                order_by = " ORDER BY MIN({}) DESC ".format(x)
                x = vdf[x].discretize(h=h, return_enum_trans=True)[0].replace(
                    "{}", x
                ) + " AS {}".format(x)
            else:
                query = "SELECT {}, {} FROM {} GROUP BY 1 ORDER BY 2 DESC LIMIT {}".format(
                    x, y, vdf.__genSQL__(), max_cardinality
                )
                vdf._VERTICAPY_VARIABLES_["cursor"].execute(query)
                result = vdf._VERTICAPY_VARIABLES_["cursor"].fetchall()
                result = [elem[0] for elem in result]
                result = [
                    "NULL" if elem == None else "'{}'".format(elem) for elem in result
                ]
                x = "(CASE WHEN {} IN ({}) THEN {} ELSE 'Others' END) AS {}".format(
                    x, ", ".join(result), x, x
                )
        query = "SELECT {}, {} FROM {}{}{}LIMIT {}".format(
            x,
            y,
            vdf.__genSQL__(),
            " GROUP BY 1" if (aggregate) else "",
            order_by,
            limit,
        )
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
            check_types([("y", y, [str, list],)])
            if isinstance(x, Iterable) and not (isinstance(x, str)):
                x = x[0]
            columns_check([x], vdf)
            if isinstance(y, str):
                columns_check([y], vdf)
                y = vdf_columns_names([y], vdf)[0]
            else:
                columns_check(y, vdf)
                y = vdf_columns_names(y, vdf)[0]
            # y
            unique = vdf[y].nunique()
            is_num = vdf[y].isnum()
            if unique > max_cardinality:
                if is_num:
                    where = " WHERE {} IS NOT NULL ".format(y)
                    y = vdf[y].discretize(h=h, return_enum_trans=True)[0].replace(
                        "{}", y
                    ) + " AS {}".format(y)
                else:
                    y = vdf[y].discretize(
                        k=max_cardinality, method="topk", return_enum_trans=True
                    )[0].replace("{}", y) + " AS {}".format(y)
        # x
        unique = vdf[x].nunique()
        is_num = vdf[x].isnum()
        if unique > max_cardinality:
            if is_num:
                x = vdf[x].discretize(h=h, return_enum_trans=True)[0].replace(
                    "{}", x
                ) + " AS {}".format(x)
            else:
                x = vdf[x].discretize(
                    k=max_cardinality, method="topk", return_enum_trans=True
                )[0].replace("{}", x) + " AS {}".format(x)
        query = "SELECT {}, {}, {} FROM {} {}{}LIMIT {}".format(
            x,
            y,
            z,
            vdf.__genSQL__(),
            where,
            " GROUP BY 1, 2 " if (aggregate) else "",
            limit,
        )
    elif kind in ("area", "area_ts", "line", "spline"):
        check_types([("y", y, [str, list],)])
        if isinstance(x, Iterable) and not (isinstance(x, str)):
            x = x[0]
        columns_check([x], vdf)
        cast = "::timestamp" if (vdf[x].isdate()) else ""
        if not (z):
            if not (aggregate):
                if isinstance(y, str):
                    columns_check([y], vdf)
                    y = vdf_columns_names([y], vdf)[0]
                else:
                    columns_check(y, vdf)
                    y = vdf_columns_names(y, vdf)
            if not (isinstance(y, str)):
                y = ", ".join(y)
                kind = "multi_" + kind
            query = "SELECT {}{}, {} FROM {} WHERE {} IS NOT NULL{}{} LIMIT {}".format(
                x,
                cast,
                y,
                vdf.__genSQL__(),
                x,
                " GROUP BY 1 " if (aggregate) else "",
                " ORDER BY 1 " if (vdf[x].isdate() or vdf[x].isnum()) else "",
                limit,
            )
        else:
            check_types([("y", y, [str, list],)])
            check_types([("z", z, [str, list],)])
            if isinstance(y, str):
                columns_check([y], vdf)
                y = vdf_columns_names([y], vdf)[0]
            else:
                columns_check(y, vdf)
                y = vdf_columns_names(y, vdf)[0]
            if isinstance(z, str):
                columns_check([z], vdf)
                z = vdf_columns_names([z], vdf)[0]
            else:
                columns_check(z, vdf)
                z = vdf_columns_names(z, vdf)[0]
            # z
            unique = vdf[z].nunique()
            is_num = vdf[z].isnum()
            z_copy = z
            if unique > max_cardinality:
                if is_num:
                    z = vdf[z].discretize(h=h, return_enum_trans=True)[0].replace(
                        "{}", z
                    ) + " AS {}".format(z)
                else:
                    z = vdf[z].discretize(
                        k=max_cardinality, method="topk", return_enum_trans=True
                    )[0].replace("{}", z) + " AS {}".format(z)
            query = "SELECT {}{}, {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL LIMIT {} OVER (PARTITION BY {} ORDER BY {} DESC)".format(
                x,
                cast,
                y,
                z,
                vdf.__genSQL__(),
                x,
                y,
                max(int(limit / unique), 1),
                z_copy,
                x,
            )
    elif kind in ("scatter", "bubble"):
        check_types([("y", y, [str, list],)])
        if isinstance(y, str):
            columns_check([y], vdf)
            y = vdf_columns_names([y], vdf)[0]
        else:
            columns_check(y, vdf)
            y = vdf_columns_names(y, vdf)[0]
        if isinstance(x, Iterable) and not (isinstance(x, str)):
            x = x[0]
        cast = "::timestamp" if (vdf[x].isdate()) else ""
        if not (z) and not (c) and (kind == "scatter"):
            query = "SELECT {}{}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL LIMIT {}".format(
                x, cast, y, vdf.__genSQL__(), x, y, limit
            )
        elif not (c) and (z):
            check_types([("z", z, [str, list],)])
            try:
                z = (
                    vdf_columns_names([z], vdf)[0]
                    if (isinstance(z, str))
                    else vdf_columns_names(z, vdf)[0]
                )
            except:
                pass
            query = "SELECT {}{}, {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL LIMIT {}".format(
                x, cast, y, z, vdf.__genSQL__(), x, y, z, limit
            )
        else:
            if z:
                check_types([("z", z, [str, list],)])
                try:
                    z = (
                        vdf_columns_names([z], vdf)[0]
                        if (isinstance(z, str))
                        else vdf_columns_names(z, vdf)[0]
                    )
                except:
                    pass
            check_types([("c", c, [str, list],)])
            try:
                c = (
                    vdf_columns_names([c], vdf)[0]
                    if (isinstance(c, str))
                    else vdf_columns_names(c, vdf)[0]
                )
            except:
                pass
            # c
            unique = vdf[c].nunique()
            is_num = vdf[c].isnum()
            c_copy = c
            if unique > max_cardinality:
                if is_num:
                    c = vdf[c].discretize(h=h, return_enum_trans=True)[0].replace(
                        "{}", c
                    ) + " AS {}".format(c)
                else:
                    c = vdf[c].discretize(
                        k=max_cardinality, method="topk", return_enum_trans=True
                    )[0].replace("{}", c) + " AS {}".format(c)
            query = "SELECT {}{}, {}{}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL{}LIMIT {} OVER (PARTITION BY {} ORDER BY {})".format(
                x,
                cast,
                y,
                ", " + str(z) if z else "",
                c,
                vdf.__genSQL__(),
                x,
                y,
                " AND {} IS NOT NULL ".format(z) if z else " ",
                max(int(limit / unique), 1),
                c_copy,
                c_copy,
            )
    elif kind == "area_range":
        check_types([("y", y, [str, list],)])
        if isinstance(x, Iterable) and not (isinstance(x, str)):
            x = x[0]
        columns_check([x], vdf)
        order_by = " ORDER BY 1 " if (vdf[x].isdate() or vdf[x].isnum()) else ""
        cast = "::timestamp" if (vdf[x].isdate()) else ""
        query = "SELECT {}{}, {} FROM {}{}{} LIMIT {}".format(
            x,
            cast,
            ", ".join(y),
            vdf.__genSQL__(),
            " GROUP BY 1 " if (aggregate) else "",
            order_by,
            limit,
        )
    elif kind == "spider":
        if not (y):
            y = "COUNT(*)"
        check_types([("y", y, [str, list],)])
        if isinstance(y, str):
            y = [y]
        if isinstance(x, Iterable) and not (isinstance(x, str)):
            x = x[0]
        columns_check([x], vdf)
        # x
        unique = vdf[x].nunique()
        is_num = vdf[x].isnum()
        if unique > max_cardinality:
            if is_num:
                x = vdf[x].discretize(h=h, return_enum_trans=True)[0].replace(
                    "{}", x
                ) + " AS {}".format(x)
            else:
                if len(y) == 1:
                    query = "SELECT {}, {} FROM {} GROUP BY 1 ORDER BY 2 DESC LIMIT {}".format(
                        x, y[0], vdf.__genSQL__(), max_cardinality
                    )
                    vdf._VERTICAPY_VARIABLES_["cursor"].execute(query)
                    result = vdf._VERTICAPY_VARIABLES_["cursor"].fetchall()
                    result = [elem[0] for elem in result]
                    result = [
                        "NULL" if elem == None else "'{}'".format(elem)
                        for elem in result
                    ]
                    x = "(CASE WHEN {} IN ({}) THEN {} ELSE 'Others' END) AS {}".format(
                        x, ", ".join(result), x, x
                    )
                else:
                    x = vdf[x].discretize(
                        k=max_cardinality, method="topk", return_enum_trans=True
                    )[0].replace("{}", x) + " AS {}".format(x)
        query = "SELECT {}, {} FROM {}{} LIMIT {}".format(
            x,
            ", ".join(y),
            vdf.__genSQL__(),
            " GROUP BY 1 " if (aggregate) else "",
            limit,
        )
    elif kind == "candlestick":
        if isinstance(x, Iterable) and not (isinstance(x, str)):
            x = x[0]
        columns_check([x], vdf)
        if aggregate:
            if isinstance(y, str):
                query = """SELECT {}::timestamp, 
			                      APPROXIMATE_PERCENTILE({} USING PARAMETERS percentile = {}) AS open,
			                      MAX({}) AS high,
			                      MIN({}) AS low,
			                      APPROXIMATE_PERCENTILE({} USING PARAMETERS percentile = {}) AS close,
			                      SUM({}) AS volume
		                	FROM {} GROUP BY 1 ORDER BY 1"""
                query = query.format(
                    x, y, 1 - alpha, y, y, y, alpha, y, vdf.__genSQL__()
                )
            else:
                check_types([("y", y, [list],)])
                query = "SELECT {}::timestamp, {} FROM {} GROUP BY 1 ORDER BY 1".format(
                    x, ", ".join(y), vdf.__genSQL__()
                )
        else:
            query = "SELECT {}::timestamp, {} FROM {} ORDER BY 1".format(
                x, ", ".join(y), vdf.__genSQL__()
            )
    if drilldown:
        return drilldown_chart(
            query=query,
            cursor=cursor,
            options=options,
            width=width,
            height=height,
            chart_type=kind,
        )
    elif kind == "candlestick":
        return candlestick(
            query=query, cursor=cursor, options=options, width=width, height=height
        )
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
            cursor=cursor,
            options=options,
            width=width,
            height=height,
            chart_type=kind,
            stock=stock,
        )
    elif kind in ("bar", "hist", "stacked_bar", "stacked_hist"):
        return bar(
            query=query,
            cursor=cursor,
            options=options,
            width=width,
            height=height,
            chart_type=kind,
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
            query=query,
            cursor=cursor,
            options=options,
            width=width,
            height=height,
            chart_type=kind,
        )
    elif kind in ("pie", "pie_half", "donut", "pie3d", "donut3d"):
        if kind == "pie_half":
            kind = "half"
        return pie(
            query=query,
            cursor=cursor,
            options=options,
            width=width,
            height=height,
            chart_type=kind,
        )
    elif kind == "heatmap":
        chart = heatmap(query=query, cursor=cursor, width=width, height=height)
        chart.set_dict_options(
            {"colorAxis": {"maxColor": "#FE5016", "minColor": "#FFFFFF"}}
        )
        chart.set_dict_options(options)
        return chart
    elif kind == "negative_bar":
        return negative_bar(
            query=query, cursor=cursor, options=options, width=width, height=height
        )
    elif kind == "spider":
        return spider(
            query=query, cursor=cursor, options=options, width=width, height=height
        )
    elif kind in ("pearson", "kendall", "cramer", "biserial", "spearman"):
        check_types([("x", x, [list],)])
        x = vdf_columns_names(x, vdf)
        data = vdf.corr(method=kind, show=False, columns=x)
        narrow_data = get_narrow_tablesample(data, use_number_as_category=True)
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
                    "minColor": "#0073E7",
                    "maxColor": "#FE5016",
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
                            [0, "#0073E7"],
                            [0.45, "#FFFFFF"],
                            [0.55, "#FFFFFF"],
                            [1, "#FE5016"],
                        ]
                    }
                }
            )
        else:
            chart.set_dict_options(
                {
                    "colorAxis": {
                        "stops": [[0, "#FFFFFF"], [0.2, "#FFFFFF"], [1, "#FE5016"]],
                        "min": 0,
                    }
                }
            )
        chart.set_dict_options(options)
        return chart


# ---#
def hchartSQL(
    query: str,
    cursor,
    kind="auto",
    width: int = 600,
    height: int = 400,
    options: dict = {},
):
    aggregate, stock = False, False
    cursor.execute("SELECT * FROM ({}) VERTICAPY_SUBTABLE LIMIT 0".format(query))
    data = cursor.fetchall()
    names = [desc[0] for desc in cursor.description]
    vdf = vdf_from_relation("({}) VERTICAPY_SUBTABLE".format(query), cursor=cursor)
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
    if kind in ("pearson", "kendall", "cramer", "biserial", "spearman", "boxplot"):
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
            raise ValueError("{} Plots need at least 2 columns.".format(kind))
        x, y, z, c = names[0], names[1:], None, None
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


# ---#
#####
#####
#####
#####
# ---#
def bar(
    query: str,
    cursor,
    options: dict = {},
    width: int = 600,
    height: int = 400,
    chart_type="regular",
):
    is_stacked = "stacked" in chart_type
    if chart_type == "stacked_hist":
        chart_type = "hist"
    if chart_type == "stacked_bar":
        chart_type = "bar"
    cursor.execute(query)
    data = cursor.fetchall()
    names = [desc[0] for desc in cursor.description]
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
    default_options["colors"] = [
        "#263133",
        "#FE5016",
        "#0073E7",
        "#19A26B",
        "#FCDB1F",
        "#000000",
        "#2A6A74",
        "#861889",
        "#00B4E0",
        "#90EE90",
        "#FF7F50",
        "#B03A89",
    ]
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
        all_categories = order_discretized_classes(list(set(columns[0])))
        all_subcategories = order_discretized_classes(list(set(columns[1])))
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


# ---#
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
    default_options["colors"] = [
        "#263133",
        "#FE5016",
        "#0073E7",
        "#19A26B",
        "#FCDB1F",
        "#000000",
        "#2A6A74",
        "#861889",
        "#00B4E0",
        "#90EE90",
        "#FF7F50",
        "#B03A89",
    ]
    chart.set_dict_options(default_options)
    aggregations = ["min", "25%", "50%", "75%", "max"]
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
                vdf.search("{} = '{}'".format(by, elem), usecols=[columns[0], by])[
                    columns[0]
                ]
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


# ---#
def candlestick(
    query: str, cursor, options: dict = {}, width: int = 600, height: int = 400
):
    cursor.execute(query)
    data = cursor.fetchall()
    names = [desc[0] for desc in cursor.description]
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
    default_options["colors"] = [
        "#263133",
        "#FE5016",
        "#0073E7",
        "#19A26B",
        "#FCDB1F",
        "#000000",
        "#2A6A74",
        "#861889",
        "#00B4E0",
        "#90EE90",
        "#FF7F50",
        "#B03A89",
    ]
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


# ---#
def drilldown_chart(
    query: list,
    cursor,
    options: dict = {},
    width: int = 600,
    height: int = 400,
    chart_type="column",
):
    cursor.execute(query[0])
    data = cursor.fetchall()
    names = [desc[0] for desc in cursor.description]
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
    default_options["colors"] = [
        "#263133",
        "#FE5016",
        "#0073E7",
        "#19A26B",
        "#FCDB1F",
        "#000000",
        "#2A6A74",
        "#861889",
        "#00B4E0",
        "#90EE90",
        "#FF7F50",
        "#B03A89",
    ]
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
    cursor.execute(query[1])
    data = cursor.fetchall()
    names = [desc[0] for desc in cursor.description]
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


# ---#
def heatmap(
    query: str = "",
    cursor=None,
    data: list = [],
    options: dict = {},
    width: int = 600,
    height: int = 400,
):
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
        "colorAxis": {"minColor": "#FFFFFF", "maxColor": "#FE5016"},
        "xAxis": {"title": {"text": ""}},
        "yAxis": {"title": {"text": ""}},
        "tooltip": {
            "formatter": "function () {return '<b>[' + this.series.xAxis.categories[this.point.x] + ', ' + this.series.yAxis.categories[this.point.y] + ']</b>: ' + this.point.value + '</b>';}"
        },
    }
    default_options["colors"] = [
        "#263133",
        "#FE5016",
        "#0073E7",
        "#19A26B",
        "#FCDB1F",
        "#000000",
        "#2A6A74",
        "#861889",
        "#00B4E0",
        "#90EE90",
        "#FF7F50",
        "#B03A89",
    ]
    chart.set_dict_options(default_options)
    if query and cursor:
        cursor.execute(query)
        data = cursor.fetchall()
        names = [desc[0] for desc in cursor.description]
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
                "xAxis": {"categories": all_categories, "title": {"text": names[0]}},
                "yAxis": {"categories": all_subcategories, "title": {"text": names[1]}},
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


# ---#
def line(
    query: str,
    cursor,
    options: dict = {},
    width: int = 600,
    height: int = 400,
    chart_type="line",
    stock: bool = False,
):
    is_ts = True if (chart_type == "area_ts") else False
    is_range = True if (chart_type == "area_range") else False
    is_date = False
    is_multi = True if ("multi" in chart_type) else False
    if chart_type in ("area_ts", "area_range", "multi_area"):
        chart_type = "area"
    if chart_type in ("multi_line"):
        chart_type = "line"
    if chart_type in ("multi_spline"):
        chart_type = "spline"
    cursor.execute(query)
    data = cursor.fetchall()
    names = [desc[0] for desc in cursor.description]
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
                            "hover": {"enabled": True, "lineColor": "rgb(100,100,100)"}
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
    default_options["colors"] = [
        "#263133",
        "#FE5016",
        "#0073E7",
        "#19A26B",
        "#FCDB1F",
        "#000000",
        "#2A6A74",
        "#861889",
        "#00B4E0",
        "#90EE90",
        "#FF7F50",
        "#B03A89",
    ]
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
                            "formatter": "function() {return Highcharts.dateFormat('%a %d %b', this.value);}"
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
                        "stops": [[0, "#FFFFFF"], [1, "#FE5016"]],
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


# ---#
def negative_bar(
    query: str, cursor, options: dict = {}, width: int = 600, height: int = 400
):
    cursor.execute(query)
    data = cursor.fetchall()
    names = [desc[0] for desc in cursor.description]
    n = len(names)
    chart = Highchart(width=width, height=height)
    columns = data_to_columns(data, n)
    all_categories = list(set(columns[0]))
    all_subcategories = order_discretized_classes(list(set(columns[1])))
    default_options = {
        "chart": {"type": "bar"},
        "title": {"text": ""},
        "subtitle": {"text": ""},
        "xAxis": [
            {"categories": all_subcategories, "reversed": False, "labels": {"step": 1}},
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
    default_options["colors"] = [
        "#263133",
        "#FE5016",
        "#0073E7",
        "#19A26B",
        "#FCDB1F",
        "#000000",
        "#2A6A74",
        "#861889",
        "#00B4E0",
        "#90EE90",
        "#FF7F50",
        "#B03A89",
    ]
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


# ---#
def pie(
    query: str,
    cursor,
    options: dict = {},
    width: int = 600,
    height: int = 400,
    chart_type="regular",
):
    cursor.execute(query)
    data = cursor.fetchall()
    names = [desc[0] for desc in cursor.description]
    n = len(names)
    chart = Highchart(width=width, height=height)
    default_options = {
        "title": {"text": ""},
        "chart": {"inverted": True},
        "xAxis": {
            "reversed": False,
            "title": {"text": names[0], "enabled": True},
            "maxPadding": 0.05,
            "showLastLabel": True,
        },
        "yAxis": {"title": {"text": names[1], "enabled": True}},
        "plotOptions": {
            "pie": {
                "allowPointSelect": True,
                "cursor": "pointer",
                "showInLegend": True,
                "size": "110%",
            }
        },
        "tooltip": {"pointFormat": str(names[1]) + ": <b>{point.y}</b>"},
    }
    if "3d" not in chart_type:
        default_options["colors"] = [
            "#263133",
            "#FE5016",
            "#0073E7",
            "#19A26B",
            "#FCDB1F",
            "#000000",
            "#2A6A74",
            "#861889",
            "#00B4E0",
            "#90EE90",
            "#FF7F50",
            "#B03A89",
        ]
    chart.set_dict_options(default_options)
    if "3d" in chart_type:
        chart.set_dict_options(
            {"chart": {"type": "pie", "options3d": {"enabled": True, "alpha": 45}}}
        )
        chart.add_JSsource("https://code.highcharts.com/6/highcharts-3d.js")
        chart_type = chart_type.replace("3d", "")
    data_pie = []
    for elem in data:
        try:
            val = float(elem[1])
        except:
            val = elem[1]
        try:
            key = float(elem[0])
        except:
            key = elem[0]
            if key == None:
                key = "None"
        data_pie += [{"name": key, "y": val}]
    data_pie[-1]["sliced"], data_pie[-1]["selected"] = True, True
    chart.add_data_set(data_pie, "pie")
    if chart_type == "half":
        chart.set_dict_options(
            {
                "plotOptions": {"pie": {"startAngle": -90, "endAngle": 90}},
                "legend": {"enabled": False},
            }
        )
    elif chart_type == "donut":
        chart.set_dict_options(
            {
                "chart": {"type": "pie"},
                "plotOptions": {"pie": {"innerSize": 100, "depth": 45}},
            }
        )
    chart.set_dict_options(options)
    return chart


# ---#
def scatter(
    query: str,
    cursor,
    options: dict = {},
    width: int = 600,
    height: int = 400,
    chart_type="regular",
):
    cursor.execute(query)
    data = cursor.fetchall()
    names = [desc[0] for desc in cursor.description]
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
        default_options["colors"] = [
            "#263133",
            "#FE5016",
            "#0073E7",
            "#19A26B",
            "#FCDB1F",
            "#000000",
            "#2A6A74",
            "#861889",
            "#00B4E0",
            "#90EE90",
            "#FF7F50",
            "#B03A89",
        ]
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


# ---#
def spider(query: str, cursor, options: dict = {}, width: int = 600, height: int = 400):
    cursor.execute(query)
    data = cursor.fetchall()
    names = [desc[0] for desc in cursor.description]
    n = len(names)
    chart = Highchart(width=width, height=height)
    default_options = {
        "chart": {"polar": True, "type": "line", "renderTo": "test"},
        "title": {"text": "", "x": -80},
        "pane": {"size": "80%"},
        "xAxis": {"tickmarkPlacement": "on", "lineWidth": 0},
        "yAxis": {"gridLineInterpolation": "polygon", "lineWidth": 0, "min": 0},
        "tooltip": {
            "shared": True,
            "pointFormat": '<span style="color:{series.color}">{series.name}: <b>{point.y:,.0f}</b><br/>',
        },
        "legend": {
            "align": "right",
            "verticalAlign": "top",
            "y": 70,
            "layout": "vertical",
        },
    }
    default_options["colors"] = [
        "#263133",
        "#FE5016",
        "#0073E7",
        "#19A26B",
        "#FCDB1F",
        "#000000",
        "#2A6A74",
        "#861889",
        "#00B4E0",
        "#90EE90",
        "#FF7F50",
        "#B03A89",
    ]
    chart.set_dict_options(default_options)
    columns = data_to_columns(data, n)
    chart.set_dict_options({"xAxis": {"categories": columns[0]}})
    for i in range(1, n):
        chart.add_data_set(columns[i], name=names[i], pointPlacement="on")
    chart.set_dict_options(options)
    return chart
