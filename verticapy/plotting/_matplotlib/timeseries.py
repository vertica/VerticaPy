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
# Standard Modules
import warnings

# MATPLOTLIB
import matplotlib.pyplot as plt

# VerticaPy Modules
from verticapy.utilities import *
from verticapy.plotting._matplotlib.core import updated_dict
from verticapy._config._notebook import ISNOTEBOOK
from verticapy.sql.read import _executeSQL
from verticapy.errors import ParameterError
from verticapy.plotting._colors import get_color, gen_colors
from verticapy.sql._utils._format import quote_ident

# Optional
try:
    from dateutil.parser import parse

    PARSER_IMPORT = True
except:
    PARSER_IMPORT = False


def parse_datetime(D: list):
    try:
        return [parse(d) for d in D]
    except:
        return copy.deepcopy(D)


def acf_plot(
    x: list,
    y: list,
    title="",
    confidence=None,
    type_bar: bool = True,
    ax=None,
    **style_kwds,
):
    tmp_style = {}
    for elem in style_kwds:
        if elem not in ("color", "colors"):
            tmp_style[elem] = style_kwds[elem]
    if "color" in style_kwds:
        color = style_kwds["color"]
    else:
        color = gen_colors()[0]
    if not (ax):
        fig, ax = plt.subplots()
        if ISNOTEBOOK:
            fig.set_size_inches(10, 3)
    if type_bar:
        ax.bar(x, y, width=0.007 * len(x), color="#444444", zorder=1, linewidth=0)
        param = {
            "s": 90,
            "marker": "o",
            "facecolors": color,
            "edgecolors": "black",
            "zorder": 2,
        }
        ax.scatter(
            x, y, **updated_dict(param, tmp_style),
        )
        ax.plot(
            [-1] + x + [x[-1] + 1],
            [0 for elem in range(len(x) + 2)],
            color=color,
            zorder=0,
        )
        ax.set_xlim(-1, x[-1] + 1)
    else:
        ax.plot(
            x, y, color=color, **tmp_style,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=90)
    if confidence:
        ax.fill_between(
            x, [-elem for elem in confidence], confidence, color=color, alpha=0.1
        )
    ax.set_xlabel("lag")
    return ax


def multi_ts_plot(
    vdf,
    order_by: str,
    columns: list = [],
    order_by_start: str = "",
    order_by_end: str = "",
    kind: str = "line",
    ax=None,
    **style_kwds,
):
    if len(columns) == 1 and kind != "area_percent":
        if kind in ("line", "step"):
            area = False
        else:
            area = True
        if kind == "step":
            step = True
        else:
            step = False
        return vdf[columns[0]].plot(
            ts=order_by,
            start_date=order_by_start,
            end_date=order_by_end,
            area=area,
            step=step,
            **style_kwds,
        )
    if not (columns):
        columns = vdf.numcol()
    for column in columns:
        if not (vdf[column].isnum()):
            if vdf._VERTICAPY_VARIABLES_["display"]["print_info"]:
                warning_message = (
                    f"The Virtual Column {column} is "
                    "not numerical.\nIt will be ignored."
                )
                warnings.warn(warning_message, Warning)
            columns.remove(column)
    if not (columns):
        raise EmptyParameter("No numerical columns found to draw the multi TS plot")
    colors = gen_colors()
    order_by_start_str, order_by_end_str = "", ""
    if order_by_start:
        order_by_start_str = f" AND {order_by} > '{order_by_start}'"
    if order_by_end:
        order_by_end_str = f" AND {order_by} < '{order_by_end}'"
    condition = " AND " + " AND ".join([f"{column} IS NOT NULL" for column in columns])
    query_result = _executeSQL(
        query=f"""
            SELECT 
                /*+LABEL('plotting._matplotlib.multi_ts_plot')*/ 
                {order_by}, 
                {", ".join(columns)} 
            FROM {vdf.__genSQL__()} 
            WHERE {order_by} IS NOT NULL
            {condition}
            ORDER BY {order_by}""",
        title="Selecting the needed points to draw the curves",
        method="fetchall",
    )
    order_by_values = [item[0] for item in query_result]
    if isinstance(order_by_values[0], str) and PARSER_IMPORT:
        order_by_values = parse_datetime(order_by_values)
    alpha = 0.3
    if not (ax):
        fig, ax = plt.subplots()
        if ISNOTEBOOK:
            fig.set_size_inches(8, 6)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    prec = [0 for item in query_result]
    for i in range(0, len(columns)):
        if kind == "area_percent":
            points = [
                sum(item[1 : i + 2]) / max(sum(item[1:]), 1e-70)
                for item in query_result
            ]
        elif kind == "area_stacked":
            points = [sum(item[1 : i + 2]) for item in query_result]
        else:
            points = [item[i + 1] for item in query_result]
        param = {"linewidth": 1}
        param_style = {
            "marker": "o",
            "markevery": 0.05,
            "markerfacecolor": colors[i],
            "markersize": 7,
        }
        if kind in ("line", "step"):
            color = colors[i]
            if len(order_by_values) < 20:
                param = {
                    **param_style,
                    "markeredgecolor": "black",
                }
            param["label"] = columns[i]
            param["linewidth"] = 2
        elif kind == "area_percent":
            color = "white"
            if len(order_by_values) < 20:
                param = {
                    **param_style,
                    "markeredgecolor": "white",
                }
        else:
            color = "black"
            if len(order_by_values) < 20:
                param = {
                    **param_style,
                    "markeredgecolor": "black",
                }
        param["color"] = color
        if "color" in style_kwds and len(order_by_values) < 20:
            param["markerfacecolor"] = get_color(style_kwds, i)
        if kind == "step":
            ax.step(order_by_values, points, **param)
        else:
            ax.plot(order_by_values, points, **param)
        if kind not in ("line", "step"):
            tmp_style = {}
            for elem in style_kwds:
                tmp_style[elem] = style_kwds[elem]
            if "color" not in tmp_style:
                tmp_style["color"] = colors[i]
            else:
                if not (isinstance(tmp_style["color"], str)):
                    tmp_style["color"] = tmp_style["color"][i % len(tmp_style["color"])]
            ax.fill_between(
                order_by_values, prec, points, label=columns[i], **tmp_style
            )
            prec = points
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    if kind == "area_percent":
        ax.set_ylim(0, 1)
    elif kind == "area_stacked":
        ax.set_ylim(0)
    ax.set_xlim(min(order_by_values), max(order_by_values))
    ax.set_xlabel(order_by)
    ax.legend(loc="center left", bbox_to_anchor=[1, 0.5])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    return ax


def range_curve(
    X,
    Y,
    param_name="",
    score_name="score",
    ax=None,
    labels=[],
    without_scatter: bool = False,
    plot_median: bool = True,
    **style_kwds,
):
    if not (ax):
        fig, ax = plt.subplots()
        if ISNOTEBOOK:
            fig.set_size_inches(8, 6)
        ax.grid()
    for i, y in enumerate(Y):
        if labels:
            label = labels[i]
        else:
            label = ""
        if plot_median:
            alpha1, alpha2 = 0.3, 0.5
        else:
            alpha1, alpha2 = 0.5, 0.9
        param = {"facecolor": get_color(style_kwds, i)}
        ax.fill_between(X, y[0], y[2], alpha=alpha1, **param)
        param = {"color": get_color(style_kwds, i)}
        for j in [0, 2]:
            ax.plot(
                X, y[j], alpha=alpha2, **updated_dict(param, style_kwds, i),
            )
        if plot_median:
            ax.plot(X, y[1], label=label, **updated_dict(param, style_kwds, i))
        if (not (without_scatter) or len(X) < 20) and plot_median:
            ax.scatter(
                X, y[1], c="white", marker="o", s=60, edgecolors="black", zorder=3,
            )
    ax.set_xlabel(param_name)
    ax.set_ylabel(score_name)
    if labels:
        ax.legend(loc="center left", bbox_to_anchor=[1, 0.5])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.set_xlim(X[0], X[-1])
    return ax


def range_curve_vdf(
    vdf,
    order_by: str,
    q: tuple = (0.25, 0.75),
    order_by_start: str = "",
    order_by_end: str = "",
    plot_median: bool = True,
    ax=None,
    **style_kwds,
):
    order_by_start_str, order_by_end_str = "", ""
    if order_by_start:
        order_by_start_str = f" AND {order_by} > '{order_by_start}'"
    if order_by_end:
        order_by_end_str = f" AND {order_by} < '{order_by_end}'"
    query_result = _executeSQL(
        query=f"""
        SELECT 
            /*+LABEL('plotting._matplotlib.range_curve_vdf')*/ 
            {order_by}, 
            APPROXIMATE_PERCENTILE({vdf.alias} USING PARAMETERS percentile = {q[0]}),
            APPROXIMATE_MEDIAN({vdf.alias}),
            APPROXIMATE_PERCENTILE({vdf.alias} USING PARAMETERS percentile = {q[1]})
        FROM {vdf.parent.__genSQL__()} 
        WHERE {order_by} IS NOT NULL 
          AND {vdf.alias} IS NOT NULL
          {order_by_start_str}
          {order_by_end_str}
        GROUP BY 1 ORDER BY 1""",
        title="Selecting points to draw the curve",
        method="fetchall",
    )
    order_by_values = [item[0] for item in query_result]
    if isinstance(order_by_values[0], str) and PARSER_IMPORT:
        order_by_values = parse_datetime(order_by_values)
    column_values = [
        [
            [float(item[1]) for item in query_result],
            [float(item[2]) for item in query_result],
            [float(item[3]) for item in query_result],
        ]
    ]
    return range_curve(
        order_by_values,
        column_values,
        order_by,
        vdf.alias,
        ax,
        [],
        True,
        plot_median,
        **style_kwds,
    )


def ts_plot(
    vdf,
    order_by: str,
    by: str = "",
    order_by_start: str = "",
    order_by_end: str = "",
    area: bool = False,
    step: bool = False,
    ax=None,
    **style_kwds,
):
    if order_by_start:
        order_by_start_str = f" AND {order_by} > '{order_by_start}'"
    else:
        order_by_start_str = ""
    if order_by_end:
        order_by_end_str = f" AND {order_by} < '{order_by_end}'"
    else:
        order_by_end_str = ""
    query = f"""
        SELECT 
            /*+LABEL('plotting._matplotlib.ts_plot')*/ 
            {order_by},
            {vdf.alias}
        FROM {vdf.parent.__genSQL__()}
        WHERE {order_by} IS NOT NULL 
          AND {vdf.alias} IS NOT NULL
          {order_by_start_str}
          {order_by_end_str}
          {{}}
        ORDER BY {order_by}, {vdf.alias}"""
    title = "Selecting points to draw the curve"
    if not (ax):
        fig, ax = plt.subplots()
        if ISNOTEBOOK:
            fig.set_size_inches(8, 6)
        ax.grid(axis="y")
    colors = gen_colors()
    plot_fun = ax.step if step else ax.plot
    plot_param = {
        "marker": "o",
        "markevery": 0.05,
        "markersize": 7,
        "markeredgecolor": "black",
    }
    if not (by):
        query_result = _executeSQL(
            query=query.format(""), title=title, method="fetchall",
        )
        order_by_values = [item[0] for item in query_result]
        if isinstance(order_by_values[0], str) and PARSER_IMPORT:
            order_by_values = parse_datetime(order_by_values)
        column_values = [float(item[1]) for item in query_result]
        param = {
            "color": colors[0],
            "linewidth": 2,
        }
        if len(order_by_values) < 20:
            param = {
                **plot_param,
                **param,
                "markerfacecolor": "white",
            }
        plot_fun(order_by_values, column_values, **updated_dict(param, style_kwds))
        if area and not (step):
            if "color" in updated_dict(param, style_kwds):
                color = updated_dict(param, style_kwds)["color"]
            ax.fill_between(order_by_values, column_values, facecolor=color, alpha=0.2)
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        ax.set_xlabel(order_by)
        ax.set_ylabel(vdf.alias)
        ax.set_xlim(min(order_by_values), max(order_by_values))
    else:
        by = quote_ident(by)
        cat = vdf.parent[by].distinct()
        all_data = []
        for column in cat:
            column_str = str(column).replace("'", "''")
            query_result = _executeSQL(
                query=query.format(f"AND {by} = '{column_str}'"),
                title=title,
                method="fetchall",
            )
            all_data += [
                [
                    [item[0] for item in query_result],
                    [float(item[1]) for item in query_result],
                    column,
                ]
            ]
            if isinstance(all_data[-1][0][0], str) and PARSER_IMPORT:
                all_data[-1][0] = parse_datetime(all_data[-1][0])
        for idx, d in enumerate(all_data):
            param = {"color": colors[idx % len(colors)]}
            if len(d[0]) < 20:
                param = {
                    **plot_param,
                    **param,
                    "markerfacecolor": colors[idx % len(colors)],
                }
            param["markerfacecolor"] = get_color(style_kwds, idx)
            plot_fun(d[0], d[1], label=d[2], **updated_dict(param, style_kwds, idx))
        ax.set_xlabel(order_by)
        ax.set_ylabel(vdf.alias)
        ax.legend(title=by, loc="center left", bbox_to_anchor=[1, 0.5])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
    return ax
