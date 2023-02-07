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
# Standard modules
import warnings

# MATPLOTLIB
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# NUMPY
import numpy as np

# VerticaPy Modules
from verticapy.utilities import *
from verticapy.utils._toolbox import executeSQL, quote_ident
from verticapy.errors import ParameterError
from verticapy.plotting._colors import gen_cmap, gen_colors

# IPython - Optional
try:
    from IPython.display import HTML
except:
    pass

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


def animated_bar(
    vdf,
    columns: list,
    order_by: str,
    by: str = "",
    order_by_start: str = "",
    order_by_end: str = "",
    limit_over: int = 6,
    limit: int = 1000000,
    fixed_xy_lim: bool = False,
    date_in_title: bool = False,
    date_f=None,
    date_style_dict: dict = {},
    interval: int = 10,
    repeat: bool = True,
    return_html: bool = True,
    pie: bool = False,
    ax=None,
    **style_kwds,
):
    if not (date_style_dict):
        date_style_dict = {
            "fontsize": 50,
            "alpha": 0.6,
            "color": "gray",
            "ha": "right",
            "va": "center",
        }
    if by:
        columns += [by]
    if date_f == None:

        def date_f(x):
            return str(x)

    colors = gen_colors()
    for c in ["color", "colors"]:
        if c in style_kwds:
            colors = style_kwds[c]
            del style_kwds[c]
            break
    if isinstance(colors, str):
        colors = []
    colors_map = {}
    idx = 2 if len(columns) >= 3 else 0
    all_cats = vdf[columns[idx]].distinct(agg=f"MAX({columns[1]})")
    for idx, i in enumerate(all_cats):
        colors_map[i] = colors[idx % len(colors)]
    order_by_start_str, order_by_end_str = "", ""
    if order_by_start:
        order_by_start_str = f"AND {order_by} > '{order_by_start}'"
    if order_by_end:
        order_by_end_str = f" AND {order_by} < '{order_by_end}'"
    condition = " AND ".join([f"{column} IS NOT NULL" for column in columns])
    query_result = executeSQL(
        query=f"""
            SELECT 
                /*+LABEL('plotting._matplotlib.animated_bar')*/ * 
            FROM 
                (SELECT 
                    {order_by},
                    {", ".join(columns)} 
                 FROM {vdf.__genSQL__()} 
                 WHERE {order_by} IS NOT NULL 
                   AND {condition}
                   {order_by_start_str}
                   {order_by_end_str} 
                 LIMIT {limit_over} 
                    OVER (PARTITION BY {order_by} 
                          ORDER BY {columns[1]} DESC)) x 
            ORDER BY {order_by} ASC, {columns[1]} ASC 
            LIMIT {limit}""",
        title="Selecting points to draw the animated bar chart.",
        method="fetchall",
    )
    order_by_values = [item[0] for item in query_result]
    column1 = [item[1] for item in query_result]
    column2 = [float(item[2]) for item in query_result]
    column3 = []
    if len(columns) >= 3:
        column3 = [item[3] for item in query_result]
        color = [colors_map[item[3]] for item in query_result]
    else:
        color = [colors_map[item[1]] for item in query_result]
    current_ts, ts_idx = order_by_values[0], 0
    bar_values = []
    n = len(order_by_values)
    for idx, elem in enumerate(order_by_values):
        if elem != current_ts or idx == n - 1:
            bar_values += [
                {
                    "y": column1[ts_idx:idx],
                    "width": column2[ts_idx:idx],
                    "c": color[ts_idx:idx],
                    "x": column3[ts_idx:idx],
                    "date": current_ts,
                }
            ]
            current_ts, ts_idx = elem, idx
    if not (ax):
        fig, ax = plt.subplots()
        if isnotebook():
            if pie:
                fig.set_size_inches(11, min(limit_over, 600))
            else:
                fig.set_size_inches(9, 6)
        ax.xaxis.grid()
        ax.set_axisbelow(True)

    def animate(i):
        ax.clear()
        if not (pie):
            ax.xaxis.grid()
            ax.set_axisbelow(True)
            min_x, max_x = min(bar_values[i]["width"]), max(bar_values[i]["width"])
            delta_x = max_x - min_x
            ax.barh(
                y=bar_values[i]["y"],
                width=bar_values[i]["width"],
                color=bar_values[i]["c"],
                alpha=0.6,
                **style_kwds,
            )
            if bar_values[i]["width"][0] > 0:
                ax.barh(
                    y=bar_values[i]["y"],
                    width=[-0.3 * delta_x for elem in bar_values[i]["y"]],
                    color=bar_values[i]["c"],
                    alpha=0.6,
                    **style_kwds,
                )
            if fixed_xy_lim:
                ax.set_xlim(min(column2), max(column2))
            else:
                ax.set_xlim(min_x - 0.3 * delta_x, max_x + 0.3 * delta_x)
            all_text = []
            for k in range(len(bar_values[i]["y"])):
                tmp_txt = []
                tmp_txt += [
                    ax.text(
                        bar_values[i]["width"][k],
                        k + 0.1,
                        bar_values[i]["y"][k],
                        ha="right",
                        fontweight="bold",
                        size=10,
                    )
                ]
                width_format = bar_values[i]["width"][k]
                if width_format - int(width_format) == 0:
                    width_format = int(width_format)
                width_format = f"{width_format:}"
                tmp_txt += [
                    ax.text(
                        bar_values[i]["width"][k] + 0.005 * delta_x,
                        k - 0.15,
                        width_format,
                        ha="left",
                        size=10,
                    )
                ]
                if len(columns) >= 3:
                    tmp_txt += [
                        ax.text(
                            bar_values[i]["width"][k],
                            k - 0.3,
                            bar_values[i]["x"][k],
                            ha="right",
                            size=10,
                            color="#333333",
                        )
                    ]
                all_text += [tmp_txt]
            if date_in_title:
                ax.set_title(date_f(bar_values[i]["date"]))
            else:
                my_text = ax.text(
                    max_x + 0.27 * delta_x,
                    int(limit_over / 2),
                    date_f(bar_values[i]["date"]),
                    **date_style_dict,
                )
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            ax.set_xlabel(columns[1])
            ax.set_yticks([])
        else:
            param = {
                "wedgeprops": {"edgecolor": "white", "alpha": 0.5},
                "textprops": {"fontsize": 10, "fontweight": "bold"},
                "autopct": "%1.1f%%",
            }

            def autopct(val):
                a = val / 100.0 * sum(bar_values[i]["width"])
                return f"{a:}"

            pie_chart = ax.pie(
                x=bar_values[i]["width"],
                labels=bar_values[i]["y"],
                colors=bar_values[i]["c"],
                **updated_dict(param, style_kwds),
            )
            for elem in pie_chart[2]:
                elem.set_fontweight("normal")
            if date_in_title:
                ax.set_title(date_f(bar_values[i]["date"]))
            else:
                my_text = ax.text(
                    1.8, 1, date_f(bar_values[i]["date"]), **date_style_dict
                )
            all_categories = []
            custom_lines = []
            if len(columns) >= 3:
                for idx, elem in enumerate(bar_values[i]["x"]):
                    if elem not in all_categories:
                        all_categories += [elem]
                        custom_lines += [
                            Line2D(
                                [0],
                                [0],
                                color=bar_values[i]["c"][idx],
                                lw=6,
                                alpha=updated_dict(param, style_kwds)["wedgeprops"][
                                    "alpha"
                                ],
                            )
                        ]
                leg = ax.legend(
                    custom_lines,
                    all_categories,
                    title=by,
                    loc="center left",
                    bbox_to_anchor=[1, 0.5],
                )
        return (ax,)

    myAnimation = animation.FuncAnimation(
        fig,
        animate,
        frames=range(0, len(bar_values)),
        interval=interval,
        blit=False,
        repeat=repeat,
    )
    if isnotebook() and return_html:
        anim = myAnimation.to_jshtml()
        plt.close("all")
        return HTML(anim)
    else:
        return myAnimation


def animated_bubble_plot(
    vdf,
    order_by: str,
    columns: list,
    by: str = "",
    label_name: str = "",
    order_by_start: str = "",
    order_by_end: str = "",
    limit_over: int = 10,
    limit: int = 1000000,
    lim_labels: int = 6,
    fixed_xy_lim: bool = False,
    bbox: list = [],
    img: str = "",
    date_in_title: bool = False,
    date_f=None,
    date_style_dict: dict = {},
    interval: int = 10,
    repeat: bool = True,
    return_html: bool = True,
    ax=None,
    **style_kwds,
):
    if not (date_style_dict):
        date_style_dict = {
            "fontsize": 100,
            "alpha": 0.6,
            "color": "gray",
            "ha": "center",
            "va": "center",
        }
    if date_f == None:

        def date_f(x):
            return str(x)

    if len(columns) == 2:
        columns += [1]
    if "color" in style_kwds:
        colors = style_kwds["color"]
    elif "colors" in style_kwds:
        colors = style_kwds["colors"]
    else:
        colors = gen_colors()
    if isinstance(colors, str):
        colors = [colors]
    param = {
        "alpha": 0.8,
        "edgecolors": "black",
    }
    if by:
        if vdf[by].isnum():
            param = {
                "alpha": 0.8,
                "cmap": gen_cmap()[0],
                "edgecolors": "black",
            }
        else:
            colors_map = {}
            all_cats = vdf[by].distinct(agg=f"MAX({columns[2]})")
            for idx, elem in enumerate(all_cats):
                colors_map[elem] = colors[idx % len(colors)]
    else:
        by = 1
    if label_name:
        columns += [label_name]
    count = vdf.shape()[0]
    if not (ax):
        fig, ax = plt.subplots()
        if isnotebook():
            fig.set_size_inches(12, 8)
        ax.grid()
        ax.set_axisbelow(True)
    else:
        fig = ax.get_figure()
    count = vdf.shape()[0]
    if columns[2] != 1:
        max_size = float(vdf[columns[2]].max())
        min_size = float(vdf[columns[2]].min())
    where = f" AND {order_by} > '{order_by_start}'" if (order_by_start) else ""
    where += f" AND {order_by} < '{order_by_end}'" if (order_by_end) else ""
    query_result = executeSQL(
        query=f"""
            SELECT 
                /*+LABEL('plotting._matplotlib.animated_bubble_plot')*/ * 
            FROM 
                (SELECT 
                    {order_by}, 
                    {", ".join([str(column) for column in columns])}, 
                    {by} 
                 FROM {vdf.__genSQL__(True)} 
                 WHERE  {columns[0]} IS NOT NULL 
                    AND {columns[1]} IS NOT NULL 
                    AND {columns[2]} IS NOT NULL
                    AND {order_by} IS NOT NULL
                    AND {by} IS NOT NULL{where} 
                 LIMIT {limit_over} OVER (PARTITION BY {order_by} 
                                ORDER BY {order_by}, {columns[2]} DESC)) x 
            ORDER BY {order_by}, 4 DESC, 3 DESC, 2 DESC 
            LIMIT {limit}""",
        title="Selecting points to draw the animated bubble plotting._matplotlib.",
        method="fetchall",
    )
    size = 50
    order_by_values = [item[0] for item in query_result]
    if columns[2] != 1:
        size = [
            1000 * (float(item[3]) - min_size) / max((max_size - min_size), 1e-50)
            for item in query_result
        ]
    column1, column2 = (
        [float(item[1]) for item in query_result],
        [float(item[2]) for item in query_result],
    )
    if label_name:
        label_columns = [item[-2] for item in query_result]
    if "cmap" in param:
        c = [float(item[4]) for item in query_result]
    elif by == 1:
        c = colors[0]
    else:
        custom_lines = []
        all_categories = []
        c = []
        for item in query_result:
            if item[-1] not in all_categories:
                all_categories += [item[-1]]
                custom_lines += [Line2D([0], [0], color=colors_map[item[-1]], lw=6)]
            c += [colors_map[item[-1]]]
    current_ts, ts_idx = order_by_values[0], 0
    scatter_values = []
    n = len(order_by_values)
    for idx, elem in enumerate(order_by_values):
        if elem != current_ts or idx == n - 1:
            scatter_values += [
                {
                    "x": column1[ts_idx:idx],
                    "y": column2[ts_idx:idx],
                    "c": c[ts_idx:idx] if isinstance(c, list) else c,
                    "s": size if isinstance(size, (float, int)) else size[ts_idx:idx],
                    "date": current_ts,
                }
            ]
            if label_name:
                scatter_values[-1]["label"] = label_columns[ts_idx:idx]
            current_ts, ts_idx = elem, idx
    im = ax.scatter(
        scatter_values[0]["x"],
        scatter_values[0]["y"],
        c=scatter_values[0]["c"],
        s=scatter_values[0]["s"],
        **updated_dict(param, style_kwds),
    )
    if label_name:
        text_plots = []
        for idx in range(lim_labels):
            text_plots += [
                ax.text(
                    scatter_values[0]["x"][idx],
                    scatter_values[0]["y"][idx],
                    scatter_values[0]["label"][idx],
                    ha="right",
                    va="bottom",
                )
            ]
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    if bbox:
        ax.set_xlim(bbox[0], bbox[1])
        ax.set_ylim(bbox[2], bbox[3])
        if not (date_in_title):
            my_text = ax.text(
                (bbox[0] + bbox[1]) / 2,
                (bbox[2] + bbox[3]) / 2,
                date_f(scatter_values[0]["date"]),
                **date_style_dict,
            )
    elif fixed_xy_lim:
        min_x, max_x = min(column1), max(column1)
        min_y, max_y = min(column2), max(column2)
        delta_x, delta_y = max_x - min_x, max_y - min_y
        ax.set_xlim(min_x - 0.02 * delta_x, max_x + 0.02 * delta_x)
        ax.set_ylim(min_y - 0.02 * delta_y, max_y + 0.02 * delta_y)
        if not (date_in_title):
            my_text = ax.text(
                (max_x + min_x) / 2,
                (max_y + min_y) / 2,
                date_f(scatter_values[0]["date"]),
                **date_style_dict,
            )
    if img:
        bim = plt.imread(img)
        if not (bbox):
            bbox = (min(column1), max(column1), min(column2), max(column2))
            ax.set_xlim(bbox[0], bbox[1])
            ax.set_ylim(bbox[2], bbox[3])
        ax.imshow(bim, extent=bbox)
    elif not (date_in_title):
        my_text = ax.text(
            (max(scatter_values[0]["x"]) + min(scatter_values[0]["x"])) / 2,
            (max(scatter_values[0]["y"]) + min(scatter_values[0]["y"])) / 2,
            date_f(scatter_values[0]["date"]),
            **date_style_dict,
        )
    if "cmap" in param:
        fig.colorbar(im, ax=ax).set_label(by)
    elif label_name:
        leg = ax.legend(
            custom_lines,
            all_categories,
            title=by,
            loc="center left",
            bbox_to_anchor=[1, 0.5],
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    def animate(i):
        array = np.array(
            [
                (scatter_values[i]["x"][j], scatter_values[i]["y"][j])
                for j in range(len(scatter_values[i]["x"]))
            ]
        )
        im.set_offsets(array)
        if columns[2] != 1:
            im.set_sizes(np.array(scatter_values[i]["s"]))
        if "cmap" in param:
            im.set_array(np.array(scatter_values[i]["c"]))
        elif label_name:
            im.set_color(np.array(scatter_values[i]["c"]))
        if "edgecolors" in updated_dict(param, style_kwds):
            im.set_edgecolor(updated_dict(param, style_kwds)["edgecolors"])
        if label_name:
            for k in range(lim_labels):
                text_plots[k].set_position(
                    (scatter_values[i]["x"][k], scatter_values[i]["y"][k])
                )
                text_plots[k].set_text(scatter_values[i]["label"][k])
        min_x, max_x = min(scatter_values[i]["x"]), max(scatter_values[i]["x"])
        min_y, max_y = min(scatter_values[i]["y"]), max(scatter_values[i]["y"])
        delta_x, delta_y = max_x - min_x, max_y - min_y
        if not (fixed_xy_lim):
            ax.set_xlim(min_x - 0.02 * delta_x, max_x + 0.02 * delta_x)
            ax.set_ylim(min_y - 0.02 * delta_y, max_y + 0.02 * delta_y)
            if not (date_in_title):
                my_text.set_position([(max_x + min_x) / 2, (max_y + min_y) / 2])
        if not (date_in_title):
            my_text.set_text(date_f(scatter_values[i]["date"]))
        else:
            ax.set_title(date_f(scatter_values[i]["date"]))
        return (ax,)

    myAnimation = animation.FuncAnimation(
        fig,
        animate,
        frames=range(1, len(scatter_values)),
        interval=interval,
        blit=False,
        repeat=repeat,
    )
    if isnotebook() and return_html:
        anim = myAnimation.to_jshtml()
        plt.close("all")
        return HTML(anim)
    else:
        return myAnimation


def animated_ts_plot(
    vdf,
    order_by: str,
    columns: list = [],
    order_by_start: str = "",
    order_by_end: str = "",
    limit: int = 1000000,
    fixed_xy_lim: bool = False,
    window_size: int = 100,
    step: int = 10,
    interval: int = 5,
    repeat: bool = True,
    return_html: bool = True,
    ax=None,
    **style_kwds,
):
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
        raise EmptyParameter(
            "No numerical columns found to draw the animated multi TS plot"
        )
    order_by_start_str, order_by_end_str, limit_str = "", "", ""
    if order_by_start:
        order_by_start_str = f" AND {order_by} > '{order_by_start}'"
    if order_by_end:
        order_by_end_str = f" AND {order_by} < '{order_by_end}'"
    if limit:
        limit_str = f" LIMIT {limit}"
    condition = " AND ".join([f"{column} IS NOT NULL" for column in columns])
    query_result = executeSQL(
        query=f"""
            SELECT 
                /*+LABEL('plotting._matplotlib.animated_ts_plot')*/ 
                {order_by},
                {", ".join(columns)} 
            FROM {vdf.__genSQL__()} 
            WHERE {order_by} IS NOT NULL
                  {order_by_start_str}
                  {order_by_end_str}
              AND {condition}
            ORDER BY {order_by}
            {limit_str}
            """,
        title="Selecting the needed points to draw the curves",
        method="fetchall",
    )
    order_by_values = [column[0] for column in query_result]
    if isinstance(order_by_values[0], str) and PARSER_IMPORT:
        order_by_values = parse_datetime(order_by_values)
    alpha = 0.3
    if not (ax):
        fig, ax = plt.subplots()
        if isnotebook():
            fig.set_size_inches(8, 6)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    else:
        fig = plt
    all_plots = []
    colors = gen_colors()
    for i in range(0, len(columns)):
        param = {
            "linewidth": 1,
            "label": columns[i],
            "linewidth": 2,
            "color": colors[i % len(colors)],
        }
        all_plots += [ax.plot([], [], **updated_dict(param, style_kwds, i))[0]]
    if len(columns) > 1:
        ax.legend(loc="center left", bbox_to_anchor=[1, 0.5])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_xlabel(order_by)
    if fixed_xy_lim:
        ax.set_xlim(order_by_values[0], order_by_values[-1])
        y_tmp = []
        for m in range(0, len(columns)):
            y_tmp += [item[m + 1] for item in query_result]
        ax.set_ylim(min(y_tmp), max(y_tmp))

    def animate(i):
        k = max(i - window_size, 0)
        x = [elem for elem in order_by_values]
        all_y = []
        for m in range(0, len(columns)):
            y = [item[m + 1] for item in query_result]
            all_plots[m].set_xdata(x[0:i])
            all_plots[m].set_ydata(y[0:i])
            all_y += y[0:i]
        if not (fixed_xy_lim):
            if i > 0:
                ax.set_ylim(min(all_y), max(all_y))
            if i > window_size:
                ax.set_xlim(x[k], x[i])
            else:
                ax.set_xlim(x[0], x[window_size])
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        return (ax,)

    myAnimation = animation.FuncAnimation(
        fig,
        animate,
        frames=range(0, len(order_by_values) - 1, step),
        interval=interval,
        blit=False,
        repeat=repeat,
    )
    if isnotebook() and return_html:
        anim = myAnimation.to_jshtml()
        plt.close("all")
        return HTML(anim)
    else:
        return myAnimation
