# (c) Copyright [2018-2021] Micro Focus or one of its affiliates.
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
from random import shuffle
import math, statistics, warnings

# Other Python Modules
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import numpy as np

# VerticaPy Modules
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.errors import *
import verticapy

#
##
#   /$$$$$$$  /$$        /$$$$$$  /$$$$$$$$
#  | $$__  $$| $$       /$$__  $$|__  $$__/
#  | $$  \ $$| $$      | $$  \ $$   | $$
#  | $$$$$$$/| $$      | $$  | $$   | $$
#  | $$____/ | $$      | $$  | $$   | $$
#  | $$      | $$      | $$  | $$   | $$
#  | $$      | $$$$$$$$|  $$$$$$/   | $$
#  |__/      |________/ \______/    |__/
##
#
#
# Functions used by vDataFrames to draw graphics which are not useful independantly.
#
# ---#
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
        if isnotebook():
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
            x, y, **updated_dict(param, tmp_style,),
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

# ---#
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
    date_f = None,
    date_style_dict: dict = {},
    interval: int = 10,
    repeat: bool = True,
    return_html: bool = True,
    pie: bool = False,
    ax=None,
    **style_kwds,
):
    if not(date_style_dict):
        date_style_dict = {"fontsize": 50, "alpha": 0.6, "color": "gray", "ha": 'right', "va": 'center',}
    if by:
        columns += [by]
    if date_f == None:
        def date_f(x):
            return str(x)
    if "color" in style_kwds:
        colors = style_kwds["color"]
        del style_kwds["color"]
    elif "colors" in style_kwds:
        colors = style_kwds["colors"]
        del style_kwds["color"]
    else:
        colors = gen_colors()
    if isinstance(colors, str):
        colors = []
    colors_map = {}
    if len(columns) >= 3:
        all_cats = vdf[columns[2]].distinct(agg="MAX({})".format(columns[1]))
        for idx, elem in enumerate(all_cats):
            colors_map[elem] = colors[idx % len(colors)]
    else:
        all_cats = vdf[columns[0]].distinct(agg="MAX({})".format(columns[1]))
        for idx, elem in enumerate(all_cats):
            colors_map[elem] = colors[idx % len(colors)]
    where = " AND {} > '{}'".format(order_by, order_by_start) if (order_by_start) else ""
    where += " AND {} < '{}'".format(order_by, order_by_end) if (order_by_end) else ""
    query = "SELECT * FROM (SELECT {}, {} FROM {} WHERE {} IS NOT NULL AND {} LIMIT {} OVER (PARTITION BY {} ORDER BY {} DESC)) x ORDER BY {} ASC, {} ASC LIMIT {}".format(order_by, ", ".join(columns), vdf.__genSQL__(), order_by, " AND ".join([f"{elem} IS NOT NULL" for elem in columns]) + where, limit_over, order_by, columns[1], order_by, columns[1], limit,)
    vdf.__executeSQL__(
        query=query,
        title="Select points to draw the animated bar chart."
    )
    query_result = vdf._VERTICAPY_VARIABLES_["cursor"].fetchall()
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
            bar_values += [{"y": column1[ts_idx:idx], 
                            "width": column2[ts_idx:idx], 
                            "c": color[ts_idx:idx], 
                            "x": column3[ts_idx:idx], 
                            "date": current_ts}]
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
        if not(pie):
            ax.xaxis.grid()
            ax.set_axisbelow(True)
            min_x, max_x = min(bar_values[i]["width"]), max(bar_values[i]["width"])
            delta_x = max_x - min_x
            ax.barh(y=bar_values[i]["y"], width=bar_values[i]["width"], color=bar_values[i]["c"], alpha=0.6, **style_kwds,)
            if bar_values[i]["width"][0] > 0:
                ax.barh(y=bar_values[i]["y"], width=[- 0.3 * delta_x for elem in bar_values[i]["y"]], color=bar_values[i]["c"], alpha=0.6, **style_kwds,)
            if fixed_xy_lim:
                ax.set_xlim(min(column2), max(column2))
            else:
                ax.set_xlim(min_x - 0.3 * delta_x, max_x + 0.3 * delta_x)
            all_text = []
            for k in range(len(bar_values[i]["y"])):
                tmp_txt = []
                tmp_txt += [ax.text(bar_values[i]["width"][k], k + 0.1, bar_values[i]["y"][k], ha="right", fontweight="bold", size=10,)]
                width_format = bar_values[i]["width"][k]
                if width_format - int(width_format) == 0:
                    width_format = int(width_format)
                width_format = f'{width_format:,}'
                tmp_txt += [ax.text(bar_values[i]["width"][k] + 0.005 * delta_x, k - 0.15, width_format, ha="left", size=10,)]
                if len(columns) >= 3:
                    tmp_txt += [ax.text(bar_values[i]["width"][k], k - 0.3, bar_values[i]["x"][k], ha="right", size=10, color="#333333",)]
                all_text += [tmp_txt]
            if date_in_title:
                ax.set_title(date_f(bar_values[i]["date"]))
            else:
                my_text = ax.text(max_x + 0.27 * delta_x, int(limit_over / 2), date_f(bar_values[i]["date"]), **date_style_dict,)
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            ax.set_xlabel(columns[1])
            ax.set_yticks([])
        else:
            param={"wedgeprops": {"edgecolor":"white", "alpha": 0.5,}, "textprops": {'fontsize': 10, 'fontweight': 'bold'}, "autopct": '%1.1f%%'}
            def autopct(val):
                a  = val / 100. * sum(bar_values[i]["width"])
                return f'{a:,}'
            pie_chart = ax.pie(x=bar_values[i]["width"], labels=bar_values[i]["y"], colors=bar_values[i]["c"], **updated_dict(param, style_kwds),)
            for elem in pie_chart[2]:
                elem.set_fontweight('normal')
            if date_in_title:
                ax.set_title(date_f(bar_values[i]["date"]))
            else:
                my_text = ax.text(1.8, 1, date_f(bar_values[i]["date"]), **date_style_dict,)
            all_categories = []
            custom_lines = []
            if len(columns) >= 3:
                for idx, elem in enumerate(bar_values[i]["x"]):
                    if elem not in all_categories:
                        all_categories += [elem]
                        custom_lines += [Line2D([0], [0], color=bar_values[i]["c"][idx], lw=6, alpha=updated_dict(param, style_kwds)["wedgeprops"]["alpha"],)]
                leg = ax.legend(custom_lines,
                                all_categories,
                                title=by,
                                loc="center left", 
                                bbox_to_anchor=[1, 0.5],)
        return ax,

    import matplotlib.animation as animation

    myAnimation = animation.FuncAnimation(fig, animate, frames=range(0, len(bar_values)), interval=interval, blit=False, repeat=repeat)
    if isnotebook() and return_html:
        from IPython.display import HTML
        anim = myAnimation.to_jshtml()
        plt.close("all")
        return HTML(anim)
    else:
        return myAnimation

# ---#
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
    date_f = None,
    date_style_dict: dict = {},
    interval: int = 10,
    repeat: bool = True,
    return_html: bool = True,
    ax=None,
    **style_kwds,
):
    if not(date_style_dict):
        date_style_dict = {"fontsize": 100, "alpha": 0.6, "color": "gray", "ha": 'center', "va": 'center',}
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
            all_cats = vdf[by].distinct(agg="MAX({})".format(columns[2]))
            for idx, elem in enumerate(all_cats):
                colors_map[elem] = colors[idx % len(colors)]
    else:
        by = 1
    if (label_name):
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
        max_size, min_size = float(vdf[columns[2]].max()), float(vdf[columns[2]].min())
    where = " AND {} > '{}'".format(order_by, order_by_start) if (order_by_start) else ""
    where += " AND {} < '{}'".format(order_by, order_by_end) if (order_by_end) else ""
    query = "SELECT * FROM (SELECT {}, {}, {} FROM {} WHERE  {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL{} LIMIT {} OVER (PARTITION BY {} ORDER BY {}, {} DESC)) x ORDER BY {}, 4 DESC, 3 DESC, 2 DESC LIMIT {}"
    query = query.format(
        order_by,
        ", ".join([str(elem) for elem in columns]),
        by,
        vdf.__genSQL__(True),
        columns[0],
        columns[1],
        columns[2],
        order_by,
        by,
        where,
        limit_over,
        order_by,
        order_by,
        columns[2],
        order_by,
        limit,
    )
    vdf.__executeSQL__(
        query=query,
        title="Select points to draw the animated bubble plot."
    )
    query_result = vdf._VERTICAPY_VARIABLES_["cursor"].fetchall()
    size = 50
    order_by_values = [item[0] for item in query_result]
    if columns[2] != 1:
        size = [1000 * (float(item[3]) - min_size) / max((max_size - min_size), 1e-50) for item in query_result]
    column1, column2 = [float(item[1]) for item in query_result], [float(item[2]) for item in query_result]
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
            scatter_values += [{"x": column1[ts_idx:idx], 
                                "y": column2[ts_idx:idx], 
                                "c": c[ts_idx:idx] if isinstance(c, list) else c, 
                                "s": size if isinstance(size, (float, int)) else size[ts_idx:idx],
                                "date": current_ts,}]
            if label_name:
                scatter_values[-1]["label"] = label_columns[ts_idx:idx]
            current_ts, ts_idx = elem, idx
    im = ax.scatter(scatter_values[0]["x"], scatter_values[0]["y"], c=scatter_values[0]["c"], s=scatter_values[0]["s"], **updated_dict(param, style_kwds),)
    if label_name:
        text_plots = []
        for idx in range(lim_labels):
            text_plots += [ax.text(scatter_values[0]["x"][idx], scatter_values[0]["y"][idx], scatter_values[0]["label"][idx], ha="right", va="bottom")]
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    if bbox:
        ax.set_xlim(bbox[0], bbox[1])
        ax.set_ylim(bbox[2], bbox[3])
        if not(date_in_title):
            my_text = ax.text((bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2, date_f(scatter_values[0]["date"]), **date_style_dict,)
    elif (fixed_xy_lim):
        min_x, max_x = min(column1), max(column1)
        min_y, max_y = min(column2), max(column2)
        delta_x, delta_y = max_x - min_x, max_y - min_y
        ax.set_xlim(min_x - 0.02 * delta_x, max_x + 0.02 * delta_x)
        ax.set_ylim(min_y - 0.02 * delta_y, max_y + 0.02 * delta_y)
        if not(date_in_title):
            my_text = ax.text((max_x + min_x) / 2, (max_y + min_y) / 2, date_f(scatter_values[0]["date"]), **date_style_dict,)
    if img:
        bim = plt.imread(img)
        if not (bbox):
            bbox = (min(column1), max(column1), min(column2), max(column2))
            ax.set_xlim(bbox[0], bbox[1])
            ax.set_ylim(bbox[2], bbox[3])
        ax.imshow(bim, extent=bbox)
    elif not(date_in_title):
        my_text = ax.text((max(scatter_values[0]["x"]) + min(scatter_values[0]["x"])) / 2, (max(scatter_values[0]["y"]) + min(scatter_values[0]["y"])) / 2, date_f(scatter_values[0]["date"]), **date_style_dict,)
    if "cmap" in param:
        fig.colorbar(im, ax=ax).set_label(by)
    elif label_name:
        leg = ax.legend(
            custom_lines,
            all_categories,
            title=by,
            loc="center left", 
            bbox_to_anchor=[1, 0.5],)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    def animate(i):
        array = np.array([(scatter_values[i]["x"][j], scatter_values[i]["y"][j]) for j in range(len(scatter_values[i]["x"]))])
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
                text_plots[k].set_position((scatter_values[i]["x"][k], scatter_values[i]["y"][k]))
                text_plots[k].set_text(scatter_values[i]["label"][k])
        min_x, max_x = min(scatter_values[i]["x"]), max(scatter_values[i]["x"])
        min_y, max_y = min(scatter_values[i]["y"]), max(scatter_values[i]["y"])
        delta_x, delta_y = max_x - min_x, max_y - min_y
        if not(fixed_xy_lim):
            ax.set_xlim(min_x - 0.02 * delta_x, max_x + 0.02 * delta_x)
            ax.set_ylim(min_y - 0.02 * delta_y, max_y + 0.02 * delta_y)
            if not(date_in_title):
                my_text.set_position([(max_x + min_x) / 2, (max_y + min_y) / 2])
        if not(date_in_title):
            my_text.set_text(date_f(scatter_values[i]["date"]))
        else:
            ax.set_title(date_f(scatter_values[i]["date"]))
        return ax,

    import matplotlib.animation as animation

    myAnimation = animation.FuncAnimation(fig, animate, frames=range(1, len(scatter_values)), interval=interval, blit=False, repeat=repeat)
    if isnotebook() and return_html:
        from IPython.display import HTML
        anim = myAnimation.to_jshtml()
        plt.close("all")
        return HTML(anim)
    else:
        return myAnimation

# ---#
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
                warning_message = "The Virtual Column {} is not numerical.\nIt will be ignored.".format(
                    column
                )
                warnings.warn(warning_message, Warning)
            columns.remove(column)
    if not (columns):
        raise EmptyParameter("No numerical columns found to draw the animated multi TS plot")
    query = "SELECT {}, {} FROM {} WHERE {} IS NOT NULL".format(
        order_by, ", ".join(columns), vdf.__genSQL__(), order_by,
    )
    query += (
        " AND {} > '{}'".format(order_by, order_by_start) if (order_by_start) else ""
    )
    query += " AND {} < '{}'".format(order_by, order_by_end) if (order_by_end) else ""
    query += " AND " + " AND ".join(
        ["{} IS NOT NULL".format(column) for column in columns]
    )
    query += " ORDER BY {}".format(order_by)
    if limit:
        query += " LIMIT {}".format(limit)
    vdf.__executeSQL__(query=query, title="Select the needed points to draw the curves")
    query_result = vdf._VERTICAPY_VARIABLES_["cursor"].fetchall()
    order_by_values = [item[0] for item in query_result]
    try:
        if isinstance(order_by_values[0], str):
            from dateutil.parser import parse

            order_by_values = [parse(elem) for elem in order_by_values]
    except:
        pass
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
        param = {"linewidth": 1, "label": columns[i], "linewidth": 2, "color": colors[i % len(colors)]}
        all_plots += [ax.plot([], [], **updated_dict(param, style_kwds, i),)[0]]
    if len(columns) > 1:
        ax.legend(loc="center left", bbox_to_anchor=[1, 0.5])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_xlabel(order_by)
    if (fixed_xy_lim):
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
            all_plots[m].set_xdata(x[0:i],)
            all_plots[m].set_ydata(y[0:i],)
            all_y += y[0:i]
        if not(fixed_xy_lim):
            if i > 0:
                ax.set_ylim(min(all_y), max(all_y))
            if i > window_size:
                ax.set_xlim(x[k], x[i])
            else:
                ax.set_xlim(x[0], x[window_size])
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        return ax,

    import matplotlib.animation as animation

    myAnimation = animation.FuncAnimation(fig, animate, frames=range(0, len(order_by_values) - 1, step), interval=interval, blit=False, repeat=repeat)
    if isnotebook() and return_html:
        from IPython.display import HTML
        anim = myAnimation.to_jshtml()
        plt.close("all")
        return HTML(anim)
    else:
        return myAnimation

# ---#
def bar(
    vdf,
    method: str = "density",
    of=None,
    max_cardinality: int = 6,
    bins: int = 0,
    h: float = 0,
    ax=None,
    **style_kwds,
):
    x, y, z, h, is_categorical = compute_plot_variables(
        vdf, method=method, of=of, max_cardinality=max_cardinality, bins=bins, h=h
    )
    if not (ax):
        fig, ax = plt.subplots()
        if isnotebook():
            fig.set_size_inches(10, min(int(len(x) / 1.8) + 1, 600))
        ax.xaxis.grid()
        ax.set_axisbelow(True)
    param = {"color": gen_colors()[0], "alpha": 0.86}
    ax.barh(
        x, y, h, **updated_dict(param, style_kwds, 0),
    )
    ax.set_ylabel(vdf.alias)
    if is_categorical:
        if vdf.category() == "text":
            new_z = []
            for item in z:
                new_z += [item[0:47] + "..."] if (len(str(item)) > 50) else [item]
        else:
            new_z = z
        ax.set_yticks(x)
        ax.set_yticklabels(new_z, rotation=0)
    else:
        ax.set_yticks([elem - round(h / 2 / 0.94, 10) for elem in x])
    if method.lower() == "density":
        ax.set_xlabel("Density")
    elif (method.lower() in ["avg", "min", "max", "sum"] or "%" == method[-1]) and (
        of != None
    ):
        aggregate = "{}({})".format(method.upper(), of)
        ax.set_xlabel(aggregate)
    elif method.lower() == "count":
        ax.set_xlabel("Frequency")
    else:
        ax.set_xlabel(method)
    return ax


# ---#
def bar2D(
    vdf,
    columns: list,
    method: str = "density",
    of: str = "",
    max_cardinality: tuple = (6, 6),
    h: tuple = (None, None),
    stacked: bool = False,
    fully_stacked: bool = False,
    density: bool = False,
    ax=None,
    **style_kwds,
):
    colors = gen_colors()
    if fully_stacked:
        if method != "density":
            raise ParameterError(
                "Fully Stacked Bar works only with the 'density' method."
            )
    if density:
        if method != "density":
            raise ParameterError("Pyramid Bar works only with the 'density' method.")
        unique = vdf.nunique(columns)["unique"]
        if unique[1] != 2 and unique[0] != 2:
            raise ParameterError(
                "One of the 2 columns must have 2 categories to draw a Pyramid Bar."
            )
        if unique[1] != 2:
            columns = [columns[1], columns[0]]
    all_columns = vdf.pivot_table(
        columns, method=method, of=of, h=h, max_cardinality=max_cardinality, show=False
    ).values
    all_columns = [[column] + all_columns[column] for column in all_columns]
    n = len(all_columns)
    m = len(all_columns[0])
    n_groups = m - 1
    bar_width = 0.5
    if not (ax):
        fig, ax = plt.subplots()
        if isnotebook():
            if density:
                fig.set_size_inches(10, min(m * 3, 600) / 8 + 1)
            else:
                fig.set_size_inches(10, min(m * 3, 600) / 2 + 1)
        ax.set_axisbelow(True)
        ax.xaxis.grid()
    if not (fully_stacked):
        for i in range(1, n):
            current_column = all_columns[i][1:m]
            for idx, item in enumerate(current_column):
                try:
                    current_column[idx] = float(item)
                except:
                    current_column[idx] = 0
            current_label = str(all_columns[i][0])
            param = {"alpha": 0.86, "color": colors[(i - 1) % len(colors)]}
            if stacked:
                if i == 1:
                    last_column = [0 for item in all_columns[i][1:m]]
                else:
                    for idx, item in enumerate(all_columns[i - 1][1:m]):
                        try:
                            last_column[idx] += float(item)
                        except:
                            last_column[idx] += 0
                ax.barh(
                    [elem for elem in range(n_groups)],
                    current_column,
                    bar_width,
                    label=current_label,
                    left=last_column,
                    **updated_dict(param, style_kwds, i - 1),
                )
            elif density:
                if i == 2:
                    current_column = [-elem for elem in current_column]
                ax.barh(
                    [elem for elem in range(n_groups)],
                    current_column,
                    bar_width / 1.5,
                    label=current_label,
                    **updated_dict(param, style_kwds, i - 1),
                )
            else:
                ax.barh(
                    [elem + (i - 1) * bar_width / (n - 1) for elem in range(n_groups)],
                    current_column,
                    bar_width / (n - 1),
                    label=current_label,
                    **updated_dict(param, style_kwds, i - 1),
                )
        if stacked:
            ax.set_yticks([elem for elem in range(n_groups)])
            ax.set_yticklabels(all_columns[0][1:m])
        else:
            ax.set_yticks(
                [
                    elem + bar_width / 2 - bar_width / 2 / (n - 1)
                    for elem in range(n_groups)
                ],
            )
            ax.set_yticklabels(all_columns[0][1:m])
        ax.set_ylabel(columns[0])
        if method.lower() == "mean":
            method = "avg"
        if method.lower() == "mean":
            method = "avg"
        if method.lower() == "density":
            ax.set_xlabel("Density")
        elif (method.lower() in ["avg", "min", "max", "sum"]) and (of != None):
            ax.set_xlabel("{}({})".format(method, of))
        elif method.lower() == "count":
            ax.set_xlabel("Frequency")
        else:
            ax.set_xlabel(method)
    else:
        total = [0 for item in range(1, m)]
        for i in range(1, n):
            for j in range(1, m):
                if not (isinstance(all_columns[i][j], str)):
                    total[j - 1] += float(
                        all_columns[i][j] if (all_columns[i][j] != None) else 0
                    )
        for i in range(1, n):
            for j in range(1, m):
                if not (isinstance(all_columns[i][j], str)):
                    if total[j - 1] != 0:
                        all_columns[i][j] = (
                            float(
                                all_columns[i][j] if (all_columns[i][j] != None) else 0
                            )
                            / total[j - 1]
                        )
                    else:
                        all_columns[i][j] = 0
        for i in range(1, n):
            current_column = all_columns[i][1:m]
            for idx, item in enumerate(current_column):
                try:
                    current_column[idx] = float(item)
                except:
                    current_column[idx] = 0
            current_label = str(all_columns[i][0])
            if i == 1:
                last_column = [0 for item in all_columns[i][1:m]]
            else:
                for idx, item in enumerate(all_columns[i - 1][1:m]):
                    try:
                        last_column[idx] += float(item)
                    except:
                        last_column[idx] += 0
            param = {"color": colors[(i - 1) % len(colors)], "alpha": 0.86}
            ax.barh(
                [elem for elem in range(n_groups)],
                current_column,
                bar_width,
                label=current_label,
                left=last_column,
                **updated_dict(param, style_kwds, i - 1),
            )
        ax.set_yticks([elem for elem in range(n_groups)])
        ax.set_yticklabels(all_columns[0][1:m])
        ax.set_ylabel(columns[0])
        ax.set_xlabel("Density")
    if density or fully_stacked:
        vals = ax.get_xticks()
        max_val = max([abs(x) for x in vals])
        ax.xaxis.set_major_locator(mticker.FixedLocator(vals))
        ax.set_xticklabels(["{:,.2%}".format(abs(x)) for x in vals])
    ax.legend(title=columns[1], loc="center left", bbox_to_anchor=[1, 0.5])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    return ax


# ---#
def boxplot(
    vdf,
    by: str = "",
    h: float = 0,
    max_cardinality: int = 8,
    cat_priority: list = [],
    ax=None,
    **style_kwds,
):
    colors = []
    if "color" in style_kwds:
        if isinstance(style_kwds["color"], str):
            colors = [style_kwds["color"]]
        else:
            colors = style_kwds["color"]
        del style_kwds["color"]
    elif "colors" in style_kwds:
        if isinstance(style_kwds["colors"], str):
            colors = [style_kwds["colors"]]
        else:
            colors = style_kwds["colors"]
        del style_kwds["colors"]
    colors += gen_colors()
    # SINGLE BOXPLOT
    if by == "":
        if not (ax):
            fig, ax = plt.subplots()
            if isnotebook():
                fig.set_size_inches(6, 4)
            ax.xaxis.grid()
        if not (vdf.isnum()):
            raise TypeError("The column must be numerical in order to draw a boxplot")
        summarize = (
            vdf.parent.describe(method="numerical", columns=[vdf.alias], unique=False)
            .transpose()
            .values[vdf.alias]
        )
        for i in range(0, 2):
            del summarize[0]
        ax.set_xlabel(vdf.alias)
        box = ax.boxplot(
            summarize,
            notch=False,
            sym="",
            whis=float("Inf"),
            vert=False,
            widths=0.7,
            labels=[""],
            patch_artist=True,
            **style_kwds,
        )
        for median in box["medians"]:
            median.set(
                color="black", linewidth=1,
            )
        for patch in box["boxes"]:
            patch.set_facecolor(colors[0])
        ax.set_axisbelow(True)
        return ax
    # MULTI BOXPLOT
    else:
        try:
            if vdf.alias == by:
                raise NameError("The column and the groupby can not be the same")
            elif by not in vdf.parent.get_columns():
                raise MissingColumn("The column " + by + " doesn't exist")
            count = vdf.parent.shape()[0]
            is_numeric = vdf.parent[by].isnum()
            is_categorical = (vdf.parent[by].nunique(True) <= max_cardinality) or not (
                is_numeric
            )
            table = vdf.parent.__genSQL__()
            if not (is_categorical):
                enum_trans = (
                    vdf.parent[by]
                    .discretize(h=h, return_enum_trans=True)[0]
                    .replace("{}", by)
                    + " AS "
                    + by
                )
                enum_trans += ", {}".format(vdf.alias)
                table = "(SELECT {} FROM {}) enum_table".format(enum_trans, table)
            if not (cat_priority):
                query = "SELECT {} FROM {} WHERE {} IS NOT NULL GROUP BY {} ORDER BY COUNT(*) DESC LIMIT {}".format(
                    by, table, vdf.alias, by, max_cardinality
                )
                query_result = vdf.__executeSQL__(
                    query=query, title="Compute the categories of {}".format(by)
                ).fetchall()
                cat_priority = [item for sublist in query_result for item in sublist]
            with_summarize = False
            query = []
            lp = "(" if (len(cat_priority) == 1) else ""
            rp = ")" if (len(cat_priority) == 1) else ""
            for idx, category in enumerate(cat_priority):
                tmp_query = "SELECT MIN({}) AS min, APPROXIMATE_PERCENTILE ({} USING PARAMETERS percentile = 0.25) AS Q1, APPROXIMATE_PERCENTILE ({}".format(
                    vdf.alias, vdf.alias, vdf.alias
                )
                tmp_query += "USING PARAMETERS percentile = 0.5) AS Median, APPROXIMATE_PERCENTILE ({} USING PARAMETERS percentile = 0.75) AS Q3, MAX".format(
                    vdf.alias
                )
                tmp_query += "({}) AS max, '{}' FROM vdf_table".format(vdf.alias, "{}")
                tmp_query = (
                    tmp_query.format("None")
                    if (category in ("None", None))
                    else tmp_query.format(category)
                )
                tmp_query += (
                    " WHERE {} IS NULL".format(by)
                    if (category in ("None", None))
                    else " WHERE {} = '{}'".format(by, str(category).replace("'", "''"))
                )
                query += [lp + tmp_query + rp]
            query = "WITH vdf_table AS (SELECT * FROM {}) {}".format(
                table, " UNION ALL ".join(query)
            )
            try:
                vdf.__executeSQL__(
                    query=query,
                    title="Compute all the descriptive statistics for each category to draw the box plot",
                )
                query_result = vdf.parent._VERTICAPY_VARIABLES_["cursor"].fetchall()
            except:
                query_result = []
                for idx, category in enumerate(cat_priority):
                    tmp_query = "SELECT MIN({}) AS min, APPROXIMATE_PERCENTILE ({} USING PARAMETERS percentile = 0.25) AS Q1, APPROXIMATE_PERCENTILE ({}".format(
                        vdf.alias, vdf.alias, vdf.alias
                    )
                    tmp_query += "USING PARAMETERS percentile = 0.5) AS Median, APPROXIMATE_PERCENTILE ({} USING PARAMETERS percentile = 0.75) AS Q3, MAX".format(
                        vdf.alias
                    )
                    tmp_query += "({}) AS max, '{}' FROM {}".format(
                        vdf.alias, "{}", vdf.parent.__genSQL__()
                    )
                    tmp_query = (
                        tmp_query.format("None")
                        if (category in ("None", None))
                        else tmp_query.format(str(category).replace("'", "''"))
                    )
                    tmp_query += (
                        " WHERE {} IS NULL".format(by)
                        if (category in ("None", None))
                        else " WHERE {} = '{}'".format(
                            by, str(category).replace("'", "''")
                        )
                    )
                    vdf.__executeSQL__(
                        query=tmp_query,
                        title="Compute all the descriptive statistics for each category to draw the box plot, one at a time",
                    )
                    query_result += [
                        vdf.parent._VERTICAPY_VARIABLES_["cursor"].fetchone()
                    ]
            cat_priority = [item[-1] for item in query_result]
            result = [[float(item[i]) for i in range(0, 5)] for item in query_result]
            result.reverse()
            cat_priority.reverse()
            if vdf.parent[by].category() == "text":
                labels = []
                for item in cat_priority:
                    labels += [item[0:47] + "..."] if (len(str(item)) > 50) else [item]
            else:
                labels = cat_priority
            if not (ax):
                fig, ax = plt.subplots()
                if isnotebook():
                    fig.set_size_inches(10, 6)
                ax.yaxis.grid()
            ax.set_ylabel(vdf.alias)
            ax.set_xlabel(by)
            other_labels = []
            other_result = []
            all_idx = []
            if not (is_categorical):
                for idx, item in enumerate(labels):
                    try:
                        math.floor(int(item))
                    except:
                        try:
                            math.floor(float(item))
                        except:
                            try:
                                math.floor(float(labels[idx][1:-1].split(";")[0]))
                            except:
                                other_labels += [labels[idx]]
                                other_result += [result[idx]]
                                all_idx += [idx]
                for idx in all_idx:
                    del labels[idx]
                    del result[idx]
            if not (is_categorical):
                sorted_boxplot = sorted(
                    [
                        [float(labels[i][1:-1].split(";")[0]), labels[i], result[i]]
                        for i in range(len(labels))
                    ]
                )
                labels, result = (
                    [item[1] for item in sorted_boxplot] + other_labels,
                    [item[2] for item in sorted_boxplot] + other_result,
                )
            else:
                sorted_boxplot = sorted(
                    [(labels[i], result[i]) for i in range(len(labels))]
                )
                labels, result = (
                    [item[0] for item in sorted_boxplot],
                    [item[1] for item in sorted_boxplot],
                )
            box = ax.boxplot(
                result,
                notch=False,
                sym="",
                whis=float("Inf"),
                widths=0.5,
                labels=labels,
                patch_artist=True,
                **style_kwds,
            )
            ax.set_xticklabels(labels, rotation=90)
            for median in box["medians"]:
                median.set(
                    color="black", linewidth=1,
                )
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
            return ax
        except Exception as e:
            raise Exception(
                "{}\nAn error occured during the BoxPlot creation.".format(e)
            )


# ---#
def boxplot2D(
    vdf, columns: list = [], ax=None, **style_kwds,
):
    colors = []
    if "color" in style_kwds:
        if isinstance(style_kwds["color"], str):
            colors = [style_kwds["color"]]
        else:
            colors = style_kwds["color"]
        del style_kwds["color"]
    elif "colors" in style_kwds:
        if isinstance(style_kwds["colors"], str):
            colors = [style_kwds["colors"]]
        else:
            colors = style_kwds["colors"]
        del style_kwds["colors"]
    colors += gen_colors()
    if not (columns):
        columns = vdf.numcol()
    for column in columns:
        if column not in vdf.numcol():
            if vdf._VERTICAPY_VARIABLES_["display"]["print_info"]:
                warning_message = "The Virtual Column {} is not numerical.\nIt will be ignored.".format(
                    column
                )
                warnings.warn(warning_message, Warning)
            columns.remove(column)
    if not (columns):
        raise MissingColumn("No numerical columns found to draw the multi boxplot")
    # SINGLE BOXPLOT
    if len(columns) == 1:
        vdf[columns[0]].boxplot(
            ax=ax, **style_kwds,
        )
    # MULTI BOXPLOT
    else:
        try:
            summarize = vdf.describe(columns=columns).transpose()
            result = [summarize.values[column][3:8] for column in summarize.values]
            columns = [column for column in summarize.values]
            del columns[0]
            del result[0]
            if not (ax):
                fig, ax = plt.subplots()
                if isnotebook():
                    fig.set_size_inches(10, 6)
            box = ax.boxplot(
                result,
                notch=False,
                sym="",
                whis=float("Inf"),
                widths=0.5,
                labels=columns,
                patch_artist=True,
                **style_kwds,
            )
            ax.set_xticklabels(columns, rotation=90)
            for median in box["medians"]:
                median.set(
                    color="black", linewidth=1,
                )
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
            return ax
        except Exception as e:
            raise Exception(
                "{}\nAn error occured during the BoxPlot creation.".format(e)
            )


# ---#
def bubble(
    vdf,
    columns: list,
    catcol: str = "",
    cmap_col: str = "",
    max_nb_points: int = 1000,
    bbox: list = [],
    img: str = "",
    ax=None,
    **style_kwds,
):
    assert not(catcol) or not(cmap_col), ParameterError("Bubble Plot only accepts either a cmap column or a categorical column. It can not accept both.")
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
    if not(catcol) and not(cmap_col):
        tablesample = max_nb_points / vdf.shape()[0]
        query = "SELECT {}, {}, {} FROM {} WHERE __verticapy_split__ < {} AND {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL LIMIT {}".format(
            columns[0],
            columns[1],
            columns[2],
            vdf.__genSQL__(True),
            tablesample,
            columns[0],
            columns[1],
            columns[2],
            max_nb_points,
        )
        query_result = vdf.__executeSQL__(
            query=query, title="Select random points to draw the scatter plot"
        ).fetchall()
        size = 50
        if columns[2] != 1:
            max_size = max([float(item[2]) for item in query_result])
            min_size = min([float(item[2]) for item in query_result])
            size = [1000 * (float(item[2]) - min_size) / max((max_size - min_size), 1e-50) for item in query_result]
        column1, column2 = [float(item[0]) for item in query_result], [float(item[1]) for item in query_result]
        if not (ax):
            fig, ax = plt.subplots()
            if isnotebook():
                fig.set_size_inches(10, 6)
            ax.grid()
            ax.set_axisbelow(True)
        if bbox:
            ax.set_xlim(bbox[0], bbox[1])
            ax.set_ylim(bbox[2], bbox[3])
        if img:
            im = plt.imread(img)
            if not (bbox):
                bbox = (min(column1), max(column1), min(column2), max(column2))
                ax.set_xlim(bbox[0], bbox[1])
                ax.set_ylim(bbox[2], bbox[3])
            ax.imshow(im, extent=bbox)
        ax.set_ylabel(columns[1])
        ax.set_xlabel(columns[0])
        param = {
            "color": colors[0],
            "alpha": 0.8,
            "edgecolors": "black",
        }
        scatter = ax.scatter(
            column1, column2, s=size, **updated_dict(param, style_kwds),
        )
        if columns[2] != 1:
            leg1 = ax.legend(
                [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=colors[0],
                        label="Scatter",
                        markersize=min(size) / 100 + 15,
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=colors[0],
                        label="Scatter",
                        markersize=max(size) / 100 + 15,
                    ),
                ],
                [
                    min([item[2] for item in query_result]),
                    max([item[2] for item in query_result]),
                ],
                bbox_to_anchor=[1, 0.5],
                loc="center left",
                title=columns[2],
                labelspacing=1,
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    else:
        count = vdf.shape()[0]
        if not (ax):
            fig, ax = plt.subplots()
            if isnotebook():
                fig.set_size_inches(12, 7)
            ax.grid()
            ax.set_axisbelow(True)
        else:
            fig = plt
        if bbox:
            ax.set_xlim(bbox[0], bbox[1])
            ax.set_ylim(bbox[2], bbox[3])
        if img:
            im = plt.imread(img)
            if not (bbox):
                aggr = vdf.agg(columns=[columns[0], columns[1]], func=["min", "max"])
                bbox = (
                    aggr.values["min"][0],
                    aggr.values["max"][0],
                    aggr.values["min"][1],
                    aggr.values["max"][1],
                )
                ax.set_xlim(bbox[0], bbox[1])
                ax.set_ylim(bbox[2], bbox[3])
            ax.imshow(im, extent=bbox)
        others = []
        count = vdf.shape()[0]
        tablesample = 0.1 if (count > 10000) else 0.9
        if columns[2] != 1:
            max_size, min_size = float(vdf[columns[2]].max()), float(vdf[columns[2]].min())
        custom_lines = []
        if catcol:
            all_categories = vdf[catcol].distinct()
            groupby_cardinality = vdf[catcol].nunique(True)
            for idx, category in enumerate(all_categories):
                query = "SELECT {}, {}, {} FROM {} WHERE  __verticapy_split__ < {} AND {} = '{}' AND {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL LIMIT {}"
                query = query.format(
                    columns[0],
                    columns[1],
                    columns[2],
                    vdf.__genSQL__(True),
                    tablesample,
                    catcol,
                    str(category).replace("'", "''"),
                    columns[0],
                    columns[1],
                    columns[2],
                    int(max_nb_points / len(all_categories)),
                )
                vdf.__executeSQL__(
                    query=query,
                    title="Select random points to draw the bubble plot (category = '{}')".format(
                        str(category)
                    ),
                )
                query_result = vdf._VERTICAPY_VARIABLES_["cursor"].fetchall()
                size = 50
                if columns[2] != 1:
                    size = [1000 * (float(item[2]) - min_size) / max((max_size - min_size), 1e-50) for item in query_result]
                column1, column2 = [float(item[0]) for item in query_result], [float(item[1]) for item in query_result]
                param = {
                    "alpha": 0.8,
                    "color": colors[idx % len(colors)],
                    "edgecolors": "black",
                }
                ax.scatter(column1, column2, s=size, **updated_dict(param, style_kwds, idx),)
                custom_lines += [Line2D([0], [0], color=colors[idx % len(colors)], lw=6)]
            for idx, item in enumerate(all_categories):
                if len(str(item)) > 20:
                    all_categories[idx] = str(item)[0:20] + "..."
        else:
            query = "SELECT {}, {}, {}, {} FROM {} WHERE  __verticapy_split__ < {} AND {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL LIMIT {}"
            query = query.format(
                columns[0],
                columns[1],
                columns[2],
                cmap_col,
                vdf.__genSQL__(True),
                tablesample,
                columns[0],
                columns[1],
                columns[2],
                cmap_col,
                max_nb_points,
            )
            vdf.__executeSQL__(
                query=query,
                title="Select random points to draw the bubble plot with cmap expr."
            )
            query_result = vdf._VERTICAPY_VARIABLES_["cursor"].fetchall()
            size = 50
            if columns[2] != 1:
                size = [1000 * (float(item[2]) - min_size) / max((max_size - min_size), 1e-50) for item in query_result]
            column1, column2, column3 = [float(item[0]) for item in query_result], [float(item[1]) for item in query_result], [float(item[3]) for item in query_result]
            param = {
                "alpha": 0.8,
                "cmap": gen_cmap()[0],
                "edgecolors": "black",
            }
            im = ax.scatter(column1, column2, c=column3, s=size, **updated_dict(param, style_kwds),)
        if columns[2] != 1:
            if catcol:
                bbox_to_anchor = [1, 0.5]
                loc = "center left"
            else:
                bbox_to_anchor = [-0.1, 0.5]
                loc = "center right"
            leg1 = ax.legend(
                [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor="black",
                        label="Scatter",
                        markersize=min(size) / 100 + 15,
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor="black",
                        label="Scatter",
                        markersize=max(size) / 100 + 15,
                    ),
                ],
                [
                    min([item[2] for item in query_result]),
                    max([item[2] for item in query_result]),
                ],
                bbox_to_anchor=bbox_to_anchor,
                loc=loc,
                title=columns[2],
                labelspacing=1,
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        if catcol:
            leg2 = ax.legend(
                custom_lines,
                all_categories,
                title=catcol,
                loc="center right",
                bbox_to_anchor=[-0.06, 0.5],
            )
        else:
            fig.colorbar(im, ax=ax).set_label(cmap_col)
        if columns[2] != 1 and catcol:
            ax.add_artist(leg1)
    return ax


# ---#
def cmatrix(
    matrix,
    columns_x,
    columns_y,
    n: int,
    m: int,
    vmax: float,
    vmin: float,
    title: str = "",
    colorbar: str = "",
    x_label: str = "",
    y_label: str = "",
    with_numbers: bool = True,
    mround: int = 3,
    is_vector: bool = False,
    inverse: bool = False,
    extent: list = [],
    is_pivot: bool = False,
    ax=None,
    **style_kwds,
):
    if is_vector:
        is_vector = True
        vector = [elem for elem in matrix[1]]
        matrix_array = vector[1:]
        for i in range(len(matrix_array)):
            matrix_array[i] = round(float(matrix_array[i]), mround)
        matrix_array = [matrix_array]
        m, n = n, m
        x_label, y_label = y_label, x_label
        columns_x, columns_y = columns_y, columns_x
    else:
        matrix_array = [
            [
                round(float(matrix[i][j]), mround)
                if (matrix[i][j] != None and matrix[i][j] != "")
                else float("nan")
                for i in range(1, m + 1)
            ]
            for j in range(1, n + 1)
        ]
        if inverse:
            matrix_array.reverse()
            columns_x.reverse()
    if not (ax):
        fig, ax = plt.subplots()
        if (isnotebook() and not (inverse)) or is_pivot:
            fig.set_size_inches(min(m, 500), min(n, 500))
        else:
            fig.set_size_inches(8, 6)
    else:
        fig = plt
    param = {"cmap": gen_cmap()[0], "interpolation": "nearest"}
    if ((vmax == 1) and vmin in [0, -1]) and not (extent):
        im = ax.imshow(
            matrix_array, vmax=vmax, vmin=vmin, **updated_dict(param, style_kwds),
        )
    else:
        try:
            im = ax.imshow(
                matrix_array, extent=extent, **updated_dict(param, style_kwds),
            )
        except:
            im = ax.imshow(matrix_array, **updated_dict(param, style_kwds),)
    fig.colorbar(im, ax=ax).set_label(colorbar)
    if not (extent):
        ax.set_yticks([i for i in range(0, n)])
        ax.set_xticks([i for i in range(0, m)])
        ax.set_xticklabels(columns_y, rotation=90)
        ax.set_yticklabels(columns_x, rotation=0)
    if with_numbers:
        for y_index in range(n):
            for x_index in range(m):
                label = matrix_array[y_index][x_index]
                ax.text(
                    x_index, y_index, label, color="black", ha="center", va="center"
                )
    return ax


# ---#
def contour_plot(
    vdf,
    columns: list,
    func,
    nbins: int = 100,
    cbar_title: str = "",
    pos_label: (int, str, float) = None,
    ax=None,
    **style_kwds,
):
    if not(cbar_title) and str(type(func)) in ("<class 'function'>", "<class 'method'>",):
        cbar_title = func.__name__
    all_agg = vdf.agg(["min", "max"], columns)
    min_x, min_y = all_agg["min"]
    max_x, max_y = all_agg["max"]
    if str(type(func)) in ("<class 'function'>", "<class 'method'>",):
        xlist = np.linspace(min_x, max_x, nbins)
        ylist = np.linspace(min_y, max_y, nbins)
        X, Y = np.meshgrid(xlist, ylist)
        Z = func(X, Y)
    else:
        from verticapy.datasets import gen_meshgrid

        vdf_tmp = gen_meshgrid({str_column(columns[1])[1:-1]: {"type": float, "range": [min_y, max_y], "nbins": nbins},
                                       str_column(columns[0])[1:-1]: {"type": float, "range": [min_x, max_x], "nbins": nbins},},
                                       cursor=vdf._VERTICAPY_VARIABLES_["cursor"])
        y = "verticapy_predict"
        if isinstance(func, (str, str_sql,)):
            vdf_tmp["verticapy_predict"] = func
        else:
            if func.type in ("RandomForestClassifier", "NaiveBayes", "NearestCentroid", "KNeighborsClassifier",):
                if func.type in ("NearestCentroid", "KNeighborsClassifier",):
                    vdf_tmp = func.predict(vdf=vdf_tmp, X=columns, name="verticapy_predict", all_classes=True, key_columns=None,)
                    y = "verticapy_predict_{}".format(pos_label)
                else:
                    vdf_tmp = func.predict(vdf=vdf_tmp, X=columns, name="verticapy_predict", pos_label=pos_label)
            else:
                if func.type in ("KNeighborsRegressor",):
                    vdf_tmp = func.predict(vdf=vdf_tmp, X=columns, name="verticapy_predict", key_columns=None,)
                else:
                    vdf_tmp = func.predict(vdf=vdf_tmp, X=columns, name="verticapy_predict",)
        dataset = vdf_tmp[[columns[1], columns[0], y]].sort(columns).to_numpy()
        i, y_start, y_new = 0, dataset[0][1], dataset[0][1]
        n = len(dataset)
        X, Y, Z = [], [], []
        while i < n:
            x_tmp, y_tmp, z_tmp = [], [], []
            j = 0
            while y_start == y_new and i < n and j < nbins:
                x_tmp += [float(dataset[i][0])]
                y_tmp += [float(dataset[i][1])]
                z_tmp += [float(dataset[i][2])]
                y_new = dataset[i][1]
                j += 1
                i += 1
                if j == nbins:
                    while y_start == y_new and i < n:
                        y_new = dataset[i][1]
                        i += 1
            y_start = y_new
            X += [x_tmp]
            Y += [y_tmp]
            Z += [z_tmp]
        X, Y, Z = np.array(Y), np.array(X), np.array(Z)
    if not(ax):
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6)
    else:
        fig = plt
    param = {"linewidths": 0.5, "levels": 14, "colors": 'k',}
    param = updated_dict(param, style_kwds)
    if "cmap" in param:
        del param["cmap"]
    ax.contour(X, Y, Z, **param,)
    param = {"cmap": gen_cmap([gen_colors()[2], "#FFFFFF", gen_colors()[0],]), "levels": 14,}
    param = updated_dict(param, style_kwds)
    for elem in ["colors", "color", "linewidths", "linestyles",]:
        if elem in param:
            del param[elem]
    cp = ax.contourf(X, Y, Z, **param,)
    fig.colorbar(cp).set_label(cbar_title)
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    return ax

# ---#
def compute_plot_variables(
    vdf,
    method: str = "density",
    of: str = "",
    max_cardinality: int = 6,
    bins: int = 0,
    h: float = 0,
    pie: bool = False,
):
    other_columns = ""
    if method.lower() == "median":
        method = "50%"
    elif method.lower() == "mean":
        method = "avg"
    if (
        method.lower() not in ["avg", "min", "max", "sum", "density", "count"]
        and "%" != method[-1]
    ) and of:
        raise ParameterError(
            "Parameter 'of' must be empty when using customized aggregations."
        )
    if (
        (method.lower() in ["avg", "min", "max", "sum"])
        or (method.lower() and method[-1] == "%")
    ) and (of):
        if method.lower() in ["avg", "min", "max", "sum"]:
            aggregate = "{}({})".format(method.upper(), str_column(of))
        elif method and method[-1] == "%":
            aggregate = "APPROXIMATE_PERCENTILE({} USING PARAMETERS percentile = {})".format(
                str_column(of), float(method[0:-1]) / 100
            )
        else:
            raise ParameterError(
                "The parameter 'method' must be in [avg|mean|min|max|sum|median|q%] or a customized aggregation. Found {}.".format(
                    method
                )
            )
    elif method.lower() in ["density", "count"]:
        aggregate = "count(*)"
    elif isinstance(method, str):
        aggregate = method
        other_columns = ", " + ", ".join(
            vdf.parent.get_columns(exclude_columns=[vdf.alias])
        )
    else:
        raise ParameterError(
            "The parameter 'method' must be in [avg|mean|min|max|sum|median|q%] or a customized aggregation. Found {}.".format(
                method
            )
        )
    # depending on the cardinality, the type, the vColumn can be treated as categorical or not
    cardinality, count, is_numeric, is_date, is_categorical = (
        vdf.nunique(True),
        vdf.parent.shape()[0],
        vdf.isnum(),
        (vdf.category() == "date"),
        False,
    )
    rotation = 0 if ((is_numeric) and (cardinality > max_cardinality)) else 90
    # case when categorical
    if (((cardinality <= max_cardinality) or not (is_numeric)) or pie) and not (
        is_date
    ):
        if (is_numeric) and not (pie):
            query = "SELECT {}, {} FROM {} WHERE {} IS NOT NULL GROUP BY {} ORDER BY {} ASC LIMIT {}".format(
                vdf.alias,
                aggregate,
                vdf.parent.__genSQL__(),
                vdf.alias,
                vdf.alias,
                vdf.alias,
                max_cardinality,
            )
        else:
            table = vdf.parent.__genSQL__()
            if (pie) and (is_numeric):
                enum_trans = (
                    vdf.discretize(h=h, return_enum_trans=True)[0].replace(
                        "{}", vdf.alias
                    )
                    + " AS "
                    + vdf.alias
                )
                if of:
                    enum_trans += " , " + of
                table = "(SELECT {} FROM {}) enum_table".format(
                    enum_trans + other_columns, table
                )
            query = "(SELECT {} AS {}, {} FROM {} GROUP BY {} ORDER BY 2 DESC LIMIT {})".format(
                convert_special_type(vdf.category(), True, vdf.alias),
                vdf.alias,
                aggregate,
                table,
                convert_special_type(vdf.category(), True, vdf.alias),
                max_cardinality,
            )
            if cardinality > max_cardinality:
                query += (
                    " UNION (SELECT 'Others', {} FROM {} WHERE {} NOT IN "
                    + "(SELECT {} FROM {} GROUP BY {} ORDER BY {} DESC LIMIT {}))"
                )
                query = query.format(
                    aggregate,
                    table,
                    vdf.alias,
                    vdf.alias,
                    table,
                    vdf.alias,
                    aggregate,
                    max_cardinality,
                )
        vdf.__executeSQL__(query, title="Compute the histogram heights")
        query_result = vdf.parent._VERTICAPY_VARIABLES_["cursor"].fetchall()
        if query_result[-1][1] == None:
            del query_result[-1]
        z = [item[0] for item in query_result]
        y = (
            [item[1] / float(count) if item[1] != None else 0 for item in query_result]
            if (method.lower() == "density")
            else [item[1] if item[1] != None else 0 for item in query_result]
        )
        x = [0.4 * i + 0.2 for i in range(0, len(y))]
        h = 0.39
        is_categorical = True
    # case when date
    elif is_date:
        if (h <= 0) and (bins <= 0):
            h = vdf.numh()
        elif bins > 0:
            query = "SELECT DATEDIFF('second', MIN({}), MAX({})) FROM ".format(
                vdf.alias, vdf.alias
            )
            vdf.__executeSQL__(query=query, title="Compute the histogram interval")
            query_result = vdf.parent._VERTICAPY_VARIABLES_["cursor"].fetchone()
            h = float(query_result[0]) / bins
        min_date = vdf.min()
        converted_date = "DATEDIFF('second', '{}', {})".format(min_date, vdf.alias)
        query = "SELECT FLOOR({} / {}) * {}, {} FROM {} WHERE {} IS NOT NULL GROUP BY 1 ORDER BY 1".format(
            converted_date, h, h, aggregate, vdf.parent.__genSQL__(), vdf.alias
        )
        vdf.__executeSQL__(query=query, title="Compute the histogram heights")
        query_result = vdf.parent._VERTICAPY_VARIABLES_["cursor"].fetchall()
        x = [float(item[0]) for item in query_result]
        y = (
            [item[1] / float(count) for item in query_result]
            if (method.lower() == "density")
            else [item[1] for item in query_result]
        )
        query = ""
        for idx, item in enumerate(query_result):
            query += " UNION (SELECT TIMESTAMPADD('second' , {}, '{}'::timestamp))".format(
                math.floor(h * idx), min_date
            )
        query = query[7:-1] + ")"
        h = 0.94 * h
        vdf.parent._VERTICAPY_VARIABLES_["cursor"].execute(query)
        query_result = vdf.parent._VERTICAPY_VARIABLES_["cursor"].fetchall()
        z = [item[0] for item in query_result]
        z.sort()
        is_categorical = True
    # case when numerical
    else:
        if (h <= 0) and (bins <= 0):
            h = vdf.numh()
        elif bins > 0:
            h = float(vdf.max() - vdf.min()) / bins
        if (vdf.ctype == "int") or (h == 0):
            h = max(1.0, h)
        query = "SELECT FLOOR({} / {}) * {}, {} FROM {} WHERE {} IS NOT NULL GROUP BY 1 ORDER BY 1"
        query = query.format(
            vdf.alias, h, h, aggregate, vdf.parent.__genSQL__(), vdf.alias
        )
        vdf.__executeSQL__(query=query, title="Compute the histogram heights")
        query_result = vdf.parent._VERTICAPY_VARIABLES_["cursor"].fetchall()
        y = (
            [item[1] / float(count) for item in query_result]
            if (method.lower() == "density")
            else [item[1] for item in query_result]
        )
        x = [float(item[0]) + h / 2 for item in query_result]
        h = 0.94 * h
        z = None
    return [x, y, z, h, is_categorical]


# ---#
def gen_cmap(color: str = "", reverse: bool = False):
    if not (color):
        cm1 = LinearSegmentedColormap.from_list(
            "vml", ["#FFFFFF", gen_colors()[0]], N=1000
        )
        cm2 = LinearSegmentedColormap.from_list(
            "vml", [gen_colors()[1], "#FFFFFF", gen_colors()[0]], N=1000
        )
        return (cm1, cm2)
    else:
        if isinstance(color, list):
            return LinearSegmentedColormap.from_list("vml", color, N=1000)
        elif reverse:
            return LinearSegmentedColormap.from_list("vml", [color, "#FFFFFF"], N=1000)
        else:
            return LinearSegmentedColormap.from_list("vml", ["#FFFFFF", color], N=1000)


# ---#
def gen_colors():
    if not (verticapy.options["colors"]) or not (
        isinstance(verticapy.options["colors"], list)
    ):
        if verticapy.options["color_style"] == "sunset":
            colors = ["#36688D", "#F3CD05", "#F49F05", "#F18904", "#BDA589",]
        elif verticapy.options["color_style"] == "rgb":
            colors = ["red", "green", "blue", "orange", "yellow", "gray",]
        elif verticapy.options["color_style"] == "retro":
            colors = ["#A7414A", "#282726", "#6A8A82", "#A37C27", "#563838",]
        elif verticapy.options["color_style"] == "shimbg":
            colors = ["#0444BF", "#0584F2", "#0AAFF1", "#EDF259", "#A79674",]
        elif verticapy.options["color_style"] == "swamp":
            colors = ["#6465A5", "#6975A6", "#F3E96B", "#F28A30", "#F05837",]
        elif verticapy.options["color_style"] == "med":
            colors = ["#ABA6BF", "#595775", "#583E2E", "#F1E0D6", "#BF9887",]
        elif verticapy.options["color_style"] == "orchid":
            colors = ["#192E5B", "#1D65A6", "#72A2C0", "#00743F", "#F2A104",]
        elif verticapy.options["color_style"] == "magenta":
            colors = ["#DAA2DA", "#DBB4DA", "#DE8CF0", "#BED905", "#93A806",]
        elif verticapy.options["color_style"] == "orange":
            colors = ["#A3586D", "#5C4A72", "#F3B05A", "#F4874B", "#F46A4E",]
        elif verticapy.options["color_style"] == "vintage":
            colors = ["#80ADD7", "#0ABDA0", "#EBF2EA", "#D4DCA9", "#BF9D7A",]
        elif verticapy.options["color_style"] == "vivid":
            colors = ["#C0334D", "#D6618F", "#F3D4A0", "#F1931B", "#8F715B",]
        elif verticapy.options["color_style"] == "berries":
            colors = ["#BB1924", "#EE6C81", "#F092A5", "#777CA8", "#AFBADC",]
        elif verticapy.options["color_style"] == "refreshing":
            colors = ["#003D73", "#0878A4", "#1ECFD6", "#EDD170", "#C05640",]
        elif verticapy.options["color_style"] == "summer":
            colors = ["#728CA3", "#73C0F4", "#E6EFF3", "#F3E4C6", "#8F4F06",]
        elif verticapy.options["color_style"] == "tropical":
            colors = ["#7B8937", "#6B7436", "#F4D9C1", "#D72F01", "#F09E8C",]
        elif verticapy.options["color_style"] == "india":
            colors = ["#F1445B", "#65734B", "#94A453", "#D9C3B1", "#F03625",]
        else:
            colors = ["#FE5016", "#263133", "#0073E7", "#FDE159", "#33C180", "#FF454F",]
        all_colors = [item for item in plt_colors.cnames]
        shuffle(all_colors)
        for c in all_colors:
            if c not in colors:
                colors += [c]
        return colors
    else:
        return verticapy.options["colors"]


# ---#
def hexbin(
    vdf,
    columns: list,
    method: str = "count",
    of: str = "",
    bbox: list = [],
    img: str = "",
    ax=None,
    **style_kwds,
):
    if len(columns) != 2:
        raise ParameterError(
            "The parameter 'columns' must be exactly of size 2 to draw the hexbin"
        )
    if method.lower() == "mean":
        method = "avg"
    if (
        (method.lower() in ["avg", "min", "max", "sum"])
        and (of)
        and ((of in vdf.get_columns()) or (str_column(of) in vdf.get_columns()))
    ):
        aggregate = "{}({})".format(method, of)
        others_aggregate = method
        if method.lower() == "avg":
            reduce_C_function = statistics.mean
        elif method.lower() == "min":
            reduce_C_function = min
        elif method.lower() == "max":
            reduce_C_function = max
        elif method.lower() == "sum":
            reduce_C_function = sum
    elif method.lower() in ("count", "density"):
        aggregate = "count(*)"
        reduce_C_function = sum
    else:
        raise ParameterError(
            "The parameter 'method' must be in [avg|mean|min|max|sum|median]"
        )
    count = vdf.shape()[0]
    if method.lower() == "density":
        over = "/" + str(float(count))
    else:
        over = ""
    query = "SELECT {}, {}, {}{} FROM {} GROUP BY {}, {}".format(
        columns[0],
        columns[1],
        aggregate,
        over,
        vdf.__genSQL__(),
        columns[0],
        columns[1],
    )
    query_result = vdf.__executeSQL__(
        query=query, title="Group all the elements for the Hexbin Plot"
    ).fetchall()
    column1, column2, column3 = [], [], []
    for item in query_result:
        if (item[0] != None) and (item[1] != None) and (item[2] != None):
            column1 += [float(item[0])] * 2
            column2 += [float(item[1])] * 2
            if reduce_C_function in [min, max, statistics.mean]:
                column3 += [float(item[2])] * 2
            else:
                column3 += [float(item[2]) / 2] * 2
    if not (ax):
        fig, ax = plt.subplots()
        if isnotebook():
            fig.set_size_inches(9, 7)
        ax.set_facecolor("white")
    else:
        fig = plt
    if bbox:
        ax.set_xlim(bbox[0], bbox[1])
        ax.set_ylim(bbox[2], bbox[3])
    if img:
        im = plt.imread(img)
        if not (bbox):
            bbox = (min(column1), max(column1), min(column2), max(column2))
            ax.set_xlim(bbox[0], bbox[1])
            ax.set_ylim(bbox[2], bbox[3])
        ax.imshow(im, extent=bbox)
    ax.set_ylabel(columns[1])
    ax.set_xlabel(columns[0])
    param = {"cmap": gen_cmap()[0], "gridsize": 10, "mincnt": 1, "edgecolors": None}
    imh = ax.hexbin(
        column1,
        column2,
        C=column3,
        reduce_C_function=reduce_C_function,
        **updated_dict(param, style_kwds),
    )
    if method.lower() == "density":
        fig.colorbar(imh).set_label(method)
    else:
        fig.colorbar(imh).set_label(aggregate)
    return ax


# ---#
def hist(
    vdf,
    method: str = "density",
    of=None,
    max_cardinality: int = 6,
    bins: int = 0,
    h: float = 0,
    ax=None,
    **style_kwds,
):
    x, y, z, h, is_categorical = compute_plot_variables(
        vdf, method, of, max_cardinality, bins, h,
    )
    is_numeric = vdf.isnum()
    if not (ax):
        fig, ax = plt.subplots()
        if isnotebook():
            fig.set_size_inches(min(int(len(x) / 1.8) + 1, 600), 6)
        ax.set_axisbelow(True)
        ax.yaxis.grid()
    param = {"color": gen_colors()[0], "alpha": 0.86}
    ax.bar(
        x, y, h, **updated_dict(param, style_kwds),
    )
    ax.set_xlabel(vdf.alias)
    if is_categorical:
        if not (is_numeric):
            new_z = []
            for item in z:
                new_z += [item[0:47] + "..."] if (len(str(item)) > 50) else [item]
        else:
            new_z = z
        ax.set_xticks(x)
        ax.set_xticklabels(new_z, rotation=90)
    else:
        L = [elem - round(h / 2 / 0.94, 10) for elem in x]
        ax.set_xticks(L)
        ax.set_xticklabels(L, rotation=90)
    if method.lower() == "density":
        ax.set_ylabel("Density")
    elif (
        method.lower() in ["avg", "min", "max", "sum", "mean"] or ("%" == method[-1])
    ) and (of != None):
        aggregate = "{}({})".format(method, of)
        ax.set_ylabel(method)
    elif method.lower() == "count":
        ax.set_ylabel("Frequency")
    else:
        ax.set_ylabel(method)
    return ax


# ---#
def hist2D(
    vdf,
    columns: list,
    method="density",
    of: str = "",
    max_cardinality: tuple = (6, 6),
    h: tuple = (None, None),
    stacked: bool = False,
    ax=None,
    **style_kwds,
):
    colors = gen_colors()
    all_columns = vdf.pivot_table(
        columns, method=method, of=of, h=h, max_cardinality=max_cardinality, show=False
    ).values
    all_columns = [[column] + all_columns[column] for column in all_columns]
    n, m = len(all_columns), len(all_columns[0])
    n_groups = m - 1
    bar_width = 0.5
    if not (ax):
        fig, ax = plt.subplots()
        if isnotebook():
            fig.set_size_inches(min(600, 3 * m) / 2 + 1, 6)
        ax.set_axisbelow(True)
        ax.yaxis.grid()
    for i in range(1, n):
        current_column = all_columns[i][1:m]
        for idx, item in enumerate(current_column):
            try:
                current_column[idx] = float(item)
            except:
                current_column[idx] = 0
        current_label = str(all_columns[i][0])
        param = {
            "alpha": 0.86,
            "color": colors[(i - 1) % len(colors)],
        }
        if stacked:
            if i == 1:
                last_column = [0 for item in all_columns[i][1:m]]
            else:
                for idx, item in enumerate(all_columns[i - 1][1:m]):
                    try:
                        last_column[idx] += float(item)
                    except:
                        last_column[idx] += 0
            ax.bar(
                [elem for elem in range(n_groups)],
                current_column,
                bar_width,
                label=current_label,
                bottom=last_column,
                **updated_dict(param, style_kwds, i - 1),
            )
        else:
            ax.bar(
                [elem + (i - 1) * bar_width / (n - 1) for elem in range(n_groups)],
                current_column,
                bar_width / (n - 1),
                label=current_label,
                **updated_dict(param, style_kwds, i - 1),
            )
    if stacked:
        ax.set_xticks([elem for elem in range(n_groups)])
        ax.set_xticklabels(all_columns[0][1:m], rotation=90)
    else:
        ax.set_xticks(
            [
                elem + bar_width / 2 - bar_width / 2 / (n - 1)
                for elem in range(n_groups)
            ],
        )
        ax.set_xticklabels(all_columns[0][1:m], rotation=90)
    ax.set_xlabel(columns[0])
    if method.lower() == "mean":
        method = "avg"
    if method.lower() == "density":
        ax.set_ylabel("Density")
    elif (method.lower() in ["avg", "min", "max", "sum"]) and (of != None):
        ax.set_ylabel("{}({})".format(method, of))
    elif method.lower() == "count":
        ax.set_ylabel("Frequency")
    else:
        ax.set_ylabel(method)
    ax.legend(title=columns[1], loc="center left", bbox_to_anchor=[1, 0.5])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    return ax


# ---#
def nested_pie(
    vdf,
    columns: list,
    max_cardinality: tuple = None,
    h: tuple = None,
    ax=None,
    **style_kwds,
):
    wedgeprops = dict(width=0.3, edgecolor="w")
    tmp_style = {}
    for elem in style_kwds:
        if elem not in ("color", "colors", "wedgeprops"):
            tmp_style[elem] = style_kwds[elem]
    if "wedgeprops" in style_kwds:
        wedgeprops = style_kwds["wedgeprops"]
    if "colors" in style_kwds:
        colors, n = style_kwds["colors"], len(columns)
    elif "color" in style_kwds:
        colors, n = style_kwds["color"], len(columns)
    else:
        colors, n = gen_colors(), len(columns)
    m, k = len(colors), 0
    if isinstance(h, (int, float, type(None))):
        h = (h,) * n
    if isinstance(max_cardinality, (int, float, type(None))):
        if max_cardinality == None:
            max_cardinality = (6,) * n
        else:
            max_cardinality = (max_cardinality,) * n
    vdf_tmp = vdf[columns]
    for idx, column in enumerate(columns):
        vdf_tmp[column].discretize(h=h[idx])
        vdf_tmp[column].discretize(method="topk", k=max_cardinality[idx])
    if not (ax):
        fig, ax = plt.subplots()
        if isnotebook():
            fig.set_size_inches(8, 6)
    all_colors_dict, all_categories, all_categories_col = {}, {}, []
    for i in range(0, n):
        if i in [0]:
            pctdistance = 0.77
        elif i > 2:
            pctdistance = 0.9
        elif i > 1:
            pctdistance = 0.88
        else:
            pctdistance = 0.85
        result = (
            vdf_tmp.groupby(columns[: n - i], ["COUNT(*) AS cnt"])
            .sort(columns[: n - i])
            .to_numpy()
            .T
        )
        all_colors_dict[i] = {}
        all_categories[i] = list(dict.fromkeys(result[-2]))
        all_categories_col += [columns[n - i - 1]]
        for elem in all_categories[i]:
            all_colors_dict[i][elem] = colors[k % m]
            k += 1
        group = [int(elem) for elem in result[-1]]
        tmp_colors = [all_colors_dict[i][j] for j in result[-2]]
        if len(group) > 16:
            autopct = None
        else:
            autopct = "%1.1f%%"
        ax.pie(
            group,
            radius=0.3 * (i + 2),
            colors=tmp_colors,
            wedgeprops=wedgeprops,
            autopct=autopct,
            pctdistance=pctdistance,
            **tmp_style,
        )
        legend_colors = [all_colors_dict[i][elem] for elem in all_colors_dict[i]]
        if n == 1:
            bbox_to_anchor = [0.5, 1]
        elif n < 4:
            bbox_to_anchor = [0.4 + n * 0.23, 0.5 + 0.15 * i]
        else:
            bbox_to_anchor = [0.2 + n * 0.23, 0.5 + 0.15 * i]
        legend = plt.legend(
            [Line2D([0], [0], color=color, lw=4) for color in legend_colors],
            all_categories[i],
            bbox_to_anchor=bbox_to_anchor,
            loc="upper left",
            title=all_categories_col[i],
            labelspacing=1,
            ncol=len(all_categories[i]),
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.gca().add_artist(legend)
    return ax


# ---#
def multiple_hist(
    vdf,
    columns: list,
    method: str = "density",
    of: str = "",
    h: float = 0,
    ax=None,
    **style_kwds,
):
    colors = gen_colors()
    if len(columns) > 5:
        raise ParameterError(
            "The number of column must be <= 5 to use 'multiple_hist' method"
        )
    else:
        if not (ax):
            fig, ax = plt.subplots()
            if isnotebook():
                fig.set_size_inches(8, 6)
            ax.set_axisbelow(True)
            ax.yaxis.grid()
        alpha, all_columns, all_h = 1, [], []
        if h <= 0:
            for idx, column in enumerate(columns):
                all_h += [vdf[column].numh()]
            h = min(all_h)
        for idx, column in enumerate(columns):
            if vdf[column].isnum():
                [x, y, z, h, is_categorical] = compute_plot_variables(
                    vdf[column], method=method, of=of, max_cardinality=1, h=h
                )
                h = h / 0.94
                param = {"color": colors[idx % len(colors)]}
                plt.bar(
                    x,
                    y,
                    h,
                    label=column,
                    alpha=alpha,
                    **updated_dict(param, style_kwds, idx),
                )
                alpha -= 0.2
                all_columns += [columns[idx]]
            else:
                if vdf._VERTICAPY_VARIABLES_["display"]["print_info"]:
                    warning_message = "The Virtual Column {} is not numerical. Its histogram will not be draw.".format(
                        column
                    )
                    warnings.warn(warning_message, Warning)
        ax.set_xlabel(", ".join(all_columns))
        if method.lower() == "density":
            ax.set_ylabel("Density")
        elif (
            method.lower() in ["avg", "min", "max", "sum", "mean"]
            or ("%" == method[-1])
        ) and (of):
            ax.set_ylabel(method + "(" + of + ")")
        elif method.lower() == "count":
            ax.set_ylabel("Frequency")
        else:
            ax.set_ylabel(method)
        ax.legend(title="columns", loc="center left", bbox_to_anchor=[1, 0.5])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax


# ---#
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
                warning_message = "The Virtual Column {} is not numerical.\nIt will be ignored.".format(
                    column
                )
                warnings.warn(warning_message, Warning)
            columns.remove(column)
    if not (columns):
        raise EmptyParameter("No numerical columns found to draw the multi TS plot")
    colors = gen_colors()
    query = "SELECT {}, {} FROM {} WHERE {} IS NOT NULL".format(
        order_by, ", ".join(columns), vdf.__genSQL__(), order_by
    )
    query += (
        " AND {} > '{}'".format(order_by, order_by_start) if (order_by_start) else ""
    )
    query += " AND {} < '{}'".format(order_by, order_by_end) if (order_by_end) else ""
    query += " AND " + " AND ".join(
        ["{} IS NOT NULL".format(column) for column in columns]
    )
    query += " ORDER BY {}".format(order_by)
    vdf.__executeSQL__(query=query, title="Select the needed points to draw the curves")
    query_result = vdf._VERTICAPY_VARIABLES_["cursor"].fetchall()
    order_by_values = [item[0] for item in query_result]
    try:
        if isinstance(order_by_values[0], str):
            from dateutil.parser import parse

            order_by_values = [parse(elem) for elem in order_by_values]
    except:
        pass
    alpha = 0.3
    if not (ax):
        fig, ax = plt.subplots()
        if isnotebook():
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
        if kind in ("line", "step"):
            color = colors[i]
            if len(order_by_values) < 20:
                param = {
                    "marker": "o",
                    "markevery": 0.05,
                    "markerfacecolor": colors[i],
                    "markersize": 7,
                    "markeredgecolor": "black",
                }
            param["label"] = columns[i]
            param["linewidth"] = 2
        elif kind == "area_percent":
            color = "white"
            if len(order_by_values) < 20:
                param = {
                    "marker": "o",
                    "markevery": 0.05,
                    "markerfacecolor": colors[i],
                    "markersize": 7,
                    "markeredgecolor": "white",
                }
        else:
            color = "black"
            if len(order_by_values) < 20:
                param = {
                    "marker": "o",
                    "markevery": 0.05,
                    "markerfacecolor": colors[i],
                    "markersize": 7,
                    "markeredgecolor": "black",
                }
        param["color"] = color
        if "color" in style_kwds and len(order_by_values) < 20:
            param["markerfacecolor"] = color_dict(style_kwds, i)
        if kind == "step":
            ax.step(
                order_by_values, points, **param,
            )
        else:
            ax.plot(
                order_by_values, points, **param,
            )
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
                order_by_values, prec, points, label=columns[i], **tmp_style,
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


# ---#
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
        if isnotebook():
            fig.set_size_inches(8, 6)
        ax.grid()
    for i, elem in enumerate(Y):
        if labels:
            label = labels[i]
        else:
            label = ""
        if plot_median:
            alpha1, alpha2 = 0.3, 0.5
        else:
            alpha1, alpha2 = 0.5, 0.9
        param = {"facecolor": color_dict(style_kwds, i)}
        ax.fill_between(X, elem[0], elem[2], alpha=alpha1, **param)
        param = {"color": color_dict(style_kwds, i)}
        ax.plot(
            X, elem[0], alpha=alpha2, **updated_dict(param, style_kwds, i),
        )
        ax.plot(
            X, elem[2], alpha=alpha2, **updated_dict(param, style_kwds, i),
        )
        if plot_median:
            ax.plot(
                X, elem[1], label=label, **updated_dict(param, style_kwds, i),
            )
        if (not (without_scatter) or len(X) < 20) and plot_median:
            ax.scatter(
                X, elem[1], c="white", marker="o", s=60, edgecolors="black", zorder=3
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


# ---#
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
    query = "SELECT {}, APPROXIMATE_PERCENTILE({} USING PARAMETERS percentile = {}), APPROXIMATE_MEDIAN({}), APPROXIMATE_PERCENTILE({} USING PARAMETERS percentile = {}) FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL".format(
        order_by,
        vdf.alias,
        q[0],
        vdf.alias,
        vdf.alias,
        q[1],
        vdf.parent.__genSQL__(),
        order_by,
        vdf.alias,
    )
    query += (
        " AND {} > '{}'".format(order_by, order_by_start) if (order_by_start) else ""
    )
    query += " AND {} < '{}'".format(order_by, order_by_end) if (order_by_end) else ""
    query += " GROUP BY 1 ORDER BY 1"
    query_result = vdf.__executeSQL__(
        query=query, title="Select points to draw the curve"
    ).fetchall()
    order_by_values = [item[0] for item in query_result]
    try:
        if isinstance(order_by_values[0], str):
            from dateutil.parser import parse

            order_by_values = [parse(elem) for elem in order_by_values]
    except:
        pass
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


# ---#
def outliers_contour_plot(
    vdf,
    columns: list,
    color: str = "orange",
    outliers_color: str = "black",
    inliers_color: str = "white",
    inliers_border_color: str = "red",
    cmap: str = None,
    max_nb_points: int = 1000,
    threshold: float = 3.0,
    ax=None,
    **style_kwds,
):
    if not (cmap):
        cmap = gen_cmap(gen_colors()[2])
    all_agg = vdf.agg(["avg", "std", "min", "max"], columns)
    xlist = np.linspace(all_agg["min"][0], all_agg["max"][0], 1000)
    if not (ax):
        fig, ax = plt.subplots()
        if isnotebook():
            fig.set_size_inches(8, 6)
    if len(columns) == 1:
        if isinstance(cmap, str):
            cmap = plt.cm.get_cmap(cmap)
        min_zscore = (all_agg["min"][0] - all_agg["avg"][0]) / (all_agg["std"][0])
        max_zscore = (all_agg["max"][0] - all_agg["avg"][0]) / (all_agg["std"][0])
        for i in range(int(min_zscore) - 1, int(max_zscore) + 1):
            if abs(i) < threshold:
                alpha = 0
            else:
                alpha = (abs(i) - threshold) / (int(max_zscore) + 1 - 3)
            ax.fill_between(
                [all_agg["min"][0], all_agg["max"][0]],
                [i, i],
                [i + 1, i + 1],
                facecolor=cmap(10000),
                alpha=alpha,
            )
        ax.fill_between(
            [all_agg["min"][0], all_agg["max"][0]],
            [-threshold, -threshold],
            [threshold, threshold],
            facecolor=color,
        )
        ax.plot(
            [all_agg["min"][0], all_agg["max"][0]],
            [-threshold, -threshold],
            color=inliers_border_color,
        )
        ax.plot(
            [all_agg["min"][0], all_agg["max"][0]],
            [threshold, threshold],
            color=inliers_border_color,
        )
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        ax.set_xlabel(columns[0])
        ax.set_ylabel("ZSCORE")
        ax.set_xlim(all_agg["min"][0], all_agg["max"][0])
        ax.set_ylim(int(min_zscore) - 1, int(max_zscore) + 1)
        vdf_temp = vdf[columns]
        vdf_temp["ZSCORE"] = (vdf_temp[columns[0]] - all_agg["avg"][0]) / all_agg[
            "std"
        ][0]
        vdf_temp["ZSCORE"] = "ZSCORE + 1.5 * RANDOM()"
        search1 = "ZSCORE > {}".format(threshold)
        search2 = "ZSCORE <= {}".format(threshold)
        scatter2D(
            vdf_temp.search(search1),
            [columns[0], "ZSCORE"],
            max_nb_points=max_nb_points,
            ax=ax,
            color=outliers_color,
            **style_kwds,
        )
        scatter2D(
            vdf_temp.search(search2),
            [columns[0], "ZSCORE"],
            max_nb_points=max_nb_points,
            ax=ax,
            color=inliers_color,
            **style_kwds,
        )
        bbox_to_anchor = [1, 0.5]
    elif len(columns) == 2:
        ylist = np.linspace(all_agg["min"][1], all_agg["max"][1], 1000)
        X, Y = np.meshgrid(xlist, ylist)
        Z = np.sqrt(
            ((X - all_agg["avg"][0]) / all_agg["std"][0]) ** 2
            + ((Y - all_agg["avg"][1]) / all_agg["std"][1]) ** 2
        )
        cp = ax.contourf(X, Y, Z, colors=color)
        ax.contour(
            X, Y, Z, levels=[threshold], linewidths=2, colors=inliers_border_color
        )
        cp = ax.contourf(X, Y, Z, cmap=cmap, levels=np.linspace(threshold, Z.max(), 8))
        fig.colorbar(cp).set_label("ZSCORE")
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        search1 = "ABS(({} - {}) / {}) > {} OR ABS(({} - {}) / {}) > {}".format(
            columns[0],
            all_agg["avg"][0],
            all_agg["std"][0],
            threshold,
            columns[1],
            all_agg["avg"][1],
            all_agg["std"][1],
            threshold,
        )
        search2 = "ABS(({} - {}) / {}) <= {} AND ABS(({} - {}) / {}) <= {}".format(
            columns[0],
            all_agg["avg"][0],
            all_agg["std"][0],
            threshold,
            columns[1],
            all_agg["avg"][1],
            all_agg["std"][1],
            threshold,
        )
        bbox_to_anchor = [1, 0.5]
        scatter2D(
            vdf.search(search1),
            columns,
            max_nb_points=max_nb_points,
            ax=ax,
            color=outliers_color,
            **style_kwds,
        )
        scatter2D(
            vdf.search(search2),
            columns,
            max_nb_points=max_nb_points,
            ax=ax,
            color=inliers_color,
            **style_kwds,
        )
    ax.legend(
        [
            Line2D([0], [0], color=inliers_border_color, lw=4),
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                label="Scatter",
                markerfacecolor=inliers_color,
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                label="Scatter",
                markerfacecolor=outliers_color,
                markersize=8,
            ),
        ],
        ["threshold", "inliers", "outliers"],
        loc="center left",
        bbox_to_anchor=bbox_to_anchor,
        labelspacing=1,
    )
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    return ax


# ---#
def pie(
    vdf,
    method: str = "density",
    of=None,
    max_cardinality: int = 6,
    h: float = 0,
    donut: bool = False,
    rose: bool = False,
    ax=None,
    **style_kwds,
):
    colors = gen_colors()
    x, y, z, h, is_categorical = compute_plot_variables(
        vdf, max_cardinality=max_cardinality, method=method, of=of, pie=True
    )
    z.reverse()
    y.reverse()
    explode = [0 for i in y]
    explode[max(zip(y, range(len(y))))[1]] = 0.13
    current_explode = 0.15
    total_count = sum(y)
    for idx, item in enumerate(y):
        if (item < 0.05) or ((item > 1) and (float(item) / float(total_count) < 0.05)):
            current_explode = min(0.9, current_explode * 1.4)
            explode[idx] = current_explode
    if method.lower() == "density":
        autopct = "%1.1f%%"
    else:

        def make_autopct(values, category):
            def my_autopct(pct):
                total = sum(values)
                val = float(pct) * float(total) / 100.0
                if category == "int":
                    val = int(round(val))
                    return "{v:d}".format(v=val)
                else:
                    return "{v:f}".format(v=val)

            return my_autopct

        if (method.lower() in ["sum", "count"]) or (
            (method.lower() in ["min", "max"]) and (vdf.parent[of].category == "int")
        ):
            category = "int"
        else:
            category = None
        autopct = make_autopct(y, category)
    if not (rose):
        if not (ax):
            fig, ax = plt.subplots()
            if isnotebook():
                fig.set_size_inches(8, 6)
        param = {
            "autopct": autopct,
            "colors": colors,
            "shadow": True,
            "startangle": 290,
            "explode": explode,
            "textprops": {"color": "w"},
            "normalize": True,
        }
        if donut:
            param["wedgeprops"] = dict(width=0.4, edgecolor="w")
            param["explode"] = None
            param["pctdistance"] = 0.8
        ax.pie(
            y, labels=z, **updated_dict(param, style_kwds),
        )
        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(
            handles, labels, title=vdf.alias, loc="center left", bbox_to_anchor=[1, 0.5]
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    else:
        try:
            y, z = zip(*sorted(zip(y, z), key=lambda t: t[0]))
        except:
            pass
        N = len(z)
        width = 2 * np.pi / N
        rad = np.cumsum([width] * N)

        fig = plt.figure()
        if not (ax):
            ax = fig.add_subplot(111, polar=True)
        ax.grid(False)
        ax.spines["polar"].set_visible(False)
        ax.set_yticks([])
        ax.set_thetagrids([])
        ax.set_theta_zero_location("N")
        param = {
            "color": colors,
        }
        colors = updated_dict(param, style_kwds, -1)["color"]
        if isinstance(colors, str):
            colors = [colors] + gen_colors()
        else:
            colors = colors + gen_colors()
        style_kwds["color"] = colors
        ax.bar(
            rad, y, width=width, **updated_dict(param, style_kwds, -1),
        )
        for i in np.arange(N):
            ax.text(
                rad[i] + 0.1,
                [elem * 1.02 for elem in y][i],
                [round(elem, 2) for elem in y][i],
                rotation=rad[i] * 180 / np.pi,
                rotation_mode="anchor",
                alpha=1,
                color="black",
            )
        try:
            z, colors = zip(*sorted(zip(z, colors[:N]), key=lambda t: t[0]))
        except:
            pass
        ax.legend(
            [Line2D([0], [0], color=color,) for color in colors],
            z,
            bbox_to_anchor=[1.1, 0.5],
            loc="center left",
            title=vdf.alias,
            labelspacing=1,
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    return ax


# ---#
def pivot_table(
    vdf,
    columns: list,
    method: str = "count",
    of: str = "",
    h: tuple = (None, None),
    max_cardinality: tuple = (20, 20),
    show: bool = True,
    with_numbers: bool = True,
    ax=None,
    return_ax: bool = False,
    extent: list = [],
    **style_kwds,
):
    other_columns = ""
    if method.lower() == "median":
        method = "50%"
    elif method.lower() == "mean":
        method = "avg"
    if (
        method.lower() not in ["avg", "min", "max", "sum", "density", "count"]
        and "%" != method[-1]
    ) and of:
        raise ParameterError(
            "Parameter 'of' must be empty when using customized aggregations."
        )
    if (method.lower() in ["avg", "min", "max", "sum"]) and (of):
        aggregate = "{}({})".format(method.upper(), str_column(of))
    elif method.lower() and method[-1] == "%":
        aggregate = "APPROXIMATE_PERCENTILE({} USING PARAMETERS percentile = {})".format(
            str_column(of), float(method[0:-1]) / 100
        )
    elif method.lower() in ["density", "count"]:
        aggregate = "COUNT(*)"
    elif isinstance(method, str):
        aggregate = method
        other_columns = vdf.get_columns(exclude_columns=columns)
        other_columns = ", " + ", ".join(other_columns)
    else:
        raise ParameterError(
            "The parameter 'method' must be in [count|density|avg|mean|min|max|sum|q%]"
        )
    columns = [str_column(column) for column in columns]
    all_columns = []
    is_column_date = [False, False]
    timestampadd = ["", ""]
    for idx, column in enumerate(columns):
        is_numeric = vdf[column].isnum() and (vdf[column].nunique(True) > 2)
        is_date = vdf[column].isdate()
        where = []
        if is_numeric:
            if h[idx] == None:
                interval = vdf[column].numh()
                if interval > 0.01:
                    interval = round(interval, 2)
                elif interval > 0.0001:
                    interval = round(interval, 4)
                elif interval > 0.000001:
                    interval = round(interval, 6)
                interval = (
                    int(max(math.floor(interval), 1))
                    if (vdf[column].category() == "int")
                    else interval
                )
            else:
                interval = h[idx]
            if vdf[column].category() == "int":
                floor_end = "-1"
                interval = int(max(math.floor(interval), 1))
            else:
                floor_end = ""
            expr = "'[' || FLOOR({} / {}) * {} || ';' || (FLOOR({} / {}) * {} + {}{}) || ']'".format(
                column,
                interval,
                interval,
                column,
                interval,
                interval,
                interval,
                floor_end,
            )
            all_columns += (
                [expr]
                if (interval > 1) or (vdf[column].category() == "float")
                else ["FLOOR({}) || ''".format(column)]
            )
            order_by = "ORDER BY MIN(FLOOR({} / {}) * {}) ASC".format(
                column, interval, interval
            )
            where += ["{} IS NOT NULL".format(column)]
        elif is_date:
            interval = (
                vdf[column].numh() if (h[idx] == None) else max(math.floor(h[idx]), 1)
            )
            min_date = vdf[column].min()
            all_columns += [
                "FLOOR(DATEDIFF('second', '"
                + str(min_date)
                + "', "
                + column
                + ") / "
                + str(interval)
                + ") * "
                + str(interval)
            ]
            is_column_date[idx] = True
            timestampadd[idx] = (
                "TIMESTAMPADD('second', "
                + columns[idx]
                + "::int, '"
                + str(min_date)
                + "'::timestamp)"
            )
            order_by = "ORDER BY 1 ASC"
            where += ["{} IS NOT NULL".format(column)]
        else:
            all_columns += [column]
            order_by = "ORDER BY 1 ASC"
            distinct = vdf[column].topk(max_cardinality[idx]).values["index"]
            if len(distinct) < max_cardinality[idx]:
                where += [
                    "({} IN ({}))".format(
                        convert_special_type(vdf[column].category(), False, column),
                        ", ".join(
                            [
                                "'{}'".format(str(elem).replace("'", "''"))
                                for elem in distinct
                            ]
                        ),
                    )
                ]
            else:
                where += ["({} IS NOT NULL)".format(column)]
    where = " WHERE {}".format(" AND ".join(where))
    if len(columns) == 1:
        over = "/" + str(vdf.shape()[0]) if (method.lower() == "density") else ""
        query = "SELECT {} AS {}, {}{} FROM {}{} GROUP BY 1 {}".format(
            convert_special_type(vdf[columns[0]].category(), True, all_columns[-1]),
            columns[0],
            aggregate,
            over,
            vdf.__genSQL__(),
            where,
            order_by,
        )
        return to_tablesample(query, vdf._VERTICAPY_VARIABLES_["cursor"])
    alias = ", " + str_column(of) + " AS " + str_column(of) if of else ""
    aggr = ", " + of if (of) else ""
    subtable = "(SELECT {} AS {}, {} AS {}{}{} FROM {}{}) pivot_table".format(
        all_columns[0],
        columns[0],
        all_columns[1],
        columns[1],
        alias,
        other_columns,
        vdf.__genSQL__(),
        where,
    )
    if is_column_date[0] and not (is_column_date[1]):
        subtable = "(SELECT {} AS {}, {}{}{} FROM {}{}) pivot_table_date".format(
            timestampadd[0],
            columns[0],
            columns[1],
            aggr,
            other_columns,
            subtable,
            where,
        )
    elif is_column_date[1] and not (is_column_date[0]):
        subtable = "(SELECT {}, {} AS {}{}{} FROM {}{}) pivot_table_date".format(
            columns[0],
            timestampadd[1],
            columns[1],
            aggr,
            other_columns,
            subtable,
            where,
        )
    elif is_column_date[1] and is_column_date[0]:
        subtable = "(SELECT {} AS {}, {} AS {}{}{} FROM {}{}) pivot_table_date".format(
            timestampadd[0],
            columns[0],
            timestampadd[1],
            columns[1],
            aggr,
            other_columns,
            subtable,
            where,
        )
    over = "/" + str(vdf.shape()[0]) if (method.lower() == "density") else ""
    cast = []
    for column in columns:
        cast += [convert_special_type(vdf[column].category(), True, column)]
    query = "SELECT {} AS {}, {} AS {}, {}{} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL GROUP BY {}, {} ORDER BY {}, {} ASC".format(
        cast[0],
        columns[0],
        cast[1],
        columns[1],
        aggregate,
        over,
        subtable,
        columns[0],
        columns[1],
        columns[0],
        columns[1],
        columns[0],
        columns[1],
    )
    vdf.__executeSQL__(
        query=query, title="Group the features to compute the pivot table"
    )
    query_result = vdf.__executeSQL__(
        query=query, title="Group the features to compute the pivot table"
    ).fetchall()
    # Column0 sorted categories
    all_column0_categories = list(set([str(item[0]) for item in query_result]))
    all_column0_categories.sort()
    try:
        try:
            order = []
            for item in all_column0_categories:
                order += [float(item.split(";")[0].split("[")[1])]
        except:
            order = [float(item) for item in all_column0_categories]
        all_column0_categories = [
            x for _, x in sorted(zip(order, all_column0_categories))
        ]
    except:
        pass
    # Column1 sorted categories
    all_column1_categories = list(set([str(item[1]) for item in query_result]))
    all_column1_categories.sort()
    try:
        try:
            order = []
            for item in all_column1_categories:
                order += [float(item.split(";")[0].split("[")[1])]
        except:
            order = [float(item) for item in all_column1_categories]
        all_column1_categories = [
            x for _, x in sorted(zip(order, all_column1_categories))
        ]
    except:
        pass
    all_columns = [
        ["" for item in all_column0_categories] for item in all_column1_categories
    ]
    for item in query_result:
        j, i = (
            all_column0_categories.index(str(item[0])),
            all_column1_categories.index(str(item[1])),
        )
        all_columns[i][j] = item[2]
    all_columns = [
        [all_column1_categories[i]] + all_columns[i] for i in range(0, len(all_columns))
    ]
    all_columns = [
        [columns[0] + "/" + columns[1]] + all_column0_categories
    ] + all_columns
    if show:
        all_count = [item[2] for item in query_result]
        ax = cmatrix(
            all_columns,
            all_column0_categories,
            all_column1_categories,
            len(all_column0_categories),
            len(all_column1_categories),
            vmax=max(all_count),
            vmin=min(all_count),
            title="",
            colorbar=aggregate,
            x_label=columns[1],
            y_label=columns[0],
            with_numbers=with_numbers,
            inverse=True,
            extent=extent,
            ax=ax,
            is_pivot=True,
            **style_kwds,
        )
        if return_ax:
            return ax
    values = {all_columns[0][0]: all_columns[0][1 : len(all_columns[0])]}
    del all_columns[0]
    for column in all_columns:
        values[column[0]] = column[1 : len(column)]
    return tablesample(values=values,)


# ---#
def scatter_matrix(
    vdf, columns: list = [], **style_kwds,
):
    for column in columns:
        if (column not in vdf.get_columns()) and (
            str_column(column) not in vdf.get_columns()
        ):
            raise MissingColumn("The Virtual Column {} doesn't exist".format(column))
    if not (columns):
        columns = vdf.numcol()
    elif len(columns) == 1:
        return vdf[columns[0]].hist()
    n = len(columns)
    fig, axes = (
        plt.subplots(
            nrows=n, ncols=n, figsize=(min(1.5 * (n + 1), 500), min(1.5 * (n + 1), 500))
        )
        if isnotebook()
        else plt.subplots(
            nrows=n,
            ncols=n,
            figsize=(min(int((n + 1) / 1.1), 500), min(int((n + 1) / 1.1), 500)),
        )
    )
    random_func = random_function()
    query = "SELECT {}, {} AS rand FROM {} WHERE __verticapy_split__ < 0.5 ORDER BY rand LIMIT 1000".format(
        ", ".join(columns), random_func, vdf.__genSQL__(True)
    )
    all_scatter_points = vdf.__executeSQL__(
        query=query, title="Select random points to draw the scatter plot"
    ).fetchall()
    all_scatter_columns = []
    for i in range(n):
        all_scatter_columns += [[item[i] for item in all_scatter_points]]
    for i in range(n):
        x = columns[i]
        axes[-1][i].set_xlabel(x, rotation=90)
        axes[i][0].set_ylabel(x, rotation=0)
        axes[i][0].yaxis.get_label().set_ha("right")
        for j in range(n):
            axes[i][j].get_xaxis().set_ticks([])
            axes[i][j].get_yaxis().set_ticks([])
            y = columns[j]
            if x == y:
                x0, y0, z0, h0, is_categorical = compute_plot_variables(
                    vdf[x], method="density", max_cardinality=1
                )
                param = {"color": color_dict(style_kwds, 0), "edgecolor": "black"}
                if "edgecolor" in style_kwds:
                    param["edgecolor"] = style_kwds["edgecolor"]
                axes[i, j].bar(x0, y0, h0 / 0.94, **param)
            else:
                param = {
                    "color": color_dict(style_kwds, 1),
                    "edgecolor": "black",
                    "alpha": 0.9,
                    "s": 40,
                    "marker": "o",
                }
                axes[i, j].scatter(
                    all_scatter_columns[j],
                    all_scatter_columns[i],
                    **updated_dict(param, style_kwds, 1),
                )
    return axes


# ---#
def scatter2D(
    vdf,
    columns: list,
    max_cardinality: int = 6,
    cat_priority: list = [],
    with_others: bool = True,
    max_nb_points: int = 100000,
    bbox: list = [],
    img: str = "",
    ax=None,
    **style_kwds,
):
    colors = gen_colors()
    markers = ["^", "o", "+", "*", "h", "x", "D", "1"] * 10
    columns = [str_column(column) for column in columns]
    if (bbox) and len(bbox) != 4:
        warning_message = "Parameter 'bbox' must be a list of 4 numerics containing the 'xlim' and 'ylim'.\nIt was ignored."
        warnings.warn(warning_message, Warning)
        bbox = []
    for column in columns:
        if column not in vdf.get_columns():
            raise MissingColumn("The Virtual Column {} doesn't exist".format(column))
    if not (vdf[columns[0]].isnum()) or not (vdf[columns[1]].isnum()):
        raise TypeError(
            "The two first columns of the parameter 'columns' must be numerical"
        )
    if len(columns) == 2:
        tablesample = max_nb_points / vdf.shape()[0]
        query = "SELECT {}, {} FROM {} WHERE __verticapy_split__ < {} AND {} IS NOT NULL AND {} IS NOT NULL LIMIT {}".format(
            columns[0],
            columns[1],
            vdf.__genSQL__(True),
            tablesample,
            columns[0],
            columns[1],
            max_nb_points,
        )
        query_result = vdf.__executeSQL__(
            query=query, title="Select random points to draw the scatter plot"
        ).fetchall()
        column1, column2 = (
            [item[0] for item in query_result],
            [item[1] for item in query_result],
        )
        if not (ax):
            fig, ax = plt.subplots()
            if isnotebook():
                fig.set_size_inches(8, 6)
            ax.grid()
            ax.set_axisbelow(True)
        if bbox:
            ax.set_xlim(bbox[0], bbox[1])
            ax.set_ylim(bbox[2], bbox[3])
        if img:
            im = plt.imread(img)
            if not (bbox):
                bbox = (min(column1), max(column1), min(column2), max(column2))
                ax.set_xlim(bbox[0], bbox[1])
                ax.set_ylim(bbox[2], bbox[3])
            ax.imshow(im, extent=bbox)
        ax.set_ylabel(columns[1])
        ax.set_xlabel(columns[0])
        param = {
            "marker": "o",
            "color": colors[0],
            "s": 50,
            "edgecolors": "black",
        }
        ax.scatter(
            column1, column2, **updated_dict(param, style_kwds),
        )
        return ax
    else:
        column_groupby = columns[2]
        count = vdf.shape()[0]
        if cat_priority:
            query_result = cat_priority
        else:
            query = "SELECT {} FROM {} WHERE {} IS NOT NULL GROUP BY {} ORDER BY COUNT(*) DESC LIMIT {}".format(
                column_groupby,
                vdf.__genSQL__(),
                column_groupby,
                column_groupby,
                max_cardinality,
            )
            query_result = vdf.__executeSQL__(
                query=query, title="Compute {} categories".format(column_groupby)
            ).fetchall()
            query_result = [item for sublist in query_result for item in sublist]
        all_columns, all_scatter, all_categories = [query_result], [], query_result
        if not (ax):
            fig, ax = plt.subplots()
            if isnotebook():
                fig.set_size_inches(8, 6)
            ax.grid()
            ax.set_axisbelow(True)
        if bbox:
            ax.set_xlim(bbox[0], bbox[1])
            ax.set_ylim(bbox[2], bbox[3])
        if img:
            im = plt.imread(img)
            if not (bbox):
                aggr = vdf.agg(columns=[columns[0], columns[1]], func=["min", "max"])
                bbox = (
                    aggr.values["min"][0],
                    aggr.values["max"][0],
                    aggr.values["min"][1],
                    aggr.values["max"][1],
                )
                ax.set_xlim(bbox[0], bbox[1])
                ax.set_ylim(bbox[2], bbox[3])
            ax.imshow(im, extent=bbox)
        others = []
        groupby_cardinality = vdf[column_groupby].nunique(True)
        count = vdf.shape()[0]
        tablesample = 0.1 if (count > 10000) else 0.9
        for idx, category in enumerate(all_categories):
            if (max_cardinality < groupby_cardinality) or (
                len(cat_priority) < groupby_cardinality
            ):
                others += [
                    "{} != '{}'".format(
                        column_groupby, str(category).replace("'", "''")
                    )
                ]
            query = "SELECT {}, {} FROM {} WHERE  __verticapy_split__ < {} AND {} = '{}' AND {} IS NOT NULL AND {} IS NOT NULL LIMIT {}"
            query = query.format(
                columns[0],
                columns[1],
                vdf.__genSQL__(True),
                tablesample,
                columns[2],
                str(category).replace("'", "''"),
                columns[0],
                columns[1],
                int(max_nb_points / len(all_categories)),
            )
            vdf.__executeSQL__(
                query=query,
                title="Select random points to draw the scatter plot (category = '{}')".format(
                    str(category)
                ),
            )
            query_result = vdf._VERTICAPY_VARIABLES_["cursor"].fetchall()
            column1, column2 = (
                [float(item[0]) for item in query_result],
                [float(item[1]) for item in query_result],
            )
            all_columns += [[column1, column2]]
            param = {
                "alpha": 0.8,
                "marker": "o",
                "color": colors[idx % len(colors)],
                "s": 50,
                "edgecolors": "black",
            }
            all_scatter += [
                ax.scatter(column1, column2, **updated_dict(param, style_kwds, idx))
            ]
        if with_others and idx + 1 < groupby_cardinality:
            all_categories += ["others"]
            query = "SELECT {}, {} FROM {} WHERE {} AND {} IS NOT NULL AND {} IS NOT NULL AND __verticapy_split__ < {} LIMIT {}"
            query = query.format(
                columns[0],
                columns[1],
                vdf.__genSQL__(True),
                " AND ".join(others),
                columns[0],
                columns[1],
                tablesample,
                int(max_nb_points / len(all_categories)),
            )
            query_result = vdf.__executeSQL__(
                query=query,
                title="Select random points to draw the scatter plot (category = 'others')",
            ).fetchall()
            column1, column2 = (
                [float(item[0]) for item in query_result],
                [float(item[1]) for item in query_result],
            )
            all_columns += [[column1, column2]]
            param = {
                "alpha": 0.8,
                "marker": "o",
                "color": colors[idx + 1 % len(colors)],
                "s": 50,
                "edgecolors": "black",
            }
            all_scatter += [
                ax.scatter(column1, column2, **updated_dict(param, style_kwds, idx + 1))
            ]
        for idx, item in enumerate(all_categories):
            if len(str(item)) > 20:
                all_categories[idx] = str(item)[0:20] + "..."
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        ax.legend(
            all_scatter,
            all_categories,
            title=column_groupby,
            loc="center left",
            bbox_to_anchor=[1, 0.5],
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax


# ---#
def scatter3D(
    vdf,
    columns: list,
    max_cardinality: int = 3,
    cat_priority: list = [],
    with_others: bool = True,
    max_nb_points: int = 1000,
    ax=None,
    **style_kwds,
):
    columns = [str_column(column) for column in columns]
    colors = gen_colors()
    markers = ["^", "o", "+", "*", "h", "x", "D", "1"] * 10
    if (len(columns) < 3) or (len(columns) > 4):
        raise ParameterError(
            "3D Scatter plot can only be done with at least two columns and maximum with four columns"
        )
    else:
        for column in columns:
            if column not in vdf.get_columns():
                raise MissingColumn(
                    "The Virtual Column {} doesn't exist".format(column)
                )
        for i in range(3):
            if not (vdf[columns[i]].isnum()):
                raise TypeError(
                    "The three first columns of the parameter 'columns' must be numerical"
                )
        if len(columns) == 3:
            tablesample = max_nb_points / vdf.shape()[0]
            query = "SELECT {}, {}, {} FROM {} WHERE __verticapy_split__ < {} AND {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL LIMIT {}".format(
                columns[0],
                columns[1],
                columns[2],
                vdf.__genSQL__(True),
                tablesample,
                columns[0],
                columns[1],
                columns[2],
                max_nb_points,
            )
            query_result = vdf.__executeSQL__(
                query=query, title="Select random points to draw the scatter plot"
            ).fetchall()
            column1, column2, column3 = (
                [float(item[0]) for item in query_result],
                [float(item[1]) for item in query_result],
                [float(item[2]) for item in query_result],
            )
            if not (ax):
                if isnotebook():
                    plt.figure(figsize=(8, 6))
                ax = plt.axes(projection="3d")
            param = {
                "color": colors[0],
                "s": 50,
                "edgecolors": "black",
            }
            ax.scatter(
                column1, column2, column3, **updated_dict(param, style_kwds),
            )
            ax.set_xlabel(columns[0])
            ax.set_ylabel(columns[1])
            ax.set_zlabel(columns[2])
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            return ax
        else:
            column_groupby = columns[3]
            count = vdf.shape()[0]
            if cat_priority:
                query_result = cat_priority
            else:
                query = "SELECT {} FROM {} WHERE {} IS NOT NULL GROUP BY {} ORDER BY COUNT(*) DESC LIMIT {}".format(
                    column_groupby,
                    vdf.__genSQL__(),
                    column_groupby,
                    column_groupby,
                    max_cardinality,
                )
                query_result = vdf.__executeSQL__(
                    query=query,
                    title="Compute the vcolumn {} distinct categories".format(
                        column_groupby
                    ),
                ).fetchall()
                query_result = [item for sublist in query_result for item in sublist]
            all_columns, all_scatter, all_categories = [query_result], [], query_result
            if not (ax):
                if isnotebook():
                    plt.figure(figsize=(8, 6))
                ax = plt.axes(projection="3d")
            others = []
            groupby_cardinality = vdf[column_groupby].nunique(True)
            tablesample = 10 if (count > 10000) else 90
            for idx, category in enumerate(all_categories):
                if (max_cardinality < groupby_cardinality) or (
                    len(cat_priority) < groupby_cardinality
                ):
                    others += [
                        "{} != '{}'".format(
                            column_groupby, str(category).replace("'", "''")
                        )
                    ]
                query = "SELECT {}, {}, {} FROM {} WHERE __verticapy_split__ < {} AND {} = '{}' AND {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL limit {}"
                query = query.format(
                    columns[0],
                    columns[1],
                    columns[2],
                    vdf.__genSQL__(True),
                    tablesample,
                    columns[3],
                    str(category).replace("'", "''"),
                    columns[0],
                    columns[1],
                    columns[2],
                    int(max_nb_points / len(all_categories)),
                )
                query_result = vdf.__executeSQL__(
                    query=query,
                    title="Select random points to draw the scatter plot (category = '{}')".format(
                        category
                    ),
                ).fetchall()
                column1, column2, column3 = (
                    [float(item[0]) for item in query_result],
                    [float(item[1]) for item in query_result],
                    [float(item[2]) for item in query_result],
                )
                all_columns += [[column1, column2, column3]]
                param = {
                    "alpha": 0.8,
                    "marker": "o",
                    "color": colors[idx % len(colors)],
                    "s": 50,
                    "edgecolors": "black",
                }
                all_scatter += [
                    ax.scatter(
                        column1,
                        column2,
                        column3,
                        **updated_dict(param, style_kwds, idx),
                    )
                ]
            if with_others and idx + 1 < groupby_cardinality:
                all_categories += ["others"]
                query = "SELECT {}, {}, {} FROM {} WHERE {} AND {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL AND __verticapy_split__ < {} LIMIT {}"
                query = query.format(
                    columns[0],
                    columns[1],
                    columns[2],
                    vdf.__genSQL__(True),
                    " AND ".join(others),
                    columns[0],
                    columns[1],
                    columns[2],
                    tablesample,
                    int(max_nb_points / len(all_categories)),
                )
                query_result = vdf.__executeSQL__(
                    query=query,
                    title="Select random points to draw the scatter plot (category = 'others')",
                ).fetchall()
                column1, column2 = (
                    [float(item[0]) for item in query_result],
                    [float(item[1]) for item in query_result],
                )
                all_columns += [[column1, column2]]
                param = {
                    "alpha": 0.8,
                    "marker": "o",
                    "color": colors[idx + 1 % len(colors)],
                    "s": 50,
                    "edgecolors": "black",
                }
                all_scatter += [
                    ax.scatter(
                        column1, column2, **updated_dict(param, style_kwds, idx + 1),
                    )
                ]
            for idx, item in enumerate(all_categories):
                if len(str(item)) > 20:
                    all_categories[idx] = str(item)[0:20] + "..."
            ax.set_xlabel(columns[0])
            ax.set_ylabel(columns[1])
            ax.set_zlabel(columns[2])
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.legend(
                all_scatter,
                all_categories,
                scatterpoints=1,
                title=column_groupby,
                loc="center left",
                bbox_to_anchor=[1.1, 0.5],
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            return ax


# ---#
def spider(
    vdf,
    columns: list,
    method: str = "density",
    of: str = "",
    max_cardinality: tuple = (6, 6),
    h: tuple = (None, None),
    ax=None,
    **style_kwds,
):
    unique = vdf[columns[0]].nunique()
    if unique < 3:
        raise ParameterError(
            f"The first column of the Spider Plot must have at least 3 categories. Found {int(unique)}."
        )
    colors = gen_colors()
    all_columns = vdf.pivot_table(
        columns, method=method, of=of, h=h, max_cardinality=max_cardinality, show=False
    ).values
    all_cat = [category for category in all_columns]
    n = len(all_columns)
    m = len(all_columns[all_cat[0]])
    angles = [i / float(m) * 2 * math.pi for i in range(m)]
    angles += angles[:1]
    categories = all_columns[all_cat[0]]
    fig = plt.figure()
    if not (ax):
        ax = fig.add_subplot(111, polar=True)
    all_vals = []
    for idx, category in enumerate(all_columns):
        if idx != 0:
            values = all_columns[category]
            values += values[:1]
            for i, elem in enumerate(values):
                if isinstance(elem, str) or elem == None:
                    values[i] = 0
                else:
                    values[i] = float(elem)
            all_vals += values
            plt.xticks(angles[:-1], categories, color="grey", size=8)
            ax.set_rlabel_position(0)
            param = {"linewidth": 1, "linestyle": "solid", "color": colors[idx - 1]}
            ax.plot(
                angles,
                values,
                label=category,
                **updated_dict(param, style_kwds, idx - 1),
            )
            color = updated_dict(param, style_kwds, idx - 1)["color"]
            ax.fill(angles, values, alpha=0.1, color=color)
    ax.set_yticks([min(all_vals), (max(all_vals) + min(all_vals)) / 2, max(all_vals)])
    ax.set_rgrids(
        [min(all_vals), (max(all_vals) + min(all_vals)) / 2, max(all_vals)],
        angle=180.0,
        fmt="%0.1f",
    )
    ax.set_xlabel(columns[0])
    if method.lower() == "mean":
        method = "avg"
    if method.lower() == "density":
        ax.set_ylabel("Density")
    elif (method.lower() in ["avg", "min", "max", "sum"]) and (of != None):
        ax.set_ylabel("{}({})".format(method, of))
    elif method.lower() == "count":
        ax.set_ylabel("Frequency")
    else:
        ax.set_ylabel(method)
    if len(columns) > 1:
        ax.legend(
            title=columns[1], loc="center left", bbox_to_anchor=[1.1, 0.5],
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    return ax


# ---#
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
    if not (by):
        query = "SELECT {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL".format(
            order_by, vdf.alias, vdf.parent.__genSQL__(), order_by, vdf.alias
        )
        query += (
            " AND {} > '{}'".format(order_by, order_by_start)
            if (order_by_start)
            else ""
        )
        query += (
            " AND {} < '{}'".format(order_by, order_by_end) if (order_by_end) else ""
        )
        query += " ORDER BY {}, {}".format(order_by, vdf.alias)
        query_result = vdf.__executeSQL__(
            query=query, title="Select points to draw the curve"
        ).fetchall()
        order_by_values = [item[0] for item in query_result]
        try:
            if isinstance(order_by_values[0], str):
                from dateutil.parser import parse

                order_by_values = [parse(elem) for elem in order_by_values]
        except:
            pass
        column_values = [float(item[1]) for item in query_result]
        if not (ax):
            fig, ax = plt.subplots()
            if isnotebook():
                fig.set_size_inches(8, 6)
            ax.grid(axis="y")
        if len(order_by_values) < 20:
            param = {
                "marker": "o",
                "markevery": 0.05,
                "markerfacecolor": "white",
                "markersize": 7,
                "markeredgecolor": "black",
                "color": gen_colors()[0],
                "linewidth": 2,
            }
        else:
            param = {
                "color": gen_colors()[0],
                "linewidth": 2,
            }
        if step:
            ax.step(
                order_by_values, column_values, **updated_dict(param, style_kwds),
            )
        else:
            ax.plot(
                order_by_values, column_values, **updated_dict(param, style_kwds),
            )
        if area and not (step):
            if "color" in updated_dict(param, style_kwds):
                color = updated_dict(param, style_kwds)["color"]
            ax.fill_between(order_by_values, column_values, facecolor=color, alpha=0.2)
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        ax.set_xlabel(order_by)
        ax.set_ylabel(vdf.alias)
        ax.set_xlim(min(order_by_values), max(order_by_values))
        return ax
    else:
        colors = gen_colors()
        by = str_column(by)
        cat = vdf.parent[by].distinct()
        all_data = []
        for column in cat:
            query = "SELECT {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL".format(
                order_by, vdf.alias, vdf.parent.__genSQL__(), order_by, vdf.alias
            )
            query += (
                " AND {} > '{}'".format(order_by, order_by_start)
                if (order_by_start)
                else ""
            )
            query += (
                " AND {} < '{}'".format(order_by, order_by_end)
                if (order_by_end)
                else ""
            )
            query += " AND {} = '{}'".format(by, str(column).replace("'", "''"))
            query += " ORDER BY {}, {}".format(order_by, vdf.alias)
            query_result = vdf.__executeSQL__(
                query=query, title="Select points to draw the curve"
            ).fetchall()
            all_data += [
                [
                    [item[0] for item in query_result],
                    [float(item[1]) for item in query_result],
                    column,
                ]
            ]
            try:
                if isinstance(all_data[-1][0][0], str):
                    from dateutil.parser import parse

                    all_data[-1][0] = [parse(elem) for elem in all_data[-1][0]]
            except:
                pass
        if not (ax):
            fig, ax = plt.subplots()
            if isnotebook():
                fig.set_size_inches(8, 6)
            ax.grid(axis="y")
        for idx, elem in enumerate(all_data):
            if len(elem[0]) < 20:
                param = {
                    "marker": "o",
                    "markevery": 0.05,
                    "markerfacecolor": colors[idx % len(colors)],
                    "markersize": 7,
                    "markeredgecolor": "black",
                    "color": colors[idx % len(colors)],
                }
            else:
                param = {"color": colors[idx % len(colors)]}
            param["markerfacecolor"] = color_dict(style_kwds, idx)
            if step:
                ax.step(
                    elem[0],
                    elem[1],
                    label=elem[2],
                    **updated_dict(param, style_kwds, idx),
                )
            else:
                ax.plot(
                    elem[0],
                    elem[1],
                    label=elem[2],
                    **updated_dict(param, style_kwds, idx),
                )
        ax.set_xlabel(order_by)
        ax.set_ylabel(vdf.alias)
        ax.legend(title=by, loc="center left", bbox_to_anchor=[1, 0.5])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        return ax
