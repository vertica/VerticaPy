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
import math, collections

# Other Python Modules
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import numpy as np

# VerticaPy Modules
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.errors import *
from verticapy.plot import gen_colors

# ---#
def logit_plot(
    X: list,
    y: str,
    input_relation: str,
    coefficients: list,
    cursor=None,
    max_nb_points=50,
    ax=None,
    **style_kwds,
):
    check_types(
        [
            ("X", X, [list],),
            ("y", y, [str],),
            ("input_relation", input_relation, [str],),
            ("coefficients", coefficients, [list],),
            ("max_nb_points", max_nb_points, [int, float],),
        ]
    )
    cursor, conn = check_cursor(cursor)[0:2]
    param0 = {
        "marker": "o",
        "s": 50,
        "color": gen_colors()[0],
        "edgecolors": "black",
        "alpha": 0.8,
    }
    param1 = {
        "marker": "o",
        "s": 50,
        "color": gen_colors()[1],
        "edgecolors": "black",
    }

    def logit(x):
        return 1 / (1 + math.exp(-x))

    if len(X) == 1:
        query = "(SELECT {}, {} FROM {} WHERE {} IS NOT NULL AND {} = 0 LIMIT {})".format(
            X[0], y, input_relation, X[0], y, int(max_nb_points / 2)
        )
        query += " UNION ALL (SELECT {}, {} FROM {} WHERE {} IS NOT NULL AND {} = 1 LIMIT {})".format(
            X[0], y, input_relation, X[0], y, int(max_nb_points / 2)
        )
        cursor.execute(query)
        all_points = cursor.fetchall()
        if not (ax):
            fig, ax = plt.subplots()
            if isnotebook():
                fig.set_size_inches(8, 6)
            ax.set_axisbelow(True)
            ax.grid()
        x0, x1 = [], []
        for idx, item in enumerate(all_points):
            if item[1] == 0:
                x0 += [float(item[0])]
            else:
                x1 += [float(item[0])]
        min_logit, max_logit = min(x0 + x1), max(x0 + x1)
        step = (max_logit - min_logit) / 40.0
        x_logit = (
            arange(min_logit - 5 * step, max_logit + 5 * step, step)
            if (step > 0)
            else [max_logit]
        )
        y_logit = [logit(coefficients[0] + coefficients[1] * item) for item in x_logit]
        ax.plot(x_logit, y_logit, alpha=1, color="black")
        all_scatter = [
            ax.scatter(
                x0,
                [logit(coefficients[0] + coefficients[1] * item) for item in x0],
                **updated_dict(param1, style_kwds, 1),
            )
        ]
        all_scatter += [
            ax.scatter(
                x1,
                [logit(coefficients[0] + coefficients[1] * item) for item in x1],
                **updated_dict(param0, style_kwds, 0),
            )
        ]
        ax.set_xlabel(X[0])
        ax.set_ylabel(y)
        ax.legend(
            all_scatter,
            [0, 1],
            scatterpoints=1,
            loc="center left",
            bbox_to_anchor=[1, 0.5],
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    elif len(X) == 2:
        query = "(SELECT {}, {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL AND {} = 0 LIMIT {})".format(
            X[0], X[1], y, input_relation, X[0], X[1], y, int(max_nb_points / 2)
        )
        query += " UNION (SELECT {}, {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL AND {} = 1 LIMIT {})".format(
            X[0], X[1], y, input_relation, X[0], X[1], y, int(max_nb_points / 2)
        )
        cursor.execute(query)
        all_points = cursor.fetchall()
        x0, x1, y0, y1 = [], [], [], []
        for idx, item in enumerate(all_points):
            if item[2] == 0:
                x0 += [float(item[0])]
                y0 += [float(item[1])]
            else:
                x1 += [float(item[0])]
                y1 += [float(item[1])]
        min_logit_x, max_logit_x = min(x0 + x1), max(x0 + x1)
        step_x = (max_logit_x - min_logit_x) / 40.0
        min_logit_y, max_logit_y = min(y0 + y1), max(y0 + y1)
        step_y = (max_logit_y - min_logit_y) / 40.0
        X_logit = (
            arange(min_logit_x - 5 * step_x, max_logit_x + 5 * step_x, step_x)
            if (step_x > 0)
            else [max_logit_x]
        )
        Y_logit = (
            arange(min_logit_y - 5 * step_y, max_logit_y + 5 * step_y, step_y)
            if (step_y > 0)
            else [max_logit_y]
        )
        X_logit, Y_logit = np.meshgrid(X_logit, Y_logit)
        Z_logit = 1 / (
            1
            + np.exp(
                -(
                    coefficients[0]
                    + coefficients[1] * X_logit
                    + coefficients[2] * Y_logit
                )
            )
        )
        if not (ax):
            if isnotebook():
                plt.figure(figsize=(8, 6))
            ax = plt.axes(projection="3d")
        ax.plot_surface(
            X_logit, Y_logit, Z_logit, rstride=1, cstride=1, alpha=0.5, color="gray"
        )
        all_scatter = [
            ax.scatter(
                x0,
                y0,
                [
                    logit(
                        coefficients[0]
                        + coefficients[1] * x0[i]
                        + coefficients[2] * y0[i]
                    )
                    for i in range(len(x0))
                ],
                **updated_dict(param1, style_kwds, 1),
            )
        ]
        all_scatter += [
            ax.scatter(
                x1,
                y1,
                [
                    logit(
                        coefficients[0]
                        + coefficients[1] * x1[i]
                        + coefficients[2] * y1[i]
                    )
                    for i in range(len(x1))
                ],
                **updated_dict(param0, style_kwds, 0),
            )
        ]
        ax.set_xlabel(X[0])
        ax.set_ylabel(X[1])
        ax.set_zlabel(y)
        ax.legend(
            all_scatter,
            [0, 1],
            scatterpoints=1,
            loc="center left",
            bbox_to_anchor=[1.1, 0.5],
            title=y,
            ncol=2,
            fontsize=8,
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    else:
        raise ParameterError("The number of predictors is too big.")
    if conn:
        conn.close()
    return ax


# ---#
def lof_plot(
    input_relation: str,
    columns: list,
    lof: str,
    cursor=None,
    tablesample: float = -1,
    ax=None,
    **style_kwds,
):
    check_types(
        [
            ("input_relation", input_relation, [str],),
            ("columns", columns, [list],),
            ("lof", lof, [str],),
            ("tablesample", tablesample, [int, float],),
        ]
    )
    cursor, conn = check_cursor(cursor)[0:2]
    tablesample = (
        "TABLESAMPLE({})".format(tablesample)
        if (tablesample > 0 and tablesample < 100)
        else ""
    )
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
    param = {
        "s": 50,
        "edgecolors": "black",
        "color": colors[0],
    }
    if len(columns) == 1:
        column = str_column(columns[0])
        query = "SELECT {}, {} FROM {} {} WHERE {} IS NOT NULL".format(
            column, lof, input_relation, tablesample, column
        )
        cursor.execute(query)
        query_result = cursor.fetchall()
        column1, lof = (
            [item[0] for item in query_result],
            [item[1] for item in query_result],
        )
        column2 = [0] * len(column1)
        if not (ax):
            fig, ax = plt.subplots()
            if isnotebook():
                fig.set_size_inches(8, 2)
            ax.set_axisbelow(True)
            ax.grid()
        ax.set_xlabel(column)
        radius = [2 * 1000 * (item - min(lof)) / (max(lof) - min(lof)) for item in lof]
        ax.scatter(
            column1, column2, label="Data points", **updated_dict(param, style_kwds, 0),
        )
        ax.scatter(
            column1,
            column2,
            s=radius,
            label="Outlier scores",
            facecolors="none",
            color=colors[1],
        )
    elif len(columns) == 2:
        columns = [str_column(column) for column in columns]
        query = "SELECT {}, {}, {} FROM {} {} WHERE {} IS NOT NULL AND {} IS NOT NULL".format(
            columns[0],
            columns[1],
            lof,
            input_relation,
            tablesample,
            columns[0],
            columns[1],
        )
        cursor.execute(query)
        query_result = cursor.fetchall()
        column1, column2, lof = (
            [item[0] for item in query_result],
            [item[1] for item in query_result],
            [item[2] for item in query_result],
        )
        if not (ax):
            fig, ax = plt.subplots()
            if isnotebook():
                fig.set_size_inches(8, 6)
            ax.set_axisbelow(True)
            ax.grid()
        ax.set_ylabel(columns[1])
        ax.set_xlabel(columns[0])
        radius = [1000 * (item - min(lof)) / (max(lof) - min(lof)) for item in lof]
        ax.scatter(
            column1, column2, label="Data points", **updated_dict(param, style_kwds, 0),
        )
        ax.scatter(
            column1,
            column2,
            s=radius,
            label="Outlier scores",
            facecolors="none",
            color=colors[1],
        )
    elif len(columns) == 3:
        query = "SELECT {}, {}, {}, {} FROM {} {} WHERE {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL".format(
            columns[0],
            columns[1],
            columns[2],
            lof,
            input_relation,
            tablesample,
            columns[0],
            columns[1],
            columns[2],
        )
        cursor.execute(query)
        query_result = cursor.fetchall()
        column1, column2, column3, lof = (
            [float(item[0]) for item in query_result],
            [float(item[1]) for item in query_result],
            [float(item[2]) for item in query_result],
            [float(item[3]) for item in query_result],
        )
        if not (ax):
            if isnotebook():
                plt.figure(figsize=(8, 6))
            ax = plt.axes(projection="3d")
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        ax.set_zlabel(columns[2])
        radius = [1000 * (item - min(lof)) / (max(lof) - min(lof)) for item in lof]
        ax.scatter(
            column1,
            column2,
            column3,
            label="Data points",
            **updated_dict(param, style_kwds, 0),
        )
        ax.scatter(
            column1, column2, column3, s=radius, facecolors="none", color=colors[1],
        )
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    else:
        raise Exception(
            "LocalOutlierFactor Plot is available for a maximum of 3 columns"
        )
    if conn:
        conn.close()
    return ax


# ---#
def plot_importance(
    coeff_importances: dict,
    coeff_sign: dict = {},
    print_legend: bool = True,
    ax=None,
    **style_kwds,
):
    check_types(
        [
            ("coeff_importances", coeff_importances, [dict],),
            ("coeff_sign", coeff_sign, [dict],),
            ("print_legend", print_legend, [bool],),
        ]
    )
    coefficients, importances, signs = [], [], []
    for coeff in coeff_importances:
        coefficients += [coeff]
        importances += [coeff_importances[coeff]]
        signs += [coeff_sign[coeff]] if (coeff in coeff_sign) else [1]
    importances, coefficients, signs = zip(
        *sorted(zip(importances, coefficients, signs))
    )
    if not (ax):
        fig, ax = plt.subplots()
        if isnotebook():
            fig.set_size_inches(12, int(len(importances) / 2) + 1)
        ax.set_axisbelow(True)
        ax.grid()
    color = []
    for item in signs:
        color += (
            [color_dict(style_kwds, 0)] if (item == 1) else [color_dict(style_kwds, 1)]
        )
    param = {"alpha": 0.86}
    style_kwds = updated_dict(param, style_kwds)
    style_kwds["color"] = color
    ax.barh(range(0, len(importances)), importances, 0.9, **style_kwds)
    if print_legend:
        orange = mpatches.Patch(color=color_dict(style_kwds, 1), label="sign -")
        blue = mpatches.Patch(color=color_dict(style_kwds, 0), label="sign +")
        ax.legend(handles=[orange, blue], loc="center left", bbox_to_anchor=[1, 0.5])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_ylabel("Features")
    ax.set_xlabel("Importance")
    ax.set_yticks(range(0, len(importances)))
    ax.set_yticklabels(coefficients)
    return ax

# ---#
def plot_stepwise_ml(x: list, y: list, z: list = [], w: list = [], var: list = [], x_label: str = "n_features", y_label: str = "score", direction = "forward", ax=None, **style_kwds):
    colors = gen_colors()
    if not(ax):
        fig, ax = plt.subplots()
        if isnotebook():
            fig.set_size_inches(8, 6)
        ax.grid(axis = "y")
        ax.set_axisbelow(True)
    sign = "+" if direction == "forward" else "-"
    x_new, y_new, z_new = [], [], []
    for idx in range(len(x)):
        if idx == 0 or w[idx][0] == sign:
            x_new += [x[idx]]
            y_new += [y[idx]]
            z_new += [z[idx]]
    if len(var[0]) > 3:
        var0 = var[0][0:2] + ["..."] + var[0][-1:]
    else:
        var0 = var[0]
    if len(var[1]) > 3:
        var1 = var[1][0:2] + ["..."] + var[1][-1:]
    else:
        var1 = var[1]
    if "color" in style_kwds:
        if isinstance(style_kwds["color"], str):
            c0, c1 = style_kwds["color"], colors[1]
        else:
            c0, c1 = style_kwds["color"][0], style_kwds["color"][1]
    else:
        c0, c1 = colors[0], colors[1]
    if "color" in style_kwds:
        del style_kwds["color"]
    if direction == "forward":
        delta_ini, delta_final = 0.1, -0.15
        rot_ini, rot_final = -90, 90
        verticalalignment_init, verticalalignment_final = "top", "bottom"
        horizontalalignment = "center"
    else:
        delta_ini, delta_final = 0.35, -0.3
        rot_ini, rot_final = 90, -90
        verticalalignment_init, verticalalignment_final = "top", "bottom"
        horizontalalignment = "left"
    param = {"marker": "s", "alpha": 0.5, "edgecolors": "black", "s": 400}
    ax.scatter(x_new[1:-1], y_new[1:-1], c=c0, **updated_dict(param, style_kwds,),)
    ax.scatter([x_new[0], x_new[-1]], [y_new[0], y_new[-1]], c=c1, **updated_dict(param, style_kwds,),)
    ax.text(x_new[0] + delta_ini, y_new[0], "Initial Variables: {}".format("["+", ".join(var0)+"]"), rotation = rot_ini, verticalalignment=verticalalignment_init,)
    for idx in range(1, len(x_new)):
        dx, dy = x_new[idx] - x_new[idx - 1], y_new[idx] - y_new[idx - 1]
        ax.arrow(x_new[idx - 1], y_new[idx - 1], dx, dy, fc='k', ec='k', alpha=0.2)
        ax.text((x_new[idx] + x_new[idx - 1]) / 2, (y_new[idx] + y_new[idx - 1]) / 2, sign + " " + z_new[idx], rotation = rot_ini)
    if direction == "backward":
        ax.set_xlim(max(x) + 0.1 * (1 + max(x) - min(x)), min(x) - 0.1 - 0.1 * (1 + max(x) - min(x)))
    ax.text(x_new[-1] + delta_final, y_new[-1], "Final Variables: {}".format("["+", ".join(var1)+"]"), rotation = rot_final, verticalalignment=verticalalignment_final, horizontalalignment=horizontalalignment,)
    ax.set_xticks(x_new)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return ax

# ---#
def plot_bubble_ml(x: list, y: list, s: list = None, z: list = [], x_label: str = "time", y_label: str = "score", title: str = "Model Type", reverse: tuple = (True, True,), plt_text=True, ax=None, **style_kwds):
    if s:
        s = [min(250 + 5000 * elem, 1200) if elem != 0 else 1000 for elem in s]
    if z and s:
        data = [(x[i], y[i], s[i], z[i]) for i in range(len(x))]
        data.sort(key=lambda tup: str(tup[3]),)
        x = [elem[0] for elem in data]
        y = [elem[1] for elem in data]
        s = [elem[2] for elem in data]
        z = [elem[3] for elem in data]
    elif z:
        data = [(x[i], y[i], z[i]) for i in range(len(x))]
        data.sort(key=lambda tup: str(tup[2]),)
        x = [elem[0] for elem in data]
        y = [elem[1] for elem in data]
        z = [elem[2] for elem in data]
    colors = gen_colors()
    if not (ax):
        fig, ax = plt.subplots()
        if isnotebook():
            fig.set_size_inches(8, 6)
        ax.grid(axis = "y")
        ax.set_axisbelow(True)
    if z:
        current_cat = z[0]
        idx = 0
        i = 0
        j = 1
        all_scatter = []
        all_categories = [current_cat]
        tmp_colors = []
        while j != len(z):
            while j < len(z) and z[j] == current_cat:
                j += 1
            param = {"alpha": 0.8,
                     "marker": "o",
                     "color": colors[idx],
                     "edgecolors": "black",}
            if s:
                size = s[i:j]
            else:
                size = 50
            all_scatter += [ax.scatter(x[i:j], y[i:j], s=size, **updated_dict(param, style_kwds, idx))]
            tmp_colors += [updated_dict(param, style_kwds, idx)["color"]]
            if j < len(z):
                all_categories += [z[j]]
                current_cat = z[j]
                i = j
                idx += 1
        ax.legend(
            [Line2D([0], [0], marker="o", color="black", markerfacecolor=color, markersize=8,) for color in tmp_colors],
            all_categories,
            bbox_to_anchor=[1, 0.5],
            loc="center left",
            title=title,
            labelspacing=1,
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    else:
        param = {"alpha": 0.8,
                 "marker": "o",
                 "color": colors[0],
                 "edgecolors": "black",}
        if s:
            size = s
        else:
            size = 300
        ax.scatter(x, y, s=size, **updated_dict(param, style_kwds, 0),)
    if reverse[0]:
        ax.set_xlim(max(x) + 0.1 * (1 + max(x) - min(x)), min(x) - 0.1 - 0.1 * (1 + max(x) - min(x)))
    if reverse[1]:
        ax.set_ylim(max(y) + 0.1 * (1 + max(y) - min(y)), min(y) - 0.1 * (1 + max(y) - min(y)))
    if plt_text:
        ax.set_xlabel(x_label, loc="right")
        ax.set_ylabel(y_label, loc="top")
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        plt.text(max(x) + 0.1, max(y) + 0.1, 
                 "Modest", size=15, rotation=130.,
                 ha="center", va="center",
                 bbox=dict(boxstyle="round",
                           ec=gen_colors()[0],
                           fc=gen_colors()[0],
                           alpha=0.3),)
        plt.text(max(x) + 0.1, min(y) - 0.1, 
                 "Efficient", size=15, rotation=30.,
                 ha="center", va="center",
                 bbox=dict(boxstyle="round",
                           ec=gen_colors()[1],
                           fc=gen_colors()[1],
                           alpha=0.3),)
        plt.text(min(x) - 0.1, max(y) + 0.1, 
                 "Performant", size=15, rotation=-130.,
                 ha="center", va="center",
                 bbox=dict(boxstyle="round",
                           ec=gen_colors()[2],
                           fc=gen_colors()[2],
                           alpha=0.3),)
        plt.text(min(x) - 0.1, min(y) - 0.1, 
                 "Performant & Efficient", size=15, rotation=-30.,
                 ha="center", va="center",
                 bbox=dict(boxstyle="round",
                           ec=gen_colors()[3],
                           fc=gen_colors()[3],
                           alpha=0.3),)
    else:
        ax.set_xlabel(x_label,)
        ax.set_ylabel(y_label,)
    return ax


# ---#
def plot_BKtree(tree, pic_path: str = ""):
    try:
        from anytree import Node, RenderTree
    except:
        raise ImportError(
            "The anytree module seems to not be installed in your environment.\nTo be able to use this method, you'll have to install it."
        )
    check_types([("pic_path", pic_path, [str],)])
    try:
        import shutil

        screen_columns = shutil.get_terminal_size().columns
    except:
        import os

        screen_rows, screen_columns = os.popen("stty size", "r").read().split()
    print("-" * int(screen_columns))
    print("Bisection Levels: {}".format(max(tree["bisection_level"])))
    print("Number of Centers: {}".format(len(tree["center_id"])))
    print("Total Size: {}".format(max(tree["cluster_size"])))
    print("-" * int(screen_columns))
    tree_nodes = {}
    for idx in range(len(tree["center_id"])):
        tree_nodes[tree["center_id"][idx]] = Node(
            "[{}] (Size = {} | Score = {})".format(
                tree["center_id"][idx],
                tree["cluster_size"][idx],
                round(tree["withinss"][idx] / tree["totWithinss"][idx], 2),
            )
        )
    for idx, node_id in enumerate(tree["center_id"]):
        if (
            tree["left_child"][idx] in tree_nodes
            and tree["right_child"][idx] in tree_nodes
        ):
            tree_nodes[node_id].children = [
                tree_nodes[tree["left_child"][idx]],
                tree_nodes[tree["right_child"][idx]],
            ]
    for pre, fill, node in RenderTree(tree_nodes[0]):
        print("%s%s" % (pre, node.name))
    if pic_path:
        from anytree.dotexport import RenderTreeGraph

        RenderTreeGraph(tree_nodes[0]).to_picture(pic_path)
        if isnotebook():
            from IPython.core.display import HTML, display

            display(HTML("<img src='{}'>".format(pic_path)))
    return RenderTree(tree_nodes[0])


# ---#
def plot_tree(tree, metric: str = "probability", pic_path: str = ""):
    try:
        from anytree import Node, RenderTree
    except:
        raise ImportError(
            "The anytree module seems to not be installed in your environment.\nTo be able to use this method, you'll have to install it."
        )
    check_types([("metric", metric, [str],), ("pic_path", pic_path, [str],)])
    try:
        import shutil

        screen_columns = shutil.get_terminal_size().columns
    except:
        import os

        screen_rows, screen_columns = os.popen("stty size", "r").read().split()
    tree_id, nb_nodes, tree_depth, tree_breadth = (
        tree["tree_id"][0],
        len(tree["node_id"]),
        max(tree["node_depth"]),
        sum([1 if item else 0 for item in tree["is_leaf"]]),
    )
    print("-" * int(screen_columns))
    print("Tree Id: {}".format(tree_id))
    print("Number of Nodes: {}".format(nb_nodes))
    print("Tree Depth: {}".format(tree_depth))
    print("Tree Breadth: {}".format(tree_breadth))
    print("-" * int(screen_columns))
    tree_nodes = {}
    if "probability/variance" in tree:
        metric_tree = "probability/variance"
    else:
        metric_tree = "log_odds"
    for idx in range(nb_nodes):
        op = "<" if not (tree["is_categorical_split"][idx]) else "="
        if tree["is_leaf"][idx]:
            tree_nodes[tree["node_id"][idx]] = Node(
                "[{}] => {} ({} = {})".format(
                    tree["node_id"][idx],
                    tree["prediction"][idx],
                    metric,
                    tree[metric_tree][idx],
                )
            )
        else:
            tree_nodes[tree["node_id"][idx]] = Node(
                "[{}] ({} {} {} ?)".format(
                    tree["node_id"][idx],
                    tree["split_predictor"][idx],
                    op,
                    tree["split_value"][idx],
                )
            )
    for idx, node_id in enumerate(tree["node_id"]):
        if not (tree["is_leaf"][idx]):
            tree_nodes[node_id].children = [
                tree_nodes[tree["left_child_id"][idx]],
                tree_nodes[tree["right_child_id"][idx]],
            ]
    for pre, fill, node in RenderTree(tree_nodes[1]):
        print("%s%s" % (pre, node.name))
    if pic_path:
        from anytree.dotexport import RenderTreeGraph

        RenderTreeGraph(tree_nodes[1]).to_picture(pic_path)
        if isnotebook():
            from IPython.core.display import HTML, display

            display(HTML("<img src='{}'>".format(pic_path)))
    return RenderTree(tree_nodes[1])


# ---#
def regression_plot(
    X: list,
    y: str,
    input_relation: str,
    coefficients: list,
    cursor=None,
    max_nb_points: int = 50,
    ax=None,
    **style_kwds,
):
    check_types(
        [
            ("X", X, [list],),
            ("y", y, [str],),
            ("input_relation", input_relation, [str],),
            ("coefficients", coefficients, [list],),
            ("max_nb_points", max_nb_points, [int, float],),
        ]
    )
    cursor, conn = check_cursor(cursor)[0:2]
    param = {
        "marker": "o",
        "color": gen_colors()[0],
        "s": 50,
        "edgecolors": "black",
    }
    if len(X) == 1:
        query = "SELECT {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL LIMIT {}".format(
            X[0], y, input_relation, X[0], y, int(max_nb_points)
        )
        cursor.execute(query)
        all_points = cursor.fetchall()
        if not (ax):
            fig, ax = plt.subplots()
            if isnotebook():
                fig.set_size_inches(8, 6)
            ax.set_axisbelow(True)
            ax.grid()
        x0, y0 = (
            [float(item[0]) for item in all_points],
            [float(item[1]) for item in all_points],
        )
        min_reg, max_reg = min(x0), max(x0)
        x_reg = [min_reg, max_reg]
        y_reg = [coefficients[0] + coefficients[1] * item for item in x_reg]
        ax.plot(x_reg, y_reg, alpha=1, color="black")
        ax.scatter(
            x0, y0, **updated_dict(param, style_kwds, 0),
        )
        ax.set_xlabel(X[0])
        ax.set_ylabel(y)
    elif len(X) == 2:
        query = "(SELECT {}, {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL LIMIT {})".format(
            X[0], X[1], y, input_relation, X[0], X[1], y, int(max_nb_points)
        )
        cursor.execute(query)
        all_points = cursor.fetchall()
        x0, y0, z0 = (
            [float(item[0]) for item in all_points],
            [float(item[1]) for item in all_points],
            [float(item[2]) for item in all_points],
        )
        min_reg_x, max_reg_x = min(x0), max(x0)
        step_x = (max_reg_x - min_reg_x) / 40.0
        min_reg_y, max_reg_y = min(y0), max(y0)
        step_y = (max_reg_y - min_reg_y) / 40.0
        X_reg = (
            arange(min_reg_x - 5 * step_x, max_reg_x + 5 * step_x, step_x)
            if (step_x > 0)
            else [max_reg_x]
        )
        Y_reg = (
            arange(min_reg_y - 5 * step_y, max_reg_y + 5 * step_y, step_y)
            if (step_y > 0)
            else [max_reg_y]
        )
        X_reg, Y_reg = np.meshgrid(X_reg, Y_reg)
        Z_reg = coefficients[0] + coefficients[1] * X_reg + coefficients[2] * Y_reg
        if not (ax):
            if isnotebook():
                plt.figure(figsize=(8, 6))
            ax = plt.axes(projection="3d")
        ax.plot_surface(
            X_reg, Y_reg, Z_reg, rstride=1, cstride=1, alpha=0.5, color="gray"
        )
        ax.scatter(
            x0, y0, z0, **updated_dict(param, style_kwds, 0),
        )
        ax.set_xlabel(X[0])
        ax.set_ylabel(X[1])
        ax.set_zlabel(y + " = f(" + X[0] + ", " + X[1] + ")")
    else:
        raise ParameterError("The number of predictors is too big.")
    if conn:
        conn.close()
    return ax


# ---#
def regression_tree_plot(
    X: list,
    y: str,
    input_relation: str,
    cursor=None,
    max_nb_points: int = 10000,
    ax=None,
    **style_kwds,
):
    check_types(
        [
            ("X", X, [list],),
            ("y", y, [str],),
            ("input_relation", input_relation, [str],),
            ("max_nb_points", max_nb_points, [int, float],),
        ]
    )
    cursor, conn = check_cursor(cursor)[0:2]

    query = "SELECT {}, {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL ORDER BY RANDOM() LIMIT {}".format(
        X[0], X[1], y, input_relation, X[0], X[1], y, int(max_nb_points),
    )
    cursor.execute(query)
    all_points = cursor.fetchall()
    if not (ax):
        fig, ax = plt.subplots()
        if isnotebook():
            fig.set_size_inches(8, 6)
        ax.set_axisbelow(True)
        ax.grid()
    x0, x1, y0, y1 = (
        [float(item[0]) for item in all_points],
        [float(item[0]) for item in all_points],
        [float(item[2]) for item in all_points],
        [float(item[1]) for item in all_points],
    )
    x0, y0 = zip(*sorted(zip(x0, y0)))
    x1, y1 = zip(*sorted(zip(x1, y1)))
    color = "black"
    if "color" in style_kwds:
        if not (isinstance(style_kwds["color"], str)) and len(style_kwds["color"]) > 1:
            color = style_kwds["color"][1]
    ax.step(x1, y1, color=color)
    param = {
        "marker": "o",
        "color": gen_colors()[0],
        "s": 50,
        "edgecolors": "black",
    }
    ax.scatter(
        x0, y0, **updated_dict(param, style_kwds,),
    )
    ax.set_xlabel(X[0])
    ax.set_ylabel(y)
    if conn:
        conn.close()
    return ax


# ---#
def svm_classifier_plot(
    X: list,
    y: str,
    input_relation: str,
    coefficients: list,
    cursor=None,
    max_nb_points: int = 500,
    ax=None,
    **style_kwds,
):
    check_types(
        [
            ("X", X, [list],),
            ("y", y, [str],),
            ("input_relation", input_relation, [str],),
            ("coefficients", coefficients, [list],),
            ("max_nb_points", max_nb_points, [int, float],),
        ]
    )
    cursor, conn = check_cursor(cursor)[0:2]
    param0 = {
        "marker": "o",
        "color": gen_colors()[0],
        "s": 50,
        "edgecolors": "black",
    }
    param1 = {
        "marker": "o",
        "color": gen_colors()[1],
        "s": 50,
        "edgecolors": "black",
    }
    if len(X) == 1:
        query = "(SELECT {}, {} FROM {} WHERE {} IS NOT NULL AND {} = 0 LIMIT {})".format(
            X[0], y, input_relation, X[0], y, int(max_nb_points / 2)
        )
        query += " UNION ALL (SELECT {}, {} FROM {} WHERE {} IS NOT NULL AND {} = 1 LIMIT {})".format(
            X[0], y, input_relation, X[0], y, int(max_nb_points / 2)
        )
        cursor.execute(query)
        all_points = cursor.fetchall()
        if not (ax):
            fig, ax = plt.subplots()
            if isnotebook():
                fig.set_size_inches(8, 6)
            ax.set_axisbelow(True)
            ax.grid()
        x0, x1 = [], []
        for idx, item in enumerate(all_points):
            if item[1] == 0:
                x0 += [float(item[0])]
            else:
                x1 += [float(item[0])]
        x_svm, y_svm = (
            [-coefficients[0] / coefficients[1], -coefficients[0] / coefficients[1]],
            [-1, 1],
        )
        ax.plot(x_svm, y_svm, alpha=1, color="black")
        all_scatter = [
            ax.scatter(x0, [0 for item in x0], **updated_dict(param1, style_kwds, 1),)
        ]
        all_scatter += [
            ax.scatter(x1, [0 for item in x1], **updated_dict(param0, style_kwds, 0),)
        ]
        ax.set_xlabel(X[0])
        ax.legend(
            all_scatter,
            [0, 1],
            scatterpoints=1,
            loc="center left",
            bbox_to_anchor=[1, 0.5],
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    elif len(X) == 2:
        query = "(SELECT {}, {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL AND {} = 0 LIMIT {})".format(
            X[0], X[1], y, input_relation, X[0], X[1], y, int(max_nb_points / 2)
        )
        query += " UNION (SELECT {}, {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL AND {} = 1 LIMIT {})".format(
            X[0], X[1], y, input_relation, X[0], X[1], y, int(max_nb_points / 2)
        )
        cursor.execute(query)
        all_points = cursor.fetchall()
        if not (ax):
            fig, ax = plt.subplots()
            if isnotebook():
                fig.set_size_inches(8, 6)
            ax.set_axisbelow(True)
            ax.grid()
        x0, x1, y0, y1 = [], [], [], []
        for idx, item in enumerate(all_points):
            if item[2] == 0:
                x0 += [float(item[0])]
                y0 += [float(item[1])]
            else:
                x1 += [float(item[0])]
                y1 += [float(item[1])]
        min_svm, max_svm = min(x0 + x1), max(x0 + x1)
        x_svm, y_svm = (
            [min_svm, max_svm],
            [
                -(coefficients[0] + coefficients[1] * min_svm) / coefficients[2],
                -(coefficients[0] + coefficients[1] * max_svm) / coefficients[2],
            ],
        )
        ax.plot(x_svm, y_svm, alpha=1, color="black")
        all_scatter = [ax.scatter(x0, y0, **updated_dict(param1, style_kwds, 1),)]
        all_scatter += [ax.scatter(x1, y1, **updated_dict(param0, style_kwds, 0),)]
        ax.set_xlabel(X[0])
        ax.set_ylabel(X[1])
        ax.legend(
            all_scatter,
            [0, 1],
            scatterpoints=1,
            loc="center left",
            bbox_to_anchor=[1, 0.5],
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    elif len(X) == 3:
        query = "(SELECT {}, {}, {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL AND {} = 0 LIMIT {})".format(
            X[0],
            X[1],
            X[2],
            y,
            input_relation,
            X[0],
            X[1],
            X[2],
            y,
            int(max_nb_points / 2),
        )
        query += " UNION (SELECT {}, {}, {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL AND {} IS NOT NULL AND {} = 1 LIMIT {})".format(
            X[0],
            X[1],
            X[2],
            y,
            input_relation,
            X[0],
            X[1],
            X[2],
            y,
            int(max_nb_points / 2),
        )
        cursor.execute(query)
        all_points = cursor.fetchall()
        x0, x1, y0, y1, z0, z1 = [], [], [], [], [], []
        for idx, item in enumerate(all_points):
            if item[3] == 0:
                x0 += [float(item[0])]
                y0 += [float(item[1])]
                z0 += [float(item[2])]
            else:
                x1 += [float(item[0])]
                y1 += [float(item[1])]
                z1 += [float(item[2])]
        min_svm_x, max_svm_x = min(x0 + x1), max(x0 + x1)
        step_x = (max_svm_x - min_svm_x) / 40.0
        min_svm_y, max_svm_y = min(y0 + y1), max(y0 + y1)
        step_y = (max_svm_y - min_svm_y) / 40.0
        X_svm = (
            arange(min_svm_x - 5 * step_x, max_svm_x + 5 * step_x, step_x)
            if (step_x > 0)
            else [max_svm_x]
        )
        Y_svm = (
            arange(min_svm_y - 5 * step_y, max_svm_y + 5 * step_y, step_y)
            if (step_y > 0)
            else [max_svm_y]
        )
        X_svm, Y_svm = np.meshgrid(X_svm, Y_svm)
        Z_svm = coefficients[0] + coefficients[1] * X_svm + coefficients[2] * Y_svm
        if not (ax):
            if isnotebook():
                plt.figure(figsize=(8, 6))
            ax = plt.axes(projection="3d")
        ax.plot_surface(
            X_svm, Y_svm, Z_svm, rstride=1, cstride=1, alpha=0.5, color="gray"
        )
        param0["alpha"] = 0.8
        all_scatter = [ax.scatter(x0, y0, z0, **updated_dict(param1, style_kwds, 1),)]
        all_scatter += [ax.scatter(x1, y1, z1, **updated_dict(param0, style_kwds, 0),)]
        ax.set_xlabel(X[0])
        ax.set_ylabel(X[1])
        ax.set_zlabel(X[2])
        ax.legend(
            all_scatter,
            [0, 1],
            scatterpoints=1,
            title=y,
            loc="center left",
            bbox_to_anchor=[1.1, 0.5],
            ncol=1,
            fontsize=8,
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    else:
        raise ParameterError("The number of predictors is too big.")
    if conn:
        conn.close()
    return ax


# ---#
def voronoi_plot(
    clusters: list,
    columns: list,
    input_relation: str,
    max_nb_points: int = 1000,
    plot_crosses: bool = True,
    cursor=None,
    ax=None,
    **style_kwds,
):
    check_types(
        [
            ("clusters", clusters, [list],),
            ("columns", columns, [list],),
            ("input_relation", input_relation, [str],),
            ("max_nb_points", max_nb_points, [int],),
        ]
    )
    cursor, conn = check_cursor(cursor)[0:2]
    from scipy.spatial import voronoi_plot_2d, Voronoi

    min_x, max_x, min_y, max_y = (
        min([elem[0] for elem in clusters]),
        max([elem[0] for elem in clusters]),
        min([elem[1] for elem in clusters]),
        max([elem[1] for elem in clusters]),
    )
    dummies_point = [
        [min_x - 999, min_y - 999],
        [min_x - 999, max_y + 999],
        [max_x + 999, min_y - 999],
        [max_x + 999, max_y + 999],
    ]
    v = Voronoi(clusters + dummies_point)
    param = {"show_vertices": False}
    voronoi_plot_2d(
        v, ax=ax, **updated_dict(param, style_kwds,),
    )
    if not (ax):
        ax = plt
        ax.xlabel(columns[0])
        ax.ylabel(columns[1])
    colors = gen_colors()
    for idx, region in enumerate(v.regions):
        if not -1 in region:
            polygon = [v.vertices[i] for i in region]
            if "color" in style_kwds:
                if isinstance(style_kwds["color"], str):
                    color = style_kwds["color"]
                else:
                    color = style_kwds["color"][idx % len(style_kwds["color"])]
            else:
                color = colors[idx % len(colors)]
            ax.fill(*zip(*polygon), alpha=0.4, color=color)
    ax.plot([elem[0] for elem in clusters], [elem[1] for elem in clusters], "ko")
    ax.xlim(min_x - 0.05 * (max_x - min_x), max_x + 0.05 * (max_x - min_x))
    ax.ylim(min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y))
    if max_nb_points > 0:
        query = "SELECT {}, {} FROM {} WHERE {} IS NOT NULL AND {} IS NOT NULL ORDER BY RANDOM() LIMIT {}".format(
            columns[0],
            columns[1],
            input_relation,
            columns[0],
            columns[1],
            int(max_nb_points),
        )
        cursor.execute(query)
        all_points = cursor.fetchall()
        x, y = (
            [float(item[0]) for item in all_points],
            [float(item[1]) for item in all_points],
        )
        ax.scatter(
            x, y, color="black", s=10, alpha=1, zorder=3,
        )
        if plot_crosses:
            ax.scatter(
                [elem[0] for elem in clusters],
                [elem[1] for elem in clusters],
                color="white",
                s=200,
                linewidths=5,
                alpha=1,
                zorder=4,
                marker="x",
            )
    if conn:
        conn.close()
    return ax
