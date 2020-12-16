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
import math, collections

# Other Python Modules
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import numpy

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
    cursor = check_cursor(cursor)[0]

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
            ax.set_facecolor("#F5F5F5")
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
                alpha=1,
                marker="o",
                color=gen_colors()[1],
            )
        ]
        all_scatter += [
            ax.scatter(
                x1,
                [logit(coefficients[0] + coefficients[1] * item) for item in x1],
                alpha=0.8,
                marker="^",
                color=gen_colors()[0],
            )
        ]
        ax.set_xlabel(X[0])
        ax.set_ylabel("logit")
        ax.legend(all_scatter, [0, 1], scatterpoints=1)
        ax.set_title(y + " = logit(" + X[0] + ")")
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
        X_logit, Y_logit = numpy.meshgrid(X_logit, Y_logit)
        Z_logit = 1 / (
            1
            + numpy.exp(
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
                alpha=1,
                marker="o",
                color=gen_colors()[1],
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
                alpha=0.8,
                marker="^",
                color=gen_colors()[0],
            )
        ]
        ax.set_xlabel(X[0])
        ax.set_ylabel(X[1])
        ax.set_zlabel(y + " = logit(" + X[0] + ", " + X[1] + ")")
        ax.legend(
            all_scatter,
            [0, 1],
            scatterpoints=1,
            loc="lower left",
            title=y,
            bbox_to_anchor=(0.9, 1),
            ncol=2,
            fontsize=8,
        )
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
            ax.set_facecolor("#F5F5F5")
            ax.set_axisbelow(True)
            ax.grid()
        ax.set_title("Local Outlier Factor (LOF)")
        ax.set_xlabel(column)
        radius = [1000 * (item - min(lof)) / (max(lof) - min(lof)) for item in lof]
        ax.scatter(column1, column2, color=gen_colors()[1], s=14, label="Data points")
        ax.scatter(
            column1,
            column2,
            color=gen_colors()[0],
            s=radius,
            label="Outlier scores",
            facecolors="none",
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
            ax.set_facecolor("#F5F5F5")
            ax.set_axisbelow(True)
            ax.grid()
        ax.set_title("Local Outlier Factor (LOF)")
        ax.set_ylabel(columns[1])
        ax.set_xlabel(columns[0])
        radius = [1000 * (item - min(lof)) / (max(lof) - min(lof)) for item in lof]
        ax.scatter(column1, column2, color=gen_colors()[1], s=14, label="Data points")
        ax.scatter(
            column1,
            column2,
            color=gen_colors()[0],
            s=radius,
            label="Outlier scores",
            facecolors="none",
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
        ax.set_title("Local Outlier Factor (LOF)")
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        ax.set_zlabel(columns[2])
        radius = [1000 * (item - min(lof)) / (max(lof) - min(lof)) for item in lof]
        ax.scatter(
            column1, column2, column3, color=gen_colors()[1], label="Data points"
        )
        ax.scatter(column1, column2, column3, color=gen_colors()[0], s=radius)
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    else:
        raise Exception(
            "LocalOutlierFactor Plot is available for a maximum of 3 columns"
        )
    if conn:
        conn.close()
    return ax


# ---#
def plot_importance(
    coeff_importances: dict, coeff_sign: dict = {}, print_legend: bool = True, ax=None
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
        ax.set_facecolor("#F5F5F5")
        ax.set_axisbelow(True)
        ax.grid()
    color = []
    for item in signs:
        color += [gen_colors()[0]] if (item == 1) else [gen_colors()[1]]
    ax.barh(range(0, len(importances)), importances, 0.9, color=color, alpha=0.86)
    if print_legend:
        orange = mpatches.Patch(color=gen_colors()[1], label="sign -")
        blue = mpatches.Patch(color=gen_colors()[0], label="sign +")
        ax.legend(handles=[orange, blue], loc="lower right")
    ax.set_ylabel("Features")
    ax.set_xlabel("Importance")
    ax.set_yticks(range(0, len(importances)))
    ax.set_yticklabels(coefficients)
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
    for idx in range(nb_nodes):
        op = "<" if not (tree["is_categorical_split"][idx]) else "="
        if tree["is_leaf"][idx]:
            tree_nodes[tree["node_id"][idx]] = Node(
                "[{}] => {} ({} = {})".format(
                    tree["node_id"][idx],
                    tree["prediction"][idx],
                    metric,
                    tree["probability/variance"][idx],
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


# ---#
def regression_plot(
    X: list,
    y: str,
    input_relation: str,
    coefficients: list,
    cursor=None,
    max_nb_points: int = 50,
    ax=None,
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
            ax.set_facecolor("#F9F9F9")
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
        ax.scatter(x0, y0, alpha=1, marker="o", color=gen_colors()[0])
        ax.set_xlabel(X[0])
        ax.set_ylabel(y)
        ax.set_title(y + " = f(" + X[0] + ")")
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
        X_reg, Y_reg = numpy.meshgrid(X_reg, Y_reg)
        Z_reg = coefficients[0] + coefficients[1] * X_reg + coefficients[2] * Y_reg
        if not (ax):
            if isnotebook():
                plt.figure(figsize=(8, 6))
            ax = plt.axes(projection="3d")
        ax.plot_surface(
            X_reg, Y_reg, Z_reg, rstride=1, cstride=1, alpha=0.5, color="gray"
        )
        ax.scatter(x0, y0, z0, alpha=1, marker="o", color=gen_colors()[0])
        ax.set_xlabel(X[0])
        ax.set_ylabel(X[1])
        ax.set_zlabel(y + " = f(" + X[0] + ", " + X[1] + ")")
    else:
        raise ParameterError("The number of predictors is too big.")
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
            ax.set_facecolor("#F9F9F9")
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
            ax.scatter(x0, [0 for item in x0], marker="o", color=gen_colors()[1])
        ]
        all_scatter += [
            ax.scatter(x1, [0 for item in x1], marker="^", color=gen_colors()[0])
        ]
        ax.set_xlabel(X[0])
        ax.legend(all_scatter, [0, 1], scatterpoints=1)
        ax.set_title("svm(" + X[0] + ")")
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
            ax.set_facecolor("#F9F9F9")
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
        all_scatter = [ax.scatter(x0, y0, marker="o", color=gen_colors()[1])]
        all_scatter += [ax.scatter(x1, y1, marker="^", color=gen_colors()[0])]
        ax.set_xlabel(X[0])
        ax.set_ylabel(X[1])
        ax.legend(all_scatter, [0, 1], scatterpoints=1)
        ax.set_title("svm(" + X[0] + ", " + X[1] + ")")
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
        X_svm, Y_svm = numpy.meshgrid(X_svm, Y_svm)
        Z_svm = coefficients[0] + coefficients[1] * X_svm + coefficients[2] * Y_svm
        if not (ax):
            if isnotebook():
                plt.figure(figsize=(8, 6))
            ax = plt.axes(projection="3d")
        ax.plot_surface(
            X_svm, Y_svm, Z_svm, rstride=1, cstride=1, alpha=0.5, color="gray"
        )
        all_scatter = [
            ax.scatter(x0, y0, z0, alpha=1, marker="o", color=gen_colors()[1])
        ]
        all_scatter += [
            ax.scatter(x1, y1, z1, alpha=0.8, marker="^", color=gen_colors()[0])
        ]
        ax.set_xlabel(X[0])
        ax.set_ylabel(X[1])
        ax.set_zlabel(X[2])
        ax.set_title("svm(" + X[0] + ", " + X[1] + ", " + X[2] + ")")
        ax.legend(
            all_scatter,
            [0, 1],
            scatterpoints=1,
            loc="lower left",
            title=y,
            bbox_to_anchor=(0.9, 1),
            ncol=2,
            fontsize=8,
        )
    else:
        raise ParameterError("The number of predictors is too big.")
    if conn:
        conn.close()
    return ax


# ---#
def voronoi_plot(clusters: list, columns: list, ax=None):
    check_types([("clusters", clusters, [list],), ("columns", columns, [list],)])
    from scipy.spatial import voronoi_plot_2d, Voronoi

    v = Voronoi(clusters)
    voronoi_plot_2d(v, show_vertices=0, ax=ax)
    if not (ax):
        ax = plt
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    return ax
