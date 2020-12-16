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
from random import shuffle
import math, statistics, warnings

# Other Python Modules
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

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
    color="#444444",
    title="",
    confidence=None,
    type_bar: bool = True,
    ax=None,
):
    if not (ax):
        fig, ax = plt.subplots()
        if isnotebook():
            fig.set_size_inches(10, 3)
        ax.set_facecolor("#F5F5F5")
    if type_bar:
        ax.bar(x, y, width=0.007 * len(x), color=color, zorder=1, linewidth=0)
        ax.scatter(
            x, y, s=90, marker="o", facecolors="#FE5016", edgecolors="#FE5016", zorder=2
        )
        ax.plot(
            [-1] + x + [x[-1] + 1],
            [0 for elem in range(len(x) + 2)],
            color="#FE5016",
            zorder=0,
        )
        ax.set_xlim(-1, x[-1] + 1)
    else:
        ax.plot(x, y, color=color)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=90)
    if confidence:
        ax.fill_between(x, confidence, color="#FE5016", alpha=0.1)
        ax.fill_between(x, [-elem for elem in confidence], color="#FE5016", alpha=0.1)
    ax.set_xlabel("lag")
    return ax


# ---#
def bar(
    vdf,
    method: str = "density",
    of=None,
    max_cardinality: int = 6,
    bins: int = 0,
    h: float = 0,
    color: str = "#FE5016",
    ax=None,
):
    x, y, z, h, is_categorical = compute_plot_variables(
        vdf, method=method, of=of, max_cardinality=max_cardinality, bins=bins, h=h
    )
    if not (ax):
        fig, ax = plt.subplots()
        if isnotebook():
            fig.set_size_inches(10, min(int(len(x) / 1.8) + 1, 600))
        ax.set_facecolor("#F5F5F5")
        ax.xaxis.grid()
        ax.set_axisbelow(True)
    ax.barh(x, y, h, color=color, alpha=0.86)
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
        ax.set_yticks([elem - h / 2 / 0.94 for elem in x])
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
    ax=None,
):
    colors = gen_colors()
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
            fig.set_size_inches(10, min(m * 3, 600) / 2 + 1)
        ax.set_facecolor("#F5F5F5")
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
                    alpha=0.86,
                    color=colors[(i - 1) % len(colors)],
                    label=current_label,
                    left=last_column,
                )
            else:
                ax.barh(
                    [elem + (i - 1) * bar_width / (n - 1) for elem in range(n_groups)],
                    current_column,
                    bar_width / (n - 1),
                    alpha=0.86,
                    color=colors[(i - 1) % len(colors)],
                    label=current_label,
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
            ax.barh(
                [elem for elem in range(n_groups)],
                current_column,
                bar_width,
                alpha=0.86,
                color=colors[(i - 1) % len(colors)],
                label=current_label,
                left=last_column,
            )
        ax.set_yticks([elem for elem in range(n_groups)])
        ax.set_yticklabels(all_columns[0][1:m])
        ax.set_ylabel(columns[0])
    ax.legend(title=columns[1], loc="center left", bbox_to_anchor=[1, 0.5])
    return ax


# ---#
def boxplot(
    vdf,
    by: str = "",
    h: float = 0,
    max_cardinality: int = 8,
    cat_priority: list = [],
    ax=None,
):
    # SINGLE BOXPLOT
    if by == "":
        if not (ax):
            fig, ax = plt.subplots()
            if isnotebook():
                fig.set_size_inches(6, 4)
            ax.set_facecolor("#F5F5F5")
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
        )
        for median in box["medians"]:
            median.set(
                color="black", linewidth=1,
            )
        for patch in box["boxes"]:
            patch.set_facecolor("#FE5016")
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
                ax.set_facecolor("#F5F5F5")
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
            )
            ax.set_xticklabels(labels, rotation=90)
            colors = gen_colors()
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
def boxplot2D(vdf, columns: list = [], ax=None):
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
        vdf[columns[0]].boxplot()
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
                ax.set_facecolor("#F5F5F5")
            box = ax.boxplot(
                result,
                notch=False,
                sym="",
                whis=float("Inf"),
                widths=0.5,
                labels=columns,
                patch_artist=True,
            )
            ax.set_xticklabels(columns, rotation=90)
            colors = gen_colors()
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
    max_nb_points: int = 1000,
    bbox: list = [],
    img: str = "",
    ax=None,
):
    colors = gen_colors()
    if not (catcol):
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
        max_size = max([float(item[2]) for item in query_result])
        min_size = min([float(item[2]) for item in query_result])
        column1, column2, size = (
            [float(item[0]) for item in query_result],
            [float(item[1]) for item in query_result],
            [
                1000 * (float(item[2]) - min_size) / max((max_size - min_size), 1e-50)
                for item in query_result
            ],
        )
        if not (ax):
            fig, ax = plt.subplots()
            if isnotebook():
                fig.set_size_inches(10, 6)
            ax.set_facecolor("#F5F5F5")
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
        scatter = ax.scatter(column1, column2, color=colors[0], s=size, alpha=0.5)
        kw = dict(
            prop="sizes",
            num=6,
            color=colors[0],
            alpha=0.6,
            func=lambda s: (s * (max_size - min_size) + min_size) / 1000,
        )
        ax.legend(
            *scatter.legend_elements(**kw),
            bbox_to_anchor=[1, 0.5],
            loc="center left",
            title=columns[2]
        )
    else:
        count = vdf.shape()[0]
        all_categories = vdf[catcol].distinct()
        if not (ax):
            fig, ax = plt.subplots()
            if isnotebook():
                fig.set_size_inches(12, 7)
            ax.set_facecolor("#F5F5F5")
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
        groupby_cardinality = vdf[catcol].nunique(True)
        count = vdf.shape()[0]
        tablesample = 0.1 if (count > 10000) else 0.9
        all_columns, all_scatter = [], []
        max_size, min_size = float(vdf[columns[2]].max()), float(vdf[columns[2]].min())
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
            column1, column2, size = (
                [float(item[0]) for item in query_result],
                [float(item[1]) for item in query_result],
                [
                    1000
                    * (float(item[2]) - min_size)
                    / max((max_size - min_size), 1e-50)
                    for item in query_result
                ],
            )
            all_columns += [[column1, column2, size]]
            all_scatter += [
                ax.scatter(
                    column1, column2, s=size, alpha=0.8, color=colors[idx % len(colors)]
                )
            ]
        for idx, item in enumerate(all_categories):
            if len(str(item)) > 20:
                all_categories[idx] = str(item)[0:20] + "..."
        kw = dict(
            prop="sizes",
            num=6,
            color=colors[0],
            alpha=0.6,
            func=lambda s: (s * (max_size - min_size) + min_size) / 1000,
        )
        leg1 = ax.legend(
            *all_scatter[0].legend_elements(**kw),
            bbox_to_anchor=[1, 0.5],
            loc="center left",
            title=columns[2]
        )
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        leg2 = ax.legend(
            all_scatter,
            all_categories,
            title=catcol,
            loc="center right",
            bbox_to_anchor=[-0.06, 0.5],
        )
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
    cmap: str = "PRGn",
    title: str = "",
    colorbar: str = "",
    x_label: str = "",
    y_label: str = "",
    with_numbers: bool = True,
    mround: int = 3,
    is_vector: bool = False,
    interpolation: str = "nearest",
    inverse: bool = False,
    extent: list = [],
    is_pivot: bool = False,
    ax=None,
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
    ax.set_title(title)
    if ((vmax == 1) and vmin in [0, -1]) and not (extent):
        im = ax.imshow(
            matrix_array, cmap=cmap, interpolation=interpolation, vmax=vmax, vmin=vmin
        )
    else:
        try:
            im = ax.imshow(
                matrix_array, cmap=cmap, interpolation=interpolation, extent=extent
            )
        except:
            im = ax.imshow(matrix_array, cmap=cmap, interpolation=interpolation,)
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
def gen_cmap():
    cm1 = LinearSegmentedColormap.from_list("vml", ["#FFFFFF", gen_colors()[0]], N=1000)
    cm2 = LinearSegmentedColormap.from_list(
        "vml", [gen_colors()[1], "#FFFFFF", gen_colors()[0]], N=1000
    )
    return (cm1, cm2)


# ---#
def gen_colors():
    if not (verticapy.options["colors"]) or not (
        isinstance(verticapy.options["colors"], list)
    ):
        colors = [
            "#FE5016",
            "#263133",
            "#0073E7",
            "#19A26B",
            "#FCDB1F",
            "#2A6A74",
            "#861889",
            "#00B4E0",
            "#90EE90",
            "#FF7F50",
            "#B03A89",
        ]
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
    cmap: str = "Reds",
    gridsize: int = 10,
    color: str = "white",
    bbox: list = [],
    img: str = "",
    ax=None,
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
    imh = ax.hexbin(
        column1,
        column2,
        C=column3,
        reduce_C_function=reduce_C_function,
        gridsize=gridsize,
        color=color,
        cmap=cmap,
        mincnt=1,
        edgecolors=None,
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
    color: str = "#FE5016",
    ax=None,
):
    x, y, z, h, is_categorical = compute_plot_variables(
        vdf, method, of, max_cardinality, bins, h
    )
    is_numeric = vdf.isnum()
    if not (ax):
        fig, ax = plt.subplots()
        if isnotebook():
            fig.set_size_inches(min(int(len(x) / 1.8) + 1, 600), 6)
        ax.set_facecolor("#F5F5F5")
        ax.set_axisbelow(True)
        ax.yaxis.grid()
    ax.bar(x, y, h, color=color, alpha=0.86)
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
        ax.set_xticks([elem - h / 2 / 0.94 for elem in x])
        ax.set_xticklabels([elem - h / 2 / 0.94 for elem in x], rotation=90)
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
        ax.set_facecolor("#F5F5F5")
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
                alpha=0.86,
                color=colors[(i - 1) % len(colors)],
                label=current_label,
                bottom=last_column,
            )
        else:
            ax.bar(
                [elem + (i - 1) * bar_width / (n - 1) for elem in range(n_groups)],
                current_column,
                bar_width / (n - 1),
                alpha=0.86,
                color=colors[(i - 1) % len(colors)],
                label=current_label,
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
    return ax


# ---#
def multiple_hist(
    vdf, columns: list, method: str = "density", of: str = "", h: float = 0, ax=None
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
            ax.set_facecolor("#F5F5F5")
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
                plt.bar(
                    x, y, h, color=colors[idx % len(colors)], alpha=alpha, label=column
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
        return ax


# ---#
def multi_ts_plot(
    vdf,
    order_by: str,
    columns: list = [],
    order_by_start: str = "",
    order_by_end: str = "",
    ax=None,
):
    if len(columns) == 1:
        return vdf[columns[0]].plot(
            ts=order_by, start_date=order_by_start, end_date=order_by_end, area=False
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
        ax.set_facecolor("#F5F5F5")
        ax.grid()
    for i in range(0, len(columns)):
        ax.plot(
            order_by_values,
            [item[i + 1] for item in query_result],
            color=colors[i],
            label=columns[i],
            linewidth=2,
        )
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.set_xlabel(order_by)
    ax.legend(title="columns", loc="center left", bbox_to_anchor=[1, 0.5])
    return ax


# ---#
def pie(
    vdf,
    method: str = "density",
    of=None,
    max_cardinality: int = 6,
    h: float = 0,
    donut: bool = False,
    ax=None,
):
    colors = gen_colors() * 100
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
    if not (ax):
        fig, ax = plt.subplots()
        if isnotebook():
            fig.set_size_inches(8, 6)
        ax.set_facecolor("#F5F5F5")
    if donut:
        explode = None
        centre_circle = plt.Circle(
            (0, 0), 0.72, color="#666666", fc="white", linewidth=1.25
        )
        ax.add_artist(centre_circle)
    ax.pie(
        y,
        labels=z,
        autopct=autopct,
        colors=colors,
        shadow=True,
        startangle=290,
        explode=explode,
    )
    if method.lower() == "density":
        ax.set_title("Density")
    elif (
        method.lower() in ["avg", "min", "max", "sum", "mean"] or ("%" == method[-1])
    ) and (of):
        ax.set_title(method + "(" + of + ")")
    elif method.lower() == "count":
        ax.set_title("Frequency")
    else:
        ax.set_title(method)
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
    cmap: str = "Reds",
    with_numbers: bool = True,
    ax=None,
    interpolation: str = "nearest",
    return_ax: bool = False,
    extent: list = [],
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
        query = "SELECT {} AS {}, {} FROM {}{} GROUP BY 1 {}".format(
            convert_special_type(vdf[columns[0]].category(), True, all_columns[-1]),
            columns[0],
            aggregate,
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
            cmap=cmap,
            title="",
            colorbar=aggregate,
            x_label=columns[1],
            y_label=columns[0],
            with_numbers=with_numbers,
            interpolation=interpolation,
            inverse=True,
            extent=extent,
            ax=ax,
            is_pivot=True,
        )
        if return_ax:
            return ax
    values = {all_columns[0][0]: all_columns[0][1 : len(all_columns[0])]}
    del all_columns[0]
    for column in all_columns:
        values[column[0]] = column[1 : len(column)]
    return tablesample(values=values,)


# ---#
def scatter_matrix(vdf, columns: list = []):
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
        plt.subplots(nrows=n, ncols=n, figsize=(min(1.5 * n, 500), min(1.5 * n, 500)))
        if isnotebook()
        else plt.subplots(
            nrows=n, ncols=n, figsize=(min(int(n / 1.1), 500), min(int(n / 1.1), 500))
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
    all_h = []
    for idx, column in enumerate(columns):
        all_h += [vdf[column].numh()]
    h = min(all_h)
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
            axes[i][j].set_facecolor("#F0F0F0")
            y = columns[j]
            if x == y:
                x0, y0, z0, h0, is_categorical = compute_plot_variables(
                    vdf[x], method="density", h=h, max_cardinality=1
                )
                axes[i, j].bar(x0, y0, h0 / 0.94, color="#FE5016")
            else:
                axes[i, j].scatter(
                    all_scatter_columns[j],
                    all_scatter_columns[i],
                    color=gen_colors()[1],
                    s=4,
                    marker="o",
                )
    fig.suptitle(
        "Scatter Plot Matrix of {}".format(vdf._VERTICAPY_VARIABLES_["input_relation"])
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
            ax.set_facecolor("#F5F5F5")
            ax.grid()
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
            ax.grid()
            ax.set_axisbelow(True)
        ax.set_ylabel(columns[1])
        ax.set_xlabel(columns[0])
        ax.scatter(column1, column2, color=colors[0], s=14)
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
            ax.set_facecolor("#F5F5F5")
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
            all_scatter += [
                ax.scatter(
                    column1,
                    column2,
                    alpha=0.8,
                    marker=markers[idx],
                    color=colors[idx % len(colors)],
                )
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
            all_scatter += [
                ax.scatter(
                    column1,
                    column2,
                    alpha=0.8,
                    marker=markers[idx + 1],
                    color=colors[(idx + 1) % len(colors)],
                )
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
            ax.scatter(column1, column2, column3, color=colors[0])
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
                all_scatter += [
                    ax.scatter(
                        column1,
                        column2,
                        column3,
                        alpha=0.8,
                        marker=markers[idx],
                        color=colors[idx % len(colors)],
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
                all_scatter += [
                    ax.scatter(
                        column1,
                        column2,
                        alpha=0.8,
                        marker=markers[idx + 1],
                        color=colors[(idx + 1) % len(colors)],
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
                bbox_to_anchor=[1, 0.5],
            )
            return ax


# ---#
def ts_plot(
    vdf,
    order_by: str,
    by: str = "",
    order_by_start: str = "",
    order_by_end: str = "",
    color: str = "#FE5016",
    area: bool = False,
    ax=None,
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
            ax.set_facecolor("#F5F5F5")
            ax.grid()
        ax.plot(
            order_by_values, column_values, color=color, linewidth=2,
        )
        if area:
            area_label = "Area "
            ax.fill_between(order_by_values, column_values, facecolor=color)
        else:
            area_label = ""
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        ax.set_xlabel(order_by)
        ax.set_ylabel(vdf.alias)
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
            ax.set_facecolor("#F5F5F5")
            ax.grid()
        for idx, elem in enumerate(all_data):
            ax.plot(elem[0], elem[1], color=colors[idx % len(colors)], label=elem[2])
        ax.set_xlabel(order_by)
        ax.set_ylabel(vdf.alias)
        ax.legend(title=by, loc="center left", bbox_to_anchor=[1, 0.5])
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        return ax
