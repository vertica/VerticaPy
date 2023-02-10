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
import warnings, copy

# MATPLOTLIB
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# NUMPY
import numpy as np

# VerticaPy Modules
from verticapy.plotting._matplotlib.core import updated_dict
from verticapy._config._random import current_random
from verticapy._config._notebook import ISNOTEBOOK
from verticapy.sql.read import _executeSQL
from verticapy.errors import ParameterError
from verticapy.plotting._matplotlib.core import compute_plot_variables
from verticapy.plotting._colors import gen_colors, gen_cmap, get_color

# Global Variables
MARKERS = ["^", "o", "+", "*", "h", "x", "D", "1"]


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
    assert not (catcol) or not (cmap_col), ParameterError(
        "Bubble Plot only accepts either a cmap column or a categorical column. It can not accept both."
    )
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
    if not (catcol) and not (cmap_col):
        tablesample = max_nb_points / vdf.shape()[0]
        query_result = _executeSQL(
            query=f"""
                SELECT 
                    /*+LABEL('plotting._matplotlib.bubble')*/ 
                    {columns[0]}, 
                    {columns[1]}, 
                    {columns[2]} 
                FROM {vdf.__genSQL__(True)} 
                WHERE __verticapy_split__ < {tablesample} 
                  AND {columns[0]} IS NOT NULL
                  AND {columns[1]} IS NOT NULL
                  AND {columns[2]} IS NOT NULL 
                LIMIT {max_nb_points}""",
            title="Selecting random points to draw the scatter plot",
            method="fetchall",
        )
        size = 50
        if columns[2] != 1:
            max_size = max([float(item[2]) for item in query_result])
            min_size = min([float(item[2]) for item in query_result])
            size = [
                1000 * (float(item[2]) - min_size) / max((max_size - min_size), 1e-50)
                for item in query_result
            ]
        column1, column2 = (
            [float(item[0]) for item in query_result],
            [float(item[1]) for item in query_result],
        )
        if not (ax):
            fig, ax = plt.subplots()
            if ISNOTEBOOK:
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
            args_legends = [[], []]
            for i, fun in enumerate([min, max]):
                args_legends[0] += [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=colors[0],
                        label="Scatter",
                        markersize=fun(size) / 100 + 15,
                    )
                ]
                args_legends[1] += [fun([x[2] for x in query_result])]
            leg1 = ax.legend(
                *args_legends,
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
            if ISNOTEBOOK:
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
            max_size, min_size = (
                float(vdf[columns[2]].max()),
                float(vdf[columns[2]].min()),
            )
        custom_lines = []
        if catcol:
            all_categories = vdf[catcol].distinct()
            groupby_cardinality = vdf[catcol].nunique(True)
            for idx, category in enumerate(all_categories):
                category_str = str(category).replace("'", "''")
                query_result = _executeSQL(
                    query=f"""
                        SELECT
                            /*+LABEL('plotting._matplotlib.bubble')*/  
                            {columns[0]},
                            {columns[1]},
                            {columns[2]} 
                        FROM {vdf.__genSQL__(True)}
                        WHERE  __verticapy_split__ < {tablesample} 
                           AND {catcol} = '{category_str}'
                           AND {columns[0]} IS NOT NULL
                           AND {columns[1]} IS NOT NULL
                           AND {columns[2]} IS NOT NULL 
                        LIMIT {int(max_nb_points / len(all_categories))}""",
                    title=(
                        "Selecting random points to draw the "
                        f"bubble plot (category = '{category}')"
                    ),
                    method="fetchall",
                )
                size = 50
                if columns[2] != 1:
                    size = [
                        1000
                        * (float(item[2]) - min_size)
                        / max((max_size - min_size), 1e-50)
                        for item in query_result
                    ]
                column1, column2 = (
                    [float(item[0]) for item in query_result],
                    [float(item[1]) for item in query_result],
                )
                param = {
                    "alpha": 0.8,
                    "color": colors[idx % len(colors)],
                    "edgecolors": "black",
                }
                ax.scatter(
                    column1, column2, s=size, **updated_dict(param, style_kwds, idx)
                )
                custom_lines += [
                    Line2D([0], [0], color=colors[idx % len(colors)], lw=6)
                ]
            for idx, item in enumerate(all_categories):
                if len(str(item)) > 20:
                    all_categories[idx] = str(item)[0:20] + "..."
        else:
            query_result = _executeSQL(
                query=f"""
                    SELECT
                        /*+LABEL('plotting._matplotlib.bubble')*/ 
                        {columns[0]},
                        {columns[1]},
                        {columns[2]},
                        {cmap_col}
                    FROM {vdf.__genSQL__(True)}
                    WHERE  __verticapy_split__ < {tablesample} 
                       AND {columns[0]} IS NOT NULL
                       AND {columns[1]} IS NOT NULL
                       AND {columns[2]} IS NOT NULL
                       AND {cmap_col} IS NOT NULL
                    LIMIT {max_nb_points}""",
                title=(
                    "Selecting random points to draw the bubble plot with cmap expr."
                ),
                method="fetchall",
            )
            size = 50
            if columns[2] != 1:
                size = [
                    1000
                    * (float(item[2]) - min_size)
                    / max((max_size - min_size), 1e-50)
                    for item in query_result
                ]
            column1, column2, column3 = (
                [float(item[0]) for item in query_result],
                [float(item[1]) for item in query_result],
                [float(item[3]) for item in query_result],
            )
            param = {
                "alpha": 0.8,
                "cmap": gen_cmap()[0],
                "edgecolors": "black",
            }
            im = ax.scatter(
                column1, column2, c=column3, s=size, **updated_dict(param, style_kwds),
            )
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
        if ISNOTEBOOK:
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
        for i in [-1, 1]:
            ax.plot(
                [all_agg["min"][0], all_agg["max"][0]],
                [i * threshold, i * threshold],
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
        for searchi in [(">", outliers_color), ("<=", inliers_color)]:
            scatter(
                vdf_temp.search(f"ZSCORE {searchi[0]} {threshold}"),
                [columns[0], "ZSCORE"],
                max_nb_points=max_nb_points,
                ax=ax,
                color=searchi[1],
                **style_kwds,
            )
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
        s = []
        for op, color in [("OR", outliers_color), ("AND", inliers_color)]:
            s = f"""
                ABS(({columns[0]} - {all_agg["avg"][0]}) 
                / {all_agg["std"][0]}) <= {threshold} 
           {op} ABS(({columns[1]} - {all_agg["avg"][1]}) 
                / {all_agg["std"][1]}) <= {threshold}"""
            scatter(
                vdf.search(s),
                columns,
                max_nb_points=max_nb_points,
                ax=ax,
                color=color,
                **style_kwds,
            )
    args = [[0], [0]]
    kwds = {
        "marker": "o",
        "color": "black",
        "label": "Scatter",
        "markersize": 8,
    }
    ax.legend(
        [
            Line2D(*args, color=inliers_border_color, lw=4),
            Line2D(*args, **kwds, markerfacecolor=inliers_color),
            Line2D(*args, **kwds, markerfacecolor=outliers_color),
        ],
        ["threshold", "inliers", "outliers"],
        loc="center left",
        bbox_to_anchor=[1, 0.5],
        labelspacing=1,
    )
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    return ax


def scatter_matrix(
    vdf, columns: list = [], **style_kwds,
):
    columns = vdf.format_colnames(columns)
    if not (columns):
        columns = vdf.numcol()
    elif len(columns) == 1:
        return vdf[columns[0]].hist()
    n = len(columns)
    if ISNOTEBOOK:
        figsize = min(1.5 * (n + 1), 500), min(1.5 * (n + 1), 500)
        fig, axes = plt.subplots(nrows=n, ncols=n, figsize=figsize,)
    else:
        figsize = min(int((n + 1) / 1.1), 500), min(int((n + 1) / 1.1), 500)
        fig, axes = plt.subplots(nrows=n, ncols=n, figsize=figsize,)
    random_func = current_random()
    all_scatter_points = _executeSQL(
        query=f"""
            SELECT 
                /*+LABEL('plotting._matplotlib.scatter_matrix')*/
                {", ".join(columns)},
                {random_func} AS rand
            FROM {vdf.__genSQL__(True)}
            WHERE __verticapy_split__ < 0.5
            ORDER BY rand 
            LIMIT 1000""",
        title="Selecting random points to draw the scatter plot",
        method="fetchall",
    )
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
                param = {"color": get_color(style_kwds, 0), "edgecolor": "black"}
                if "edgecolor" in style_kwds:
                    param["edgecolor"] = style_kwds["edgecolor"]
                axes[i, j].bar(x0, y0, h0 / 0.94, **param)
            else:
                param = {
                    "color": get_color(style_kwds, 1),
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


def scatter(
    vdf,
    columns: list,
    catcol: str = "",
    max_cardinality: int = 3,
    cat_priority: list = [],
    with_others: bool = True,
    max_nb_points: int = 1000,
    bbox: list = [],
    img: str = "",
    ax=None,
    **style_kwds,
):
    columns, catcol = vdf.format_colnames(columns, catcol, expected_nb_of_cols=[2, 3])
    n = len(columns)
    for col in columns:
        if not (vdf[col].isnum()):
            raise TypeError(
                "The parameter 'columns' must only include numerical columns."
            )
    if n == 2 and (bbox) and len(bbox) != 4:
        warning_message = (
            "Parameter 'bbox' must be a list of 4 numerics containing"
            " the 'xlim' and 'ylim'.\nIt was ignored."
        )
        warnings.warn(warning_message, Warning)
        bbox = []
    colors = gen_colors()
    markers = MARKERS * 10
    param = {
        "s": 50,
        "edgecolors": "black",
        "marker": "o",
    }
    if not (ax):
        if n == 2:
            fig, ax = plt.subplots()
            if ISNOTEBOOK:
                fig.set_size_inches(8, 6)
            ax.grid()
            ax.set_axisbelow(True)
        else:
            if ISNOTEBOOK:
                plt.figure(figsize=(8, 6))
            ax = plt.axes(projection="3d")
    all_scatter, others = [], []
    if not (catcol):
        tablesample = max_nb_points / vdf.shape()[0]
        limit = max_nb_points
    else:
        tablesample = 10 if (vdf.shape()[0] > 10000) else 90
        if cat_priority:
            all_categories = copy.deepcopy(cat_priority)
        else:
            all_categories = vdf[catcol].topk(k=max_cardinality)["index"]
        limit = int(max_nb_points / len(all_categories))
        groupby_cardinality = vdf[catcol].nunique(True)
    query = f"""
        SELECT 
            /*+LABEL('plotting._matplotlib.scatter')*/
            {columns[0]},
            {columns[1]}
            {{}}
        FROM {vdf.__genSQL__(True)}
        WHERE {{}}
              {columns[0]} IS NOT NULL
          AND {columns[1]} IS NOT NULL
          {{}}
          AND __verticapy_split__ < {tablesample} 
        LIMIT {limit}"""
    if n == 3:
        condition = [f", {columns[2]}", f"{columns[2]} IS NOT NULL AND"]
    else:
        condition = ["", ""]

    def draw_points(
        idx: int = 0,
        category: str = None,
        w_others: bool = False,
        param: dict = param,
        condition: list = condition,
        all_scatter: list = all_scatter,
        others: list = others,
        ax=ax,
    ):
        condition = copy.deepcopy(condition)
        title = "Selecting random points to draw the scatter plot"
        if not (catcol):
            param["color"] = colors[0]
            condition += [""]
        elif w_others:
            param["color"] = colors[idx + 1 % len(colors)]
            condition += ["AND" + " AND ".join(others)]
        else:
            category_str = str(category).replace("'", "''")
            param = {
                **param,
                "alpha": 0.8,
                "color": colors[idx % len(colors)],
            }
            if (max_cardinality < groupby_cardinality) or (
                len(cat_priority) < groupby_cardinality
            ):
                others += [f"{catcol} != '{category_str}'"]
            condition += [f"AND {catcol} = '{category_str}'"]
            title = f" (category = '{category}')"
        query_result = _executeSQL(
            query=query.format(*condition), title=title, method="fetchall",
        )
        args = [
            [float(d[0]) for d in query_result],
            [float(d[1]) for d in query_result],
        ]
        if n == 3:
            args += [[float(d[2]) for d in query_result]]
        all_scatter += [ax.scatter(*args, **updated_dict(param, style_kwds, idx),)]

    if not (catcol):
        draw_points()
    else:
        for idx, category in enumerate(all_categories):
            draw_points(idx, category)
        if with_others and idx + 1 < groupby_cardinality:
            all_categories += ["others"]
            draw_points(idx, w_others=True)
        for idx, c in enumerate(all_categories):
            if len(str(c)) > 20:
                all_categories[idx] = str(c)[0:20] + "..."
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    if n == 2:
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
        bbox_to_anchor = [1, 0.5]
        scatterpoints = {"scatterpoints": 1}
    elif n == 3:
        ax.set_zlabel(columns[2])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        bbox_to_anchor = [1.1, 0.5]
        scatterpoints = {}
    if catcol:
        ax.legend(
            all_scatter,
            all_categories,
            title=catcol,
            loc="center left",
            bbox_to_anchor=bbox_to_anchor,
            **scatterpoints,
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    return ax
