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
import math, statistics, copy
from typing import Union

# MATPLOTLIB
import matplotlib.pyplot as plt

# NUMPY
import numpy as np

# VerticaPy Modules
from verticapy.core.tablesample import tablesample
from verticapy.plotting._matplotlib.base import updated_dict
from verticapy._utils._cast import to_varchar
from verticapy._config.config import ISNOTEBOOK
from verticapy.sql.read import to_tablesample
from verticapy._utils._sql import _executeSQL
from verticapy.core.str_sql import str_sql
from verticapy.errors import ParameterError
from verticapy.plotting._colors import gen_colors, gen_cmap
from verticapy.sql._utils._format import quote_ident


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
        if (ISNOTEBOOK and not (inverse)) or is_pivot:
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
                matrix_array, extent=extent, **updated_dict(param, style_kwds)
            )
        except:
            im = ax.imshow(matrix_array, **updated_dict(param, style_kwds))
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


def contour_plot(
    vdf,
    columns: list,
    func,
    nbins: int = 100,
    cbar_title: str = "",
    pos_label: Union[int, str, float] = None,
    ax=None,
    **style_kwds,
):
    if not (cbar_title) and str(type(func)) in (
        "<class 'function'>",
        "<class 'method'>",
    ):
        cbar_title = func.__name__
    all_agg = vdf.agg(["min", "max"], columns)
    min_x, min_y = all_agg["min"]
    max_x, max_y = all_agg["max"]
    if str(type(func)) in ("<class 'function'>", "<class 'method'>"):
        xlist = np.linspace(min_x, max_x, nbins)
        ylist = np.linspace(min_y, max_y, nbins)
        X, Y = np.meshgrid(xlist, ylist)
        Z = func(X, Y)
    else:
        from verticapy.datasets import gen_meshgrid

        vdf_tmp = gen_meshgrid(
            {
                quote_ident(columns[1])[1:-1]: {
                    "type": float,
                    "range": [min_y, max_y],
                    "nbins": nbins,
                },
                quote_ident(columns[0])[1:-1]: {
                    "type": float,
                    "range": [min_x, max_x],
                    "nbins": nbins,
                },
            }
        )
        y = "verticapy_predict"
        if isinstance(func, (str, str_sql)):
            vdf_tmp["verticapy_predict"] = func
        else:
            if func.type in (
                "XGBoostClassifier",
                "RandomForestClassifier",
                "NaiveBayes",
                "NearestCentroid",
                "KNeighborsClassifier",
            ):
                if func.type in ("NearestCentroid", "KNeighborsClassifier"):
                    vdf_tmp = func.predict_proba(
                        vdf=vdf_tmp,
                        X=columns,
                        name="verticapy_predict",
                        inplace=False,
                        key_columns=None,
                    )
                    y = f"verticapy_predict_{pos_label}"
                else:
                    vdf_tmp = func.predict_proba(
                        vdf=vdf_tmp,
                        X=columns,
                        name="verticapy_predict",
                        pos_label=pos_label,
                    )
            else:
                if func.type == "KNeighborsRegressor":
                    vdf_tmp = func.predict(
                        vdf=vdf_tmp,
                        X=columns,
                        name="verticapy_predict",
                        inplace=False,
                        key_columns=None,
                    )
                else:
                    vdf_tmp = func.predict(
                        vdf=vdf_tmp, X=columns, name="verticapy_predict"
                    )
        dataset = vdf_tmp[[columns[1], columns[0], y]].sort(columns).to_numpy()
        i, y_start, y_new = 0, dataset[0][1], dataset[0][1]
        n = len(dataset)
        X, Y, Z = [], [], []
        while i < n:
            x_tmp, y_tmp, z_tmp = [], [], []
            j, last_non_null_value = 0, 0
            while y_start == y_new and i < n and j < nbins:
                if dataset[i][2] != None:
                    last_non_null_value = float(dataset[i][2])
                x_tmp += [float(dataset[i][0])]
                y_tmp += [float(dataset[i][1])]
                z_tmp += [
                    float(
                        dataset[i][2]
                        if (dataset[i][2] != None)
                        else last_non_null_value
                    )
                ]
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
    if not (ax):
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6)
    else:
        fig = plt
    param = {"linewidths": 0.5, "levels": 14, "colors": "k"}
    param = updated_dict(param, style_kwds)
    if "cmap" in param:
        del param["cmap"]
    ax.contour(X, Y, Z, **param)
    param = {
        "cmap": gen_cmap([gen_colors()[2], "#FFFFFF", gen_colors()[0]]),
        "levels": 14,
    }
    param = updated_dict(param, style_kwds)
    for elem in ["colors", "color", "linewidths", "linestyles"]:
        if elem in param:
            del param[elem]
    cp = ax.contourf(X, Y, Z, **param)
    fig.colorbar(cp).set_label(cbar_title)
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    return ax


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
        and ((of in vdf.get_columns()) or (quote_ident(of) in vdf.get_columns()))
    ):
        aggregate = f"{method}({of})"
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
    query_result = _executeSQL(
        query=f"""
            SELECT
                /*+LABEL('plotting._matplotlib.hexbin')*/
                {columns[0]},
                {columns[1]},
                {aggregate}{over}
            FROM {vdf.__genSQL__()}
            GROUP BY {columns[0]}, {columns[1]}""",
        title="Grouping all the elements for the Hexbin Plot",
        method="fetchall",
    )
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
        if ISNOTEBOOK:
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


def pivot_table(
    vdf,
    columns: list,
    method: str = "count",
    of: str = "",
    h: tuple = (None, None),
    max_cardinality: tuple = (20, 20),
    show: bool = True,
    with_numbers: bool = True,
    fill_none: float = 0.0,
    ax=None,
    return_ax: bool = False,
    extent: list = [],
    **style_kwds,
):
    columns, of = vdf.format_colnames(columns, of)
    other_columns = ""
    method = method.lower()
    if method == "median":
        method = "50%"
    elif method == "mean":
        method = "avg"
    if (
        method not in ["avg", "min", "max", "sum", "density", "count"]
        and "%" != method[-1]
    ) and of:
        raise ParameterError(
            "Parameter 'of' must be empty when using customized aggregations."
        )
    if (method in ["avg", "min", "max", "sum"]) and (of):
        aggregate = f"{method.upper()}({of})"
    elif method and method[-1] == "%":
        aggregate = f"""APPROXIMATE_PERCENTILE({of} 
                                               USING PARAMETERS 
                                               percentile = {float(method[0:-1]) / 100})"""
    elif method in ["density", "count"]:
        aggregate = "COUNT(*)"
    elif isinstance(method, str):
        aggregate = method
        other_columns = vdf.get_columns(exclude_columns=columns)
        other_columns = ", " + ", ".join(other_columns)
    else:
        raise ParameterError(
            "The parameter 'method' must be in [count|density|avg|mean|min|max|sum|q%]"
        )
    is_column_date = [False, False]
    timestampadd = ["", ""]
    all_columns = []
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
                if vdf[column].category() == "int":
                    interval = int(max(math.floor(interval), 1))
            else:
                interval = h[idx]
            if vdf[column].category() == "int":
                floor_end = "-1"
                interval = int(max(math.floor(interval), 1))
            else:
                floor_end = ""
            expr = f"""'[' 
                      || FLOOR({column} 
                             / {interval}) 
                             * {interval} 
                      || ';' 
                      || (FLOOR({column} 
                              / {interval}) 
                              * {interval} 
                              + {interval}{floor_end}) 
                      || ']'"""
            if (interval > 1) or (vdf[column].category() == "float"):
                all_columns += [expr]
            else:
                all_columns += [f"FLOOR({column}) || ''"]
            order_by = f"""ORDER BY MIN(FLOOR({column} 
                                      / {interval}) * {interval}) ASC"""
            where += [f"{column} IS NOT NULL"]
        elif is_date:
            if h[idx] == None:
                interval = vdf[column].numh()
            else:
                interval = max(math.floor(h[idx]), 1)
            min_date = vdf[column].min()
            all_columns += [
                f"""FLOOR(DATEDIFF('second',
                                   '{min_date}',
                                   {column})
                        / {interval}) * {interval}"""
            ]
            is_column_date[idx] = True
            sql = f"""TIMESTAMPADD('second', {columns[idx]}::int, 
                                             '{min_date}'::timestamp)"""
            timestampadd[idx] = sql
            order_by = "ORDER BY 1 ASC"
            where += [f"{column} IS NOT NULL"]
        else:
            all_columns += [column]
            order_by = "ORDER BY 1 ASC"
            distinct = vdf[column].topk(max_cardinality[idx]).values["index"]
            distinct = ["'" + str(c).replace("'", "''") + "'" for c in distinct]
            if len(distinct) < max_cardinality[idx]:
                cast = to_varchar(vdf[column].category(), column)
                where += [f"({cast} IN ({', '.join(distinct)}))"]
            else:
                where += [f"({column} IS NOT NULL)"]
    where = f" WHERE {' AND '.join(where)}"
    over = "/" + str(vdf.shape()[0]) if (method == "density") else ""
    if len(columns) == 1:
        cast = to_varchar(vdf[columns[0]].category(), all_columns[-1])
        return to_tablesample(
            query=f"""
                SELECT 
                    {cast} AS {columns[0]},
                    {aggregate}{over} 
                FROM {vdf.__genSQL__()}
                {where}
                GROUP BY 1 {order_by}"""
        )
    aggr = f", {of}" if (of) else ""
    cols, cast = [], []
    for i in range(2):
        if is_column_date[0]:
            cols += [f"{timestampadd[i]} AS {columns[i]}"]
        else:
            cols += [columns[i]]
        cast += [to_varchar(vdf[columns[i]].category(), columns[i])]
    query_result = _executeSQL(
        query=f"""
            SELECT 
                /*+LABEL('plotting._matplotlib.pivot_table')*/
                {cast[0]} AS {columns[0]},
                {cast[1]} AS {columns[1]},
                {aggregate}{over}
            FROM (SELECT 
                      {cols[0]},
                      {cols[1]}
                      {aggr}
                      {other_columns} 
                  FROM 
                      (SELECT 
                          {all_columns[0]} AS {columns[0]},
                          {all_columns[1]} AS {columns[1]}
                          {aggr}
                          {other_columns} 
                       FROM {vdf.__genSQL__()}{where}) 
                       pivot_table) pivot_table_date
            WHERE {columns[0]} IS NOT NULL 
              AND {columns[1]} IS NOT NULL
            GROUP BY {columns[0]}, {columns[1]}
            ORDER BY {columns[0]}, {columns[1]} ASC""",
        title="Grouping the features to compute the pivot table",
        method="fetchall",
    )
    all_columns_categories = []
    for i in range(2):
        L = list(set([str(item[i]) for item in query_result]))
        L.sort()
        try:
            try:
                order = []
                for item in L:
                    order += [float(item.split(";")[0].split("[")[1])]
            except:
                order = [float(item) for item in L]
            L = [x for _, x in sorted(zip(order, L))]
        except:
            pass
        all_columns_categories += [copy.deepcopy(L)]
    all_column0_categories, all_column1_categories = all_columns_categories
    all_columns = [
        [fill_none for item in all_column0_categories]
        for item in all_column1_categories
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
    return tablesample(values=values)
