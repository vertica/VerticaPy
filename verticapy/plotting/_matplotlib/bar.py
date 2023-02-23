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
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from verticapy._config.colors import get_color
from verticapy._config.config import ISNOTEBOOK
from verticapy.errors import ParameterError

from verticapy.plotting._matplotlib.base import compute_plot_variables, updated_dict


def bar(
    vdf,
    method: str = "density",
    of=None,
    max_cardinality: int = 6,
    nbins: int = 0,
    h: float = 0,
    ax=None,
    **style_kwds,
):
    x, y, z, h, is_categorical = compute_plot_variables(
        vdf, method=method, of=of, max_cardinality=max_cardinality, nbins=nbins, h=h
    )
    if not (ax):
        fig, ax = plt.subplots()
        if ISNOTEBOOK:
            fig.set_size_inches(10, min(int(len(x) / 1.8) + 1, 600))
        ax.xaxis.grid()
        ax.set_axisbelow(True)
    param = {"color": get_color()[0], "alpha": 0.86}
    ax.barh(x, y, h, **updated_dict(param, style_kwds, 0))
    ax.set_ylabel(vdf._alias)
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
        aggregate = f"{method.upper()}({of})"
        ax.set_xlabel(aggregate)
    elif method.lower() == "count":
        ax.set_xlabel("Frequency")
    else:
        ax.set_xlabel(method)
    return ax


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
    colors = get_color()
    if fully_stacked:
        if method != "density":
            raise ParameterError(
                "Fully Stacked Bar works only with the 'density' method."
            )
    if density:
        if method != "density":
            raise ParameterError("Pyramid Bar works only with the 'density' method.")
        unique = vdf.nunique(columns)["approx_unique"]
        if unique[1] != 2 and unique[0] != 2:
            raise ParameterError(
                "One of the 2 columns must have 2 categories to draw a Pyramid Bar."
            )
        if unique[1] != 2:
            columns = [columns[1], columns[0]]
    all_columns = vdf.pivot_table(
        columns, method=method, of=of, h=h, max_cardinality=max_cardinality, show=False,
    ).values
    all_columns = [[column] + all_columns[column] for column in all_columns]
    n = len(all_columns)
    m = len(all_columns[0])
    n_groups = m - 1
    bar_width = 0.5
    if not (ax):
        fig, ax = plt.subplots()
        if ISNOTEBOOK:
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
                ]
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
            ax.set_xlabel(f"{method}({of})")
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


def hist(
    vdf,
    method: str = "density",
    of=None,
    max_cardinality: int = 6,
    nbins: int = 0,
    h: float = 0,
    ax=None,
    **style_kwds,
):
    x, y, z, h, is_categorical = compute_plot_variables(
        vdf, method, of, max_cardinality, nbins, h
    )
    is_numeric = vdf.isnum()
    if not (ax):
        fig, ax = plt.subplots()
        if ISNOTEBOOK:
            fig.set_size_inches(min(int(len(x) / 1.8) + 1, 600), 6)
        ax.set_axisbelow(True)
        ax.yaxis.grid()
    param = {"color": get_color()[0], "alpha": 0.86}
    ax.bar(x, y, h, **updated_dict(param, style_kwds))
    ax.set_xlabel(vdf._alias)
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
        aggregate = f"{method}({of})"
        ax.set_ylabel(method)
    elif method.lower() == "count":
        ax.set_ylabel("Frequency")
    else:
        ax.set_ylabel(method)
    return ax


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
    colors = get_color()
    all_columns = vdf.pivot_table(
        columns, method=method, of=of, h=h, max_cardinality=max_cardinality, show=False,
    ).values
    all_columns = [[column] + all_columns[column] for column in all_columns]
    n, m = len(all_columns), len(all_columns[0])
    n_groups = m - 1
    bar_width = 0.5
    if not (ax):
        fig, ax = plt.subplots()
        if ISNOTEBOOK:
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
        ax.set_ylabel(f"{method}({of})")
    elif method.lower() == "count":
        ax.set_ylabel("Frequency")
    else:
        ax.set_ylabel(method)
    ax.legend(title=columns[1], loc="center left", bbox_to_anchor=[1, 0.5])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    return ax


def multiple_hist(
    vdf,
    columns: list,
    method: str = "density",
    of: str = "",
    h: float = 0,
    ax=None,
    **style_kwds,
):
    colors = get_color()
    if len(columns) > 5:
        raise ParameterError(
            "The number of column must be <= 5 to use 'multiple_hist' method"
        )
    else:
        if not (ax):
            fig, ax = plt.subplots()
            if ISNOTEBOOK:
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
                if vdf._vars["display"]["print_info"]:
                    warning_message = (
                        f"The Virtual Column {column} is not numerical."
                        " Its histogram will not be drawn."
                    )
                    warnings.warn(warning_message, Warning)
        ax.set_xlabel(", ".join(all_columns))
        if method.lower() == "density":
            ax.set_ylabel("Density")
        elif (
            method.lower() in ["avg", "min", "max", "sum", "mean"]
            or ("%" == method[-1])
        ) and (of):
            ax.set_ylabel(f"{method}({of})")
        elif method.lower() == "count":
            ax.set_ylabel("Frequency")
        else:
            ax.set_ylabel(method)
        ax.legend(title="columns", loc="center left", bbox_to_anchor=[1, 0.5])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax
