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
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from verticapy._config.colors import get_color
from verticapy._config.config import ISNOTEBOOK

from verticapy.plotting._matplotlib.base import compute_plot_variables, updated_dict


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
        colors, n = get_color(), len(columns)
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
        if ISNOTEBOOK:
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
    colors = get_color()
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
            (method.lower() in ["min", "max"]) and (vdf._parent[of].category == "int")
        ):
            category = "int"
        else:
            category = None
        autopct = make_autopct(y, category)
    if not (rose):
        if not (ax):
            fig, ax = plt.subplots()
            if ISNOTEBOOK:
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
            handles,
            labels,
            title=vdf._alias,
            loc="center left",
            bbox_to_anchor=[1, 0.5],
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
            colors = [colors] + get_color()
        else:
            colors = colors + get_color()
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
            [Line2D([0], [0], color=color) for color in colors],
            z,
            bbox_to_anchor=[1.1, 0.5],
            loc="center left",
            title=vdf._alias,
            labelspacing=1,
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    return ax
