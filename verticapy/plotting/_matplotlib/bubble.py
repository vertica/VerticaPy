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
from typing import Literal, Optional, TYPE_CHECKING

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import verticapy._config.config as conf
from verticapy._typing import SQLColumns
from verticapy._utils._sql._sys import _executeSQL
from verticapy.errors import ParameterError

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame

from verticapy.plotting._matplotlib.base import MatplotlibBase


class BubblePlot(MatplotlibBase):
    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["bubble"]:
        return "bubble"

    @property
    def _compute_method(self) -> Literal["sample"]:
        return "sample"

    def draw(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        catcol: str = "",
        cmap_col: str = "",
        max_nb_points: int = 1000,
        bbox: list = [],
        img: str = "",
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a bubble plot using the Matplotlib API.
        """
        if (catcol) and (cmap_col):
            raise ParameterError(
                "Bubble Plot only accepts either a cmap column "
                "or a categorical column. It can not accept both."
            )
        if isinstance(columns, str):
            columns = [columns]
        if len(columns) == 2:
            columns += [1]
        if "color" in style_kwargs:
            colors = style_kwargs["color"]
        elif "colors" in style_kwargs:
            colors = style_kwargs["colors"]
        else:
            colors = self.get_colors()
        if isinstance(colors, str):
            colors = [colors]
        if not (catcol) and not (cmap_col):
            TableSample = max_nb_points / vdf.shape()[0]
            query_result = _executeSQL(
                query=f"""
                    SELECT 
                        /*+LABEL('plotting._matplotlib.bubble')*/ 
                        {columns[0]}, 
                        {columns[1]}, 
                        {columns[2]} 
                    FROM {vdf._genSQL(True)} 
                    WHERE __verticapy_split__ < {TableSample} 
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
                    1000
                    * (float(item[2]) - min_size)
                    / max((max_size - min_size), 1e-50)
                    for item in query_result
                ]
            column1, column2 = (
                [float(item[0]) for item in query_result],
                [float(item[1]) for item in query_result],
            )
            ax, fig = self._get_ax_fig(ax, size=(10, 6), set_axis_below=True, grid=True)
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
                column1, column2, s=size, **self._update_dict(param, style_kwargs),
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
                if conf._get_import_success("jupyter"):
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
                    aggr = vdf.agg(
                        columns=[columns[0], columns[1]], func=["min", "max"]
                    )
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
            TableSample = 0.1 if (count > 10000) else 0.9
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
                            FROM {vdf._genSQL(True)}
                            WHERE  __verticapy_split__ < {TableSample} 
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
                        column1,
                        column2,
                        s=size,
                        **self._update_dict(param, style_kwargs, idx),
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
                        FROM {vdf._genSQL(True)}
                        WHERE  __verticapy_split__ < {TableSample} 
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
                    "cmap": self.get_cmap(idx=0),
                    "edgecolors": "black",
                }
                im = ax.scatter(
                    column1,
                    column2,
                    c=column3,
                    s=size,
                    **self._update_dict(param, style_kwargs),
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
