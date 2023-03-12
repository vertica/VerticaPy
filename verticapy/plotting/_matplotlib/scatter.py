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
import copy, warnings
from typing import Literal, Optional, TYPE_CHECKING

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

import verticapy._config.config as conf
from verticapy._typing import SQLColumns
from verticapy._utils._sql._sys import _executeSQL

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame

from verticapy.plotting._matplotlib.base import MatplotlibBase


class ScatterMatrix(MatplotlibBase):
    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["scatter"]:
        return "scatter_matrix"

    def draw(
        self, vdf: "vDataFrame", columns: SQLColumns = [], **style_kwargs,
    ) -> Axes:
        """
        Draws a scatter matrix using the Matplotlib API.
        """
        if isinstance(columns, str):
            columns = [columns]
        columns = vdf._format_colnames(columns)
        if not (columns):
            columns = vdf.numcol()
        elif len(columns) == 1:
            return vdf[columns[0]].hist()
        n = len(columns)
        if conf._get_import_success("jupyter"):
            figsize = min(1.5 * (n + 1), 500), min(1.5 * (n + 1), 500)
            fig, axes = plt.subplots(nrows=n, ncols=n, figsize=figsize,)
        else:
            figsize = min(int((n + 1) / 1.1), 500), min(int((n + 1) / 1.1), 500)
            fig, axes = plt.subplots(nrows=n, ncols=n, figsize=figsize,)
        sample = vdf[columns].sample(n=1000).to_numpy()
        data = {"sample": sample}
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
                    self._compute_plot_params(
                        vdf[x], method="density", max_cardinality=1
                    )
                    data[f"{i}_{j}"] = copy.deepcopy(self.data)
                    params = {
                        "color": self.get_colors(d=style_kwargs, idx=0),
                        "edgecolor": "black",
                    }
                    if "edgecolor" in style_kwargs:
                        params["edgecolor"] = style_kwargs["edgecolor"]
                    axes[i, j].bar(
                        self.data["x"], self.data["y"], self.data["width"], **params
                    )
                else:
                    params = {
                        "color": get_colors(d=style_kwargs, idx=1),
                        "edgecolor": "black",
                        "alpha": 0.9,
                        "s": 40,
                        "marker": "o",
                    }
                    params = self._update_dict(params, style_kwargs, 1)
                    axes[i, j].scatter(
                        sample[:, j], sample[:, i], **params,
                    )
        self.data = data
        return axes


class ScatterPlot(MatplotlibBase):
    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["scatter"]:
        return "scatter"

    @property
    def _compute_method(self) -> Literal["sample"]:
        return "sample"

    @property
    def _markers(self):
        return ["^", "o", "+", "*", "h", "x", "D", "1"]

    def draw(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        catcol: str = "",
        max_cardinality: int = 3,
        cat_priority: list = [],
        with_others: bool = True,
        max_nb_points: int = 1000,
        bbox: list = [],
        img: str = "",
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a scatter plot using the Matplotlib API.
        """
        columns, catcol = vdf._format_colnames(
            columns, catcol, expected_nb_of_cols=[2, 3]
        )
        if isinstance(columns, str):
            columns = [columns]
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
        colors = self.get_colors()
        markers = self._markers * 10
        param = {
            "s": 50,
            "edgecolors": "black",
            "marker": "o",
        }
        if not (ax):
            if n == 2:
                ax, fig = self._get_ax_fig(
                    ax, size=(8, 6), set_axis_below=True, grid=True
                )
            else:
                if conf._get_import_success("jupyter"):
                    plt.figure(figsize=(8, 6))
                ax = plt.axes(projection="3d")
        all_scatter, others = [], []
        if not (catcol):
            TableSample = max_nb_points / vdf.shape()[0]
            limit = max_nb_points
        else:
            TableSample = 10 if (vdf.shape()[0] > 10000) else 90
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
            FROM {vdf._genSQL(True)}
            WHERE {{}}
                  {columns[0]} IS NOT NULL
              AND {columns[1]} IS NOT NULL
              {{}}
              AND __verticapy_split__ < {TableSample} 
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
            ax: Axes = ax,
        ) -> None:
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
            all_scatter += [
                ax.scatter(*args, **self._update_dict(param, style_kwargs, idx),)
            ]

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
