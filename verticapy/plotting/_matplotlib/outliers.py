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
import numpy as np

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from verticapy._config.colors import get_cmap, get_colors
from verticapy._typing import SQLColumns

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame

from verticapy.plotting._matplotlib.scatter import ScatterPlot


class OutliersPlot(ScatterPlot):
    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["outliers"]:
        return "outliers"

    def draw(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        color: str = "orange",
        outliers_color: str = "black",
        inliers_color: str = "white",
        inliers_border_color: str = "red",
        cmap: str = None,
        max_nb_points: int = 1000,
        threshold: float = 3.0,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws an outliers contour plot using the Matplotlib API.
        """
        if isinstance(columns, str):
            columns = [columns]
        if not (cmap):
            cmap = get_cmap(get_colors()[2])
        all_agg = vdf.agg(["avg", "std", "min", "max"], columns)
        xlist = np.linspace(all_agg["min"][0], all_agg["max"][0], 1000)
        ax, fig = self._get_ax_fig(ax, size=(8, 6), set_axis_below=False, grid=False)
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
                super().draw(
                    vdf_temp.search(f"ZSCORE {searchi[0]} {threshold}"),
                    [columns[0], "ZSCORE"],
                    max_nb_points=max_nb_points,
                    ax=ax,
                    color=searchi[1],
                    **style_kwargs,
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
            cp = ax.contourf(
                X, Y, Z, cmap=cmap, levels=np.linspace(threshold, Z.max(), 8)
            )
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
                super().draw(
                    vdf.search(s),
                    columns,
                    max_nb_points=max_nb_points,
                    ax=ax,
                    color=color,
                    **style_kwargs,
                )
        args = [[0], [0]]
        kwargs = {
            "marker": "o",
            "color": "black",
            "label": "Scatter",
            "markersize": 8,
        }
        ax.legend(
            [
                Line2D(*args, color=inliers_border_color, lw=4),
                Line2D(*args, **kwargs, markerfacecolor=inliers_color),
                Line2D(*args, **kwargs, markerfacecolor=outliers_color),
            ],
            ["threshold", "inliers", "outliers"],
            loc="center left",
            bbox_to_anchor=[1, 0.5],
            labelspacing=1,
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax
