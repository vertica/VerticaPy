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
from typing import Literal, Optional
import scipy.spatial as scipy_st

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from verticapy._typing import ArrayLike, SQLColumns
from verticapy._utils._sql._sys import _executeSQL

from verticapy.plotting._matplotlib.base import MatplotlibBase


class VoronoiPlot(MatplotlibBase):
    @property
    def _category(self) -> Literal["plot"]:
        return "plot"

    @property
    def _kind(self) -> Literal["voronoi"]:
        return "voronoi"

    def draw(
        self,
        clusters: ArrayLike,
        columns: SQLColumns,
        input_relation: str,
        max_nb_points: int = 1000,
        plot_crosses: bool = True,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a KMeans Voronoi plot using the Matplotlib API.
        """
        if isinstance(columns, str):
            columns = [columns]
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
        if hasattr(clusters, "tolist"):
            clusters = clusters.tolist()
        v = scipy_st.Voronoi(clusters + dummies_point)
        param = {"show_vertices": False}
        scipy_st.voronoi_plot_2d(v, ax=ax, **self._update_dict(param, style_kwargs))
        if not (ax):
            ax = plt
            ax.xlabel(columns[0])
            ax.ylabel(columns[1])
        colors = self.get_colors()
        for idx, region in enumerate(v.regions):
            if not -1 in region:
                polygon = [v.vertices[i] for i in region]
                if "color" in style_kwargs:
                    if isinstance(style_kwargs["color"], str):
                        color = style_kwargs["color"]
                    else:
                        color = style_kwargs["color"][idx % len(style_kwargs["color"])]
                else:
                    color = colors[idx % len(colors)]
                ax.fill(*zip(*polygon), alpha=0.4, color=color)
        ax.plot([elem[0] for elem in clusters], [elem[1] for elem in clusters], "ko")
        ax.xlim(min_x - 0.05 * (max_x - min_x), max_x + 0.05 * (max_x - min_x))
        ax.ylim(min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y))
        if max_nb_points > 0:
            all_points = _executeSQL(
                query=f"""
                    SELECT 
                        /*+LABEL('plotting._matplotlib.voronoi_plot')*/ 
                        {columns[0]}, 
                        {columns[1]} 
                    FROM {input_relation} 
                    WHERE {columns[0]} IS NOT NULL 
                      AND {columns[1]} IS NOT NULL 
                    ORDER BY RANDOM() 
                    LIMIT {int(max_nb_points)}""",
                method="fetchall",
                print_time_sql=False,
            )
            x, y = (
                [float(c[0]) for c in all_points],
                [float(c[1]) for c in all_points],
            )
            ax.scatter(
                x, y, color="black", s=10, alpha=1, zorder=3,
            )
            if plot_crosses:
                ax.scatter(
                    [c[0] for c in clusters],
                    [c[1] for c in clusters],
                    color="white",
                    s=200,
                    linewidths=5,
                    alpha=1,
                    zorder=4,
                    marker="x",
                )
        return ax
