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
import math
from typing import Optional, TYPE_CHECKING
import numpy as np

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from verticapy._config.colors import get_colors
from verticapy._typing import SQLColumns
from verticapy.errors import ParameterError

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame

from verticapy.plotting.base import PlottingBase


class SpiderPlot(PlottingBase):
    def spider(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        method: str = "density",
        of: str = "",
        max_cardinality: tuple[int, int] = (6, 6),
        h: tuple[Optional[float], Optional[float]] = (None, None),
        ax: Optional[Axes] = None,
        **style_kwds,
    ) -> Axes:
        """
        Draws a spider plot using the Matplotlib API.
        """
        if isinstance(columns, str):
            columns = [columns]
        unique = vdf[columns[0]].nunique(True)
        if unique < 3:
            raise ParameterError(
                "The column used to draw the Spider Plot must "
                f"have at least 3 categories. Found {int(unique)}."
            )
        matrix, x_labels, y_labels = self._compute_pivot_table(
            vdf, columns, method=method, of=of, h=h, max_cardinality=max_cardinality,
        )[0:3]
        m = matrix.shape[0]
        angles = [i / float(m) * 2 * math.pi for i in range(m)]
        angles += angles[:1]
        fig = plt.figure()
        if not (ax):
            ax = fig.add_subplot(111, polar=True)
        spider_vals = np.array([])
        colors = get_colors()
        for i, category in enumerate(y_labels):
            if len(matrix.shape) == 1:
                values = np.concatenate((matrix, matrix[:1]))
            else:
                values = np.concatenate((matrix[:, i], matrix[:, i][:1]))
            spider_vals = np.concatenate((spider_vals, values))
            plt.xticks(angles[:-1], x_labels, color="grey", size=8)
            ax.set_rlabel_position(0)
            params = {"linewidth": 1, "linestyle": "solid", "color": colors[i]}
            params = self.updated_dict(params, style_kwds, i)
            args = [angles, values]
            ax.plot(*args, label=category, **params)
            ax.fill(*args, alpha=0.1, color=params["color"])
        y_ticks = [
            min(spider_vals),
            (max(spider_vals) + min(spider_vals)) / 2,
            max(spider_vals),
        ]
        ax.set_yticks(y_ticks)
        ax.set_rgrids(y_ticks, angle=180.0, fmt="%0.1f")
        ax.set_xlabel(columns[0])
        ax.set_ylabel(self._map_method(method, of)[0])
        if len(columns) > 1:
            ax.legend(
                title=columns[1], loc="center left", bbox_to_anchor=[1.1, 0.5],
            )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        return ax
