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
from typing import Optional, TYPE_CHECKING

from matplotlib.axes import Axes

from verticapy._typing import SQLColumns
from verticapy.errors import ParameterError

from verticapy.core.tablesample.base import TableSample

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame

from verticapy.plotting._matplotlib.heatmap import HeatMap


class PivotTable(HeatMap):
    def pivot_table(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        method: str = "count",
        of: str = "",
        h: tuple[Optional[float], Optional[float]] = (None, None),
        max_cardinality: tuple[int, int] = (20, 20),
        fill_none: float = 0.0,
        show: bool = True,
        with_numbers: bool = True,
        ax: Optional[Axes] = None,
        return_ax: bool = False,
        extent: list = [],
        **style_kwds,
    ) -> Axes:
        """
        Draws a pivot table using the Matplotlib API.
        """
        matrix, x_labels, y_labels, vmin, vmax, aggregate = self._compute_pivot_table(
            vdf=vdf,
            columns=columns,
            method=method,
            of=of,
            h=h,
            max_cardinality=max_cardinality,
            fill_none=fill_none,
        )
        if show:
            ax = self.color_matrix(
                matrix,
                x_labels,
                y_labels,
                vmax=vmax,
                vmin=vmin,
                colorbar=aggregate,
                with_numbers=with_numbers,
                extent=extent,
                ax=ax,
                is_pivot=True,
                **style_kwds,
            )
            ax.set_ylabel(columns[0])
            if len(columns) > 1:
                ax.set_xlabel(columns[1])
            if return_ax:
                return ax
        values = {"index": x_labels}
        if len(matrix.shape) == 1:
            values[aggregate] = list(matrix)
        else:
            for idx in range(matrix.shape[1]):
                values[y_labels[idx]] = list(matrix[:, idx])
        return TableSample(values=values)
