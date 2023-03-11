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

from verticapy._typing import SQLColumns
from verticapy.errors import ParameterError

from verticapy.core.tablesample.base import TableSample

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame

from verticapy.plotting._matplotlib.heatmap import HeatMap


class PivotTable(HeatMap):
    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["pivot"]:
        return "pivot"

    @property
    def _compute_method(self) -> Literal["2D"]:
        return "2D"

    def draw(
        self,
        show: bool = True,
        with_numbers: bool = True,
        ax: Optional[Axes] = None,
        return_ax: bool = False,
        extent: list = [],
        **style_kwargs,
    ) -> Axes:
        """
        Draws a pivot table using the Matplotlib API.
        """
        if show:
            ax = super().draw(
                self.data["matrix"],
                self.data["x_labels"],
                self.data["y_labels"],
                vmax=self.data["matrix"].max(),
                vmin=self.data["matrix"].min(),
                colorbar=self._map_method(method, of)[1],
                with_numbers=with_numbers,
                extent=extent,
                ax=ax,
                is_pivot=True,
                **style_kwargs,
            )
            ax.set_ylabel(self.layout["columns"][0])
            if len(self.layout["columns"]) > 1:
                ax.set_xlabel(self.layout["columns"][1])
            if return_ax:
                return ax
        values = {"index": self.data["x_labels"]}
        if len(self.data["matrix"].shape) == 1:
            values[self.layout["aggregate"]] = list(self.data["matrix"])
        else:
            for idx in range(self.data["matrix"].shape[1]):
                values[self.data["y_labels"][idx]] = list(self.data["matrix"][:, idx])
        return TableSample(values=values)
