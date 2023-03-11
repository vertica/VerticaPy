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

from verticapy._config.colors import get_colors
from verticapy._typing import SQLColumns

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame

from verticapy.plotting._matplotlib.base import MatplotlibBase


class Histogram(MatplotlibBase):
    @property
    def _category(self) -> Literal["chart"]:
        return "chart"

    @property
    def _kind(self) -> Literal["hist"]:
        return "hist"

    def draw(
        self,
        vdf: "vDataFrame",
        columns: SQLColumns,
        method: str = "density",
        of: str = "",
        h: float = 0.0,
        ax: Optional[Axes] = None,
        **style_kwargs,
    ) -> Axes:
        """
        Draws a muli-histogram chart using the Matplotlib API.
        """
        if isinstance(columns, str):
            columns = [columns]
        colors = get_colors()
        if len(columns) > 5:
            raise ValueError(
                "The number of column must be <= 5 to use 'multiple_hist' method"
            )
        else:
            ax, fig = self._get_ax_fig(ax, size=(8, 6), set_axis_below=True, grid="y")
            alpha, all_columns, all_h = 1, [], []
            if h <= 0:
                for idx, column in enumerate(columns):
                    all_h += [vdf[column].numh()]
                h = min(all_h)
            data = {}
            for idx, column in enumerate(columns):
                if vdf[column].isnum():
                    self._compute_plot_params(
                        vdf[column], method=method, of=of, max_cardinality=1, h=h
                    )
                    params = {"color": colors[idx % len(colors)]}
                    params = self._update_dict(params, style_kwargs, idx)
                    plt.bar(
                        self.data["x"],
                        self.data["y"],
                        self.data["width"],
                        label=column,
                        alpha=alpha,
                        **params,
                    )
                    alpha -= 0.2
                    all_columns += [columns[idx]]
                    data[column] = copy.deepcopy(self.data)
                else:
                    if vdf._vars["display"]["print_info"]:
                        warning_message = (
                            f"The Virtual Column {column} is not numerical."
                            " Its histogram will not be drawn."
                        )
                        warnings.warn(warning_message, Warning)
            ax.set_xlabel(", ".join(all_columns))
            ax.set_ylabel(self._map_method(method, of)[0])
            ax.legend(title="columns", loc="center left", bbox_to_anchor=[1, 0.5])
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            self.data = data
            return ax
