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
import copy
from typing import Union

from matplotlib.axes import Axes
from matplotlib.pyplot import Figure
import matplotlib.pyplot as plt

import verticapy._config.config as conf
from verticapy._typing import ArrayLike

from verticapy.plotting.base import PlottingBase


class MatplotlibBase(PlottingBase):
    @staticmethod
    def _get_ax_fig(
        ax,
        size: tuple[int, int] = (8, 6),
        set_axis_below: bool = True,
        grid: Union[str, bool] = True,
    ) -> tuple[Axes, Figure]:
        if not (ax):
            fig, ax = plt.subplots()
            if conf._get_import_success("jupyter"):
                fig.set_size_inches(*size)
            if grid:
                if grid in ("x", "y"):
                    ax.grid(axis=grid)
                else:
                    ax.grid()
            ax.set_axisbelow(set_axis_below)
            return ax, fig
        else:
            return ax, plt

    def _format_string(x: ArrayLike, th: int = 50) -> ArrayLike:
        res = copy.deepcopy(x)
        if isinstance(x[0], str):
            n = len(res)
            for i in range(n):
                if len(str(item)) > th:
                    res[i] = res[i][: th - 3] + "..."
        return res
