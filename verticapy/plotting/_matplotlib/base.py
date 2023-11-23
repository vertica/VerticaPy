"""
Copyright  (c)  2018-2024 Open Text  or  one  of its
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
from typing import Optional, Union

from matplotlib.axes import Axes
from matplotlib.pyplot import Figure
import matplotlib.pyplot as plt

import verticapy._config.config as conf
from verticapy._utils._sql._format import format_type
from verticapy._typing import ArrayLike

from verticapy.plotting.base import PlottingBase


class MatplotlibBase(PlottingBase):
    @staticmethod
    def _get_ax_fig(
        ax,
        size: tuple[int, int] = (8, 6),
        set_axis_below: bool = True,
        grid: Union[str, bool] = True,
        dim: int = 2,
        style_kwargs: Optional[dict] = None,
    ) -> tuple[Axes, Figure]:
        style_kwargs = format_type(style_kwargs, dtype=dict)
        kwargs = copy.deepcopy(style_kwargs)
        if "figsize" in kwargs and isinstance(kwargs, tuple):
            size = kwargs["figsize"]
            del kwargs["size"]
        if "width" in kwargs:
            size = (kwargs["width"], size[1])
            del kwargs["width"]
        if "height" in kwargs:
            size = (size[0], kwargs["height"])
            del kwargs["height"]
        if not ax and dim == 3:
            if conf.get_import_success("IPython"):
                plt.figure(figsize=size)
            ax = plt.axes(projection="3d")
            return ax, plt, kwargs
        elif not ax:
            fig, ax = plt.subplots()
            if conf.get_import_success("IPython"):
                fig.set_size_inches(*size)
            if grid:
                if grid in ("x", "y"):
                    ax.grid(axis=grid)
                else:
                    ax.grid()
            ax.set_axisbelow(set_axis_below)
            return ax, fig, kwargs
        else:
            return ax, plt, kwargs

    @staticmethod
    def _get_matrix_fig_size(
        n: int,
    ) -> tuple[int, int]:
        if conf.get_import_success("IPython"):
            return min(1.5 * (n + 1), 500), min(1.5 * (n + 1), 500)
        else:
            return min(int((n + 1) / 1.1), 500), min(int((n + 1) / 1.1), 500)

    @staticmethod
    def _format_string(x: ArrayLike, th: int = 50) -> ArrayLike:
        res = copy.deepcopy(x)
        if isinstance(x[0], str):
            n = len(res)
            for i in range(n):
                if len(str(res[i])) > th:
                    res[i] = str(res[i][: th - 3]) + "..."
        return res
