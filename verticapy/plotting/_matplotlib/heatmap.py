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
from typing import Optional

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from verticapy._config.colors import get_cmap
import verticapy._config.config as conf
from verticapy._typing import ArrayLike

from verticapy.plotting.base import PlottingBase


class HeatMap(PlottingBase):
    def cmatrix(
        self,
        matrix: ArrayLike,
        columns_x: list[str],
        columns_y: list[str],
        n: int,
        m: int,
        vmax: float,
        vmin: float,
        title: str = "",
        colorbar: str = "",
        x_label: str = "",
        y_label: str = "",
        with_numbers: bool = True,
        mround: int = 3,
        is_vector: bool = False,
        inverse: bool = False,
        extent: list = [],
        is_pivot: bool = False,
        ax: Optional[Axes] = None,
        **style_kwds,
    ) -> Axes:
        """
        Draws a heatmap using the Matplotlib API.
        """
        if is_vector:
            vector = [elem for elem in matrix[1]]
            matrix_array = vector[1:]
            for i in range(len(matrix_array)):
                matrix_array[i] = round(float(matrix_array[i]), mround)
            matrix_array = [matrix_array]
            m, n = n, m
            x_label, y_label = y_label, x_label
            columns_x, columns_y = columns_y, columns_x
        else:
            matrix_array = [
                [
                    round(float(matrix[i][j]), mround)
                    if (matrix[i][j] != None and matrix[i][j] != "")
                    else float("nan")
                    for i in range(1, m + 1)
                ]
                for j in range(1, n + 1)
            ]
            if inverse:
                matrix_array.reverse()
                columns_x.reverse()
        if not (ax):
            fig, ax = plt.subplots()
            if (conf._get_import_success("jupyter") and not (inverse)) or is_pivot:
                fig.set_size_inches(min(m, 500), min(n, 500))
            else:
                fig.set_size_inches(8, 6)
        else:
            fig = plt
        param = {"cmap": get_cmap()[0], "interpolation": "nearest"}
        if ((vmax == 1) and vmin in [0, -1]) and not (extent):
            im = ax.imshow(
                matrix_array,
                vmax=vmax,
                vmin=vmin,
                **self.updated_dict(param, style_kwds),
            )
        else:
            try:
                im = ax.imshow(
                    matrix_array, extent=extent, **self.updated_dict(param, style_kwds)
                )
            except:
                im = ax.imshow(matrix_array, **self.updated_dict(param, style_kwds))
        fig.colorbar(im, ax=ax).set_label(colorbar)
        if not (extent):
            ax.set_yticks([i for i in range(0, n)])
            ax.set_xticks([i for i in range(0, m)])
            ax.set_xticklabels(columns_y, rotation=90)
            ax.set_yticklabels(columns_x, rotation=0)
        if with_numbers:
            for y_index in range(n):
                for x_index in range(m):
                    label = matrix_array[y_index][x_index]
                    ax.text(
                        x_index, y_index, label, color="black", ha="center", va="center"
                    )
        return ax
