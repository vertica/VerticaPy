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
from typing import Optional
import numpy as np

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from verticapy._config.colors import get_cmap
import verticapy._config.config as conf

from verticapy.plotting.base import PlottingBase


class HeatMap(PlottingBase):
    def color_matrix(
        self,
        matrix: np.ndarray,
        x_labels: list[str],
        y_labels: list[str],
        vmax: Optional[float] = None,
        vmin: Optional[float] = None,
        title: str = "",
        colorbar: str = "",
        with_numbers: bool = True,
        mround: int = 3,
        is_vector: bool = False,
        extent: list = [],
        is_pivot: bool = False,
        ax: Optional[Axes] = None,
        **style_kwds,
    ) -> Axes:
        """
        Draws a heatmap using the Matplotlib API.
        """
        if len(matrix.shape) == 1:
            n, m = matrix.shape[0], 1
        else:
            n, m = matrix.shape
        matrix_array = copy.deepcopy(matrix)
        x_l = list(x_labels)
        y_l = list(y_labels)
        if is_pivot:
            np.flip(matrix_array, axis=1)
            x_l.reverse()
        if not (ax):
            fig, ax = plt.subplots()
            if (conf._get_import_success("jupyter") and not (is_pivot)) or is_pivot:
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
            ax.set_xticklabels(y_l, rotation=90)
            ax.set_yticklabels(x_l, rotation=0)
        if with_numbers:
            matrix_array = matrix_array.round(mround)
            for y_index in range(n):
                for x_index in range(m):
                    label = matrix_array[y_index][x_index]
                    ax.text(
                        x_index, y_index, label, color="black", ha="center", va="center"
                    )
        return ax
